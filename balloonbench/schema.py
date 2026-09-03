"""The BalloonBench ground-truth contract.

Pydantic v2 models mirroring ``schema/characteristic.schema.json`` (SPEC.md section 4),
plus the structural rules JSON Schema cannot express.

This module is the contract every other module depends on. Do not change it without
updating every consumer and re-running ``pytest tests/``.

The rules enforced here, and the reason each exists:

R1  Balloon ids are unique and contiguous from 1. An inspection plan with a gap in the
    balloon sequence is not a valid inspection plan.
R2  A geometric tolerance that locates or orients a feature is meaningless without
    something to locate or orient it against, so position, perpendicularity, parallelism,
    angularity, concentricity, symmetry and both runouts require at least one datum
    reference. Form tolerances describe a surface against itself, so flatness,
    straightness, circularity and cylindricity must carry none.
R3  A material condition modifier shifts the tolerance zone as a feature departs from its
    maximum or least material size. A surface with no size has no such departure, so the
    modifier is rejected on form tolerances, on profile of a surface, and on runout.
R4  ``upper_tol`` and ``lower_tol`` always store signed deviations from ``nominal``, even
    when ``tol_style`` is ``limit``. Converting to limit form is the renderer's job.
R5  A basic dimension is theoretically exact, so ``is_basic`` implies both deviations are
    zero.
R6  Every bbox lies inside the rendered image and has positive extent, otherwise it cannot
    be matched against a prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "characteristic.schema.json"

# --- vocabularies -------------------------------------------------------------------

Source = Literal["synthetic", "real"]
Units = Literal["mm", "inch"]
Projection = Literal["first_angle", "third_angle"]
SheetSize = Literal["A4", "A3", "A2", "A1", "A0"]
Kind = Literal["dimension", "geometric_tolerance", "note", "surface_finish", "thread"]
DimType = Literal["linear", "diameter", "radius", "angular", "chamfer", "thread"]
TolStyle = Literal["bilateral", "unilateral", "limit", "fit", "general", "basic"]
MaterialModifier = Literal["MMC", "LMC"]
DatumFeatureType = Literal[
    "planar_face", "cylindrical_feature", "axis", "width", "spherical"
]
GtolSymbol = Literal[
    "flatness",
    "straightness",
    "circularity",
    "cylindricity",
    "perpendicularity",
    "parallelism",
    "angularity",
    "position",
    "concentricity",
    "symmetry",
    "circular_runout",
    "total_runout",
    "profile_surface",
]

#: Form tolerances. They describe a surface against itself (R2).
FORM_SYMBOLS: frozenset[str] = frozenset(
    {"flatness", "straightness", "circularity", "cylindricity"}
)

#: Symbols that are meaningless without a datum reference frame (R2).
DATUM_REQUIRED_SYMBOLS: frozenset[str] = frozenset(
    {
        "position",
        "perpendicularity",
        "parallelism",
        "angularity",
        "concentricity",
        "symmetry",
        "circular_runout",
        "total_runout",
    }
)

#: Symbols that never apply to a feature of size, so never carry a modifier (R3).
NO_MODIFIER_SYMBOLS: frozenset[str] = FORM_SYMBOLS | {
    "profile_surface",
    "circular_runout",
    "total_runout",
}

#: Datum feature types that have a size, and so may carry a modifier on a datum
#: reference (R3, applied to the reference rather than the tolerance).
FEATURE_OF_SIZE_DATUMS: frozenset[str] = frozenset(
    {"cylindrical_feature", "axis", "width", "spherical"}
)

DatumLabel = Annotated[str, Field(pattern=r"^[A-Z]{1,2}$")]
BBox = Annotated[list[float], Field(min_length=4, max_length=4)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- leaf models --------------------------------------------------------------------


class Sheet(_Strict):
    size: SheetSize
    scale: str = Field(pattern=r"^[0-9]+:[0-9]+$")


class TitleBlock(_Strict):
    part_number: str = Field(min_length=1)
    revision: str = ""
    material: str | None = None
    general_tolerance: str | None = None
    surface_finish_default: str | None = None


class Datum(_Strict):
    label: DatumLabel
    feature_type: DatumFeatureType
    view: str = Field(min_length=1)
    bbox: BBox
    geometry_ref: str | None = None

    @model_validator(mode="after")
    def _bbox_has_extent(self) -> Datum:
        _check_bbox_extent(self.bbox, f"datum {self.label}")
        return self


class DatumRef(_Strict):
    label: DatumLabel
    modifier: MaterialModifier | None = None


class Provenance(_Strict):
    generator_version: str
    generator_seed: int | None = None
    degradation_profile: str | None = None
    house_style: str | None = None
    labeler: str | None = None


# --- characteristic -----------------------------------------------------------------


class Characteristic(_Strict):
    id: int = Field(ge=1)
    kind: Kind
    view: str = Field(min_length=1)
    bbox: BBox
    leader_target_bbox: BBox | None = None

    # dimension fields
    dim_type: DimType | None = None
    nominal: float | None = None
    upper_tol: float | None = None
    lower_tol: float | None = None
    tol_style: TolStyle | None = None
    fit_class: str | None = None
    is_basic: bool = False
    is_reference: bool = False
    is_critical: bool = False

    # geometric tolerance fields
    gtol_symbol: GtolSymbol | None = None
    gtol_value: float | None = Field(default=None, gt=0)
    gtol_zone: Literal["diametral", "linear"] | None = None
    material_modifier: MaterialModifier | None = None
    datum_refs: list[DatumRef] = Field(default_factory=list)

    # shared
    raw_text: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Characteristic:
        where = f"characteristic {self.id}"
        _check_bbox_extent(self.bbox, where)
        if self.leader_target_bbox is not None:
            _check_bbox_extent(self.leader_target_bbox, f"{where} leader target")

        if self.kind == "dimension":
            if self.dim_type is None:
                raise ValueError(f"{where}: a dimension requires dim_type")
            if self.nominal is None:
                raise ValueError(f"{where}: a dimension requires nominal")
            if self.gtol_symbol is not None:
                raise ValueError(f"{where}: a dimension must not carry gtol_symbol")

        if self.kind == "geometric_tolerance":
            if self.gtol_symbol is None:
                raise ValueError(
                    f"{where}: a geometric tolerance requires gtol_symbol"
                )
            if self.gtol_value is None:
                raise ValueError(f"{where}: a geometric tolerance requires gtol_value")

            # R2
            n_refs = len(self.datum_refs)
            if self.gtol_symbol in DATUM_REQUIRED_SYMBOLS and n_refs == 0:
                raise ValueError(
                    f"{where}: {self.gtol_symbol} locates or orients a feature and "
                    f"requires at least one datum reference"
                )
            if self.gtol_symbol in FORM_SYMBOLS and n_refs > 0:
                raise ValueError(
                    f"{where}: {self.gtol_symbol} is a form tolerance and must not "
                    f"reference a datum"
                )

            # R3
            if (
                self.material_modifier is not None
                and self.gtol_symbol in NO_MODIFIER_SYMBOLS
            ):
                raise ValueError(
                    f"{where}: {self.gtol_symbol} does not apply to a feature of size, "
                    f"so it cannot carry a {self.material_modifier} modifier"
                )

            labels = [r.label for r in self.datum_refs]
            if len(labels) != len(set(labels)):
                raise ValueError(f"{where}: repeated datum reference in {labels}")
        else:
            if self.datum_refs:
                raise ValueError(
                    f"{where}: datum_refs are only meaningful on a geometric tolerance"
                )
            if self.material_modifier is not None:
                raise ValueError(
                    f"{where}: material_modifier is only meaningful on a geometric "
                    f"tolerance"
                )

        # R5
        if self.is_basic and (self.upper_tol or self.lower_tol):
            raise ValueError(
                f"{where}: a basic dimension is theoretically exact, so upper_tol and "
                f"lower_tol must both be 0"
            )

        # R4 is a storage convention, checked only for orientation of the deviations.
        if (
            self.upper_tol is not None
            and self.lower_tol is not None
            and self.upper_tol < self.lower_tol
        ):
            raise ValueError(
                f"{where}: upper_tol {self.upper_tol} is below lower_tol "
                f"{self.lower_tol}; both are signed deviations from nominal"
            )

        if self.fit_class is not None and self.tol_style != "fit":
            raise ValueError(
                f"{where}: fit_class {self.fit_class!r} requires tol_style 'fit'"
            )

        return self


# --- drawing ------------------------------------------------------------------------


class Drawing(_Strict):
    drawing_id: str = Field(min_length=1)
    source: Source
    part_ref: str | None = None
    units: Units
    projection: Projection
    image_size: tuple[int, int] | None = None
    sheet: Sheet
    title_block: TitleBlock
    datums: list[Datum] = Field(default_factory=list)
    characteristics: list[Characteristic] = Field(min_length=1)
    provenance: Provenance

    @model_validator(mode="after")
    def _validate(self) -> Drawing:
        # R1
        ids = [c.id for c in self.characteristics]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate balloon ids: {dupes}")
        if sorted(ids) != list(range(1, len(ids) + 1)):
            raise ValueError(
                f"balloon ids must be contiguous from 1; got {sorted(ids)}"
            )

        labels = [d.label for d in self.datums]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate datum labels: {sorted(labels)}")
        known = set(labels)

        by_label = {d.label: d for d in self.datums}
        for c in self.characteristics:
            for ref in c.datum_refs:
                if ref.label not in known:
                    raise ValueError(
                        f"characteristic {c.id} references datum {ref.label!r}, which "
                        f"is not declared in datums {sorted(known)}"
                    )
                # R3 applied to the reference itself.
                if (
                    ref.modifier is not None
                    and by_label[ref.label].feature_type not in FEATURE_OF_SIZE_DATUMS
                ):
                    raise ValueError(
                        f"characteristic {c.id}: datum {ref.label} is a "
                        f"{by_label[ref.label].feature_type}, which has no size, so it "
                        f"cannot carry a {ref.modifier} modifier"
                    )

        # R6
        if self.image_size is not None:
            w, h = self.image_size
            for d in self.datums:
                _check_bbox_inside(d.bbox, w, h, f"datum {d.label}")
            for c in self.characteristics:
                _check_bbox_inside(c.bbox, w, h, f"characteristic {c.id}")
                if c.leader_target_bbox is not None:
                    _check_bbox_inside(
                        c.leader_target_bbox, w, h, f"characteristic {c.id} leader target"
                    )
        return self

    # --- convenience ---------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> Drawing:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(
            self.model_dump_json(indent=indent, exclude_none=False) + "\n",
            encoding="utf-8",
        )


# --- helpers ------------------------------------------------------------------------


def _check_bbox_extent(bbox: list[float], where: str) -> None:
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"{where}: bbox {bbox} has non-positive extent; expected x1 > x0 and y1 > y0"
        )


def _check_bbox_inside(bbox: list[float], w: int, h: int, where: str) -> None:
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise ValueError(
            f"{where}: bbox {bbox} falls outside the {w}x{h} image"
        )


def load_json_schema() -> dict:
    """The frozen JSON Schema document, for cross-checking the pydantic models."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
