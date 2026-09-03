"""Generate one complete drawing: sheet, artifacts, and validated ground truth.

The single entry point a caller needs. Given a family and a seed it produces the artifact
set SPEC.md section 7.4 asks for -- vector PDF, DXF, PNG, STEP, ground-truth JSON and a QA
overlay -- and it validates the JSON against the frozen schema before returning, so a
drawing that cannot be described correctly is never written at all.

Two ordering rules matter here and are easy to get backwards.

**The PNG must exist before ``image_size`` is set.** The schema's R6 checks every box lies
inside the image, and that check is only meaningful against the image that was actually
produced. So the PDF is rasterised first, its real size is read back, and the size is
asserted against what the layout predicted rather than assumed to match.

**Datums are established before characteristics reference them.** A datum whose symbol could
not be placed does not exist on the sheet, so every characteristic referring to it is
dropped too. Keeping the reference would produce ground truth asserting a datum a reader
cannot find, which is worse than a slightly sparser drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from balloonbench import __version__
from balloonbench.drawgen.annotate import Annotation, place_all
from balloonbench.drawgen.render import (
    build_primitives,
    render_dxf,
    render_overlay,
    render_pdf,
    render_png,
)
from balloonbench.drawgen.schemes import (
    plan_drawing,
    sheet_decorations,
)
from balloonbench.drawgen.styles import HouseStyle, sample_style
from balloonbench.drawgen.titleblock import (
    TitleBlockData,
    border_primitives,
    title_block_primitives,
)
from balloonbench.drawgen.views import PROJECTION_ANGLES, layout_for_part
from balloonbench.partgen.registry import build_part
from balloonbench.partgen.types import BuiltPart
from balloonbench.schema import Drawing

__all__ = ["DrawingBundle", "generate_drawing"]

DEFAULT_DPI = 300.0

#: Materials and their general tolerance notes. The note is what governs every dimension the
#: sheet leaves untoleranced, so it has to be plausible for the material and the process:
#: a cast body is not held to the same general tolerance as a machined plate.
_MATERIALS: dict[str, tuple[str, ...]] = {
    "flange": ("EN-GJL-250", "S355JR", "AISI 304"),
    "shaft": ("42CrMo4", "C45E", "AISI 316"),
    "plate_bracket": ("S275JR", "AL 6082-T6", "AISI 304"),
    "housing": ("EN-GJL-250", "AL 6082-T6", "EN-GJS-500-7"),
    "valve_body": ("EN-GJS-400-15", "ASTM A216 WCB", "EN-GJL-250"),
}
_GENERAL_TOLERANCES: tuple[str, ...] = (
    "ISO 2768-mK", "ISO 2768-fH", "ISO 2768-m", "ISO 2768-mH",
)
_FINISHES: tuple[str | None, ...] = (None, "Ra 3.2", "Ra 6.3", "Ra 1.6")
_SHEETS: tuple[str, ...] = ("A4", "A3", "A3", "A2")


@dataclass
class DrawingBundle:
    """Everything one generated drawing consists of."""

    drawing_id: str
    part: BuiltPart
    drawing: Drawing
    paths: dict[str, Path]
    style: HouseStyle
    dropped: int

    @property
    def characteristics(self) -> int:
        return len(self.drawing.characteristics)


def _drawing_seed(family: str, seed: int) -> int:
    """A per-family generator seed for every *presentation* choice.

    Seeding on the integer alone made the style, sheet size and projection convention a
    pure function of the seed, so seed 7 was a legacy-shop A2 third-angle sheet for all five
    families at once. A run over seeds 0..199 then produced a dataset in which family and
    house style were correlated, and any per-style finding would have been confounded by
    which parts happened to land in that style.

    ``zlib.crc32`` rather than ``hash``: the built-in is salted per interpreter, so it would
    make a drawing irreproducible between processes -- exactly the property this seed exists
    to provide.
    """
    import zlib

    return (seed * 1_000_003 + zlib.crc32(family.encode("utf-8"))) % (2**32)


def _title_block(
    family: str, seed: int, style: HouseStyle, rng: np.random.Generator, scale_text: str,
    sheet: str, projection: str,
) -> TitleBlockData:
    part_number = f"BB-{family[:3].upper()}-{seed:05d}"
    return TitleBlockData(
        part_number=part_number,
        revision=str(rng.choice(list("ABC"))),
        material=str(rng.choice(_MATERIALS.get(family, ("S275JR",)))),
        general_tolerance=str(rng.choice(_GENERAL_TOLERANCES)),
        surface_finish_default=(
            None if (f := rng.choice(len(_FINISHES))) == 0 else _FINISHES[int(f)]
        ),
        title=family.replace("_", " ").upper(),
        scale_text=scale_text,
        projection=projection,
        sheet_size=sheet,
        units=style.units,
    )


def _characteristic(
    index: int,
    ann: Annotation,
    layout,
    dpi: float,
) -> dict[str, Any]:
    assert ann.bbox is not None
    payload = dict(ann.payload)
    payload.setdefault("kind", ann.kind)
    payload["id"] = index
    payload["view"] = ann.view
    payload["bbox"] = list(layout.bbox_to_pixel(ann.bbox, dpi))
    if ann.target_box is not None:
        payload["leader_target_bbox"] = list(layout.bbox_to_pixel(ann.target_box, dpi))
    return payload


def generate_drawing(
    family: str,
    seed: int,
    out_dir: Path,
    *,
    dpi: float = DEFAULT_DPI,
    style: HouseStyle | None = None,
    projection: str | None = None,
    sheet: str | None = None,
    write_artifacts: bool = True,
) -> DrawingBundle:
    """Build a part, draw it, and return the validated bundle.

    Every random choice -- house style, projection convention, sheet size, title-block
    fields, and each dimension's tolerance presentation -- is drawn from a generator seeded
    from ``seed`` alone, so ``(family, seed)`` reproduces the drawing exactly. That is the
    reproducibility contract CLAUDE.md states, and it is why the style is sampled here
    rather than passed in by a caller that might vary independently.
    """
    rng = np.random.default_rng(_drawing_seed(family, seed))
    part = build_part(family, seed)

    style = style or sample_style(rng)
    projection = projection or str(rng.choice(PROJECTION_ANGLES))
    sheet = sheet or str(rng.choice(_SHEETS))

    layout = layout_for_part(part.shape, family, sheet=sheet, projection=projection)
    datum_records, annotations = plan_drawing(part, layout, style, rng)

    decorations = sheet_decorations(part, layout, style)
    everything: list[Annotation] = [d.annotation for d in datum_records] + annotations
    placed = place_all(everything, layout, style, extra_ink=decorations)
    placed_set = {id(a) for a in placed}

    live_datums = [d for d in datum_records if id(d.annotation) in placed_set]
    live_labels = {d.label for d in live_datums}

    callouts: list[Annotation] = []
    dropped = len(everything) - len(placed)
    for ann in placed:
        if ann.kind == "datum":
            continue
        refs = ann.payload.get("datum_refs") or []
        if any(r["label"] not in live_labels for r in refs):
            # The datum this frame points to never made it onto the sheet.
            dropped += 1
            continue
        callouts.append(ann)

    # Placement order is by importance, which is a generation detail. Sorting by position
    # gives balloon ids that run down and across the sheet the way a person numbers them,
    # and makes the ids stable against any future change to importance weights.
    callouts.sort(key=lambda a: (-(a.bbox[3]), a.bbox[0]))

    drawing_id = f"{family}-{seed:05d}"
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}

    tb = _title_block(family, seed, style, rng, layout.scale_text, sheet, projection)
    chrome = border_primitives(layout, style) + title_block_primitives(layout, style, tb)
    primitives = build_primitives(layout, style, placed, chrome, decorations)

    pdf_path = out_dir / f"{drawing_id}.pdf"
    png_path = out_dir / f"{drawing_id}_{int(dpi)}.png"
    render_pdf(pdf_path, layout, style, primitives)
    size = render_png(pdf_path, png_path, dpi)
    predicted = layout.pixel_size(dpi)
    if abs(size[0] - predicted[0]) > 1 or abs(size[1] - predicted[1]) > 1:
        raise RuntimeError(
            f"rasteriser produced {size} but the layout predicts {predicted}; the two "
            f"must agree or every bounding box is wrong"
        )
    paths["pdf"] = pdf_path
    paths["png"] = png_path

    if write_artifacts:
        paths["dxf"] = render_dxf(out_dir / f"{drawing_id}.dxf", layout, style, primitives)
        if part.step_path is not None:
            paths["step"] = part.step_path

    characteristics = [
        _characteristic(i, ann, layout, dpi) for i, ann in enumerate(callouts, start=1)
    ]
    datums = [
        {
            "label": d.label,
            "feature_type": d.feature_type,
            "view": d.view,
            "bbox": list(layout.bbox_to_pixel(d.annotation.bbox, dpi)),
            "geometry_ref": d.geometry_ref,
        }
        for d in live_datums
    ]

    drawing = Drawing.model_validate(
        {
            "drawing_id": drawing_id,
            "source": "synthetic",
            "part_ref": str(part.step_path) if part.step_path else None,
            "units": style.units,
            "projection": projection,
            "image_size": list(size),
            "sheet": {"size": sheet, "scale": layout.scale_text},
            "title_block": {
                "part_number": tb.part_number,
                "revision": tb.revision,
                "material": tb.material,
                "general_tolerance": tb.general_tolerance,
                "surface_finish_default": tb.surface_finish_default,
            },
            "datums": datums,
            "characteristics": characteristics,
            "provenance": {
                "generator_version": __version__,
                "generator_seed": seed,
                "degradation_profile": None,
                "house_style": style.name,
                "labeler": None,
            },
        }
    )

    if write_artifacts:
        json_path = out_dir / f"{drawing_id}.json"
        drawing.to_json(json_path)
        paths["json"] = json_path

        overlay_boxes = [
            (d["feature_type"] and "datum", tuple(d["bbox"]), d["label"]) for d in datums
        ] + [
            (c["kind"], tuple(c["bbox"]), str(c["id"])) for c in characteristics
        ]
        paths["overlay"] = render_overlay(
            png_path, out_dir / f"{drawing_id}_overlay.png", overlay_boxes
        )

    return DrawingBundle(
        drawing_id=drawing_id,
        part=part,
        drawing=drawing,
        paths=paths,
        style=style,
        dropped=dropped,
    )
