"""The contract between geometry and annotation.

A BalloonBench part is not just a solid. It is a solid **plus** a description of what the
solid *is*, semantically: this cylinder is the main bore, those six are a bolt pattern,
that plane is the mounting face. Without that description, dimensioning the solid would be
the open research problem SPEC.md section 6 warns about. With it, ground truth is exact by
construction, because we annotate features we named rather than features we recovered.

Two ideas carry the module.

**Stable face identity.** Every face gets an id like ``face_0007``. Ids must be identical
across a rebuild from the same seed, or nothing downstream can rely on them -- the
annotator would point at a different face than the verifier checks. OCCT's own traversal
order is not a contract, so ids are assigned from a sorted geometric key rather than from
traversal order. See :class:`FaceIndex`.

**Features of size.** GD&T draws a hard line between a feature that has a size (a bore, a
shaft, a slot width -- something with two opposed points you could caliper) and a surface
that does not (a plane, a profile). Only the former can carry a material condition
modifier. :class:`SemanticFeature` records which it is, so the schema rule R3 is satisfied
by construction rather than checked after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape

__all__ = [
    "FEATURES_OF_SIZE",
    "BuiltPart",
    "FaceIndex",
    "FaceInfo",
    "PartFamily",
    "SemanticFeature",
    "UnbuildableParams",
]

#: Feature kinds that have a size in the GD&T sense: two opposed points a caliper could
#: span. Only these may carry a material condition modifier (schema rule R3).
FEATURES_OF_SIZE: frozenset[str] = frozenset(
    {
        "through_hole",
        "blind_hole",
        "counterbore",
        "boss",
        "slot",
        "thread",
        "cylindrical_face",
        "groove",
        "keyway",
    }
)

#: Feature kinds that are surfaces, not features of size.
SURFACE_FEATURES: frozenset[str] = frozenset({"planar_face", "chamfer", "fillet"})


class UnbuildableParams(ValueError):
    """Sampled parameters that do not describe a manufacturable part.

    SPEC.md section 6.3 requires reject-and-resample rather than clamping: silently
    clamping a wall thickness back to its minimum produces a part whose dimensions no
    longer match the distribution we claim to sample from, and it hides the fact that the
    sampler is generating invalid combinations.
    """


# --- face identity ------------------------------------------------------------------

#: OCCT surface types we classify, mapped to a short stable name used in the sort key.
_SURFACE_NAMES: dict[Any, str] = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "plane",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "cylinder",
    GeomAbs_SurfaceType.GeomAbs_Cone: "cone",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "sphere",
    GeomAbs_SurfaceType.GeomAbs_Torus: "torus",
    GeomAbs_SurfaceType.GeomAbs_BezierSurface: "bezier",
    GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "offset",
    GeomAbs_SurfaceType.GeomAbs_OtherSurface: "other",
}

#: Quantisation applied to the sort key. Coarse enough that floating-point noise between
#: two builds cannot reorder two faces, fine enough that genuinely distinct faces on the
#: parts we generate never collide.
_KEY_DECIMALS = 4


@dataclass(frozen=True)
class FaceInfo:
    """One face, classified and measured.

    ``radius``, ``axis`` and ``normal`` are populated only where they mean something: a
    radius for a cylinder or cone, an axis for a surface of revolution, a normal for a
    plane. The verifier's B-rep index (SPEC.md section 12.1) is built on the same
    measurements, so they live here rather than being recomputed there.
    """

    fid: str
    surface: str
    area: float
    centroid: tuple[float, float, float]
    radius: float | None = None
    axis: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    normal: tuple[float, float, float] | None = None

    @property
    def is_cylindrical(self) -> bool:
        return self.surface == "cylinder"

    @property
    def is_planar(self) -> bool:
        return self.surface == "plane"


def _round3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    # Normalising -0.0 to 0.0 keeps the sort key stable: the two compare equal but a
    # tuple containing one is not equal to a tuple containing the other under ==.
    return tuple(round(c, _KEY_DECIMALS) + 0.0 for c in v)  # type: ignore[return-value]


class FaceIndex:
    """Deterministic ids for the faces of a solid.

    Ids are assigned by sorting faces on a purely geometric key -- surface type, then
    area, then centroid -- and numbering the result. Two builds from the same parameters
    produce the same geometry, so they produce the same sort, so they produce the same
    ids. Traversal order, by contrast, is an implementation detail of OCCT that can change
    with a version bump or with an unrelated modelling-order change, which would silently
    repoint every annotation on every previously generated drawing.

    The cost is that a genuinely symmetric part can have faces whose keys tie -- the six
    bolt holes of a flange, for instance, have equal area and mirrored centroids. Ties are
    broken by the full centroid ordering, which is stable but arbitrary; that is fine,
    because a symmetric part's bolt holes are interchangeable by definition, and the
    semantic feature records the set rather than relying on which one got which id.
    """

    def __init__(self, shape: TopoDS_Shape) -> None:
        self._faces: dict[str, TopoDS_Face] = {}
        self._info: dict[str, FaceInfo] = {}
        self._build(shape)

    # -- construction ---------------------------------------------------------------

    def _build(self, shape: TopoDS_Shape) -> None:
        measured: list[tuple[tuple, TopoDS_Face, dict]] = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        seen: list[TopoDS_Face] = []
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            explorer.Next()
            # TopExp_Explorer visits a face once per containing shell, so the same face
            # can appear twice on a compound. IsSame compares topology, not geometry.
            if any(face.IsSame(other) for other in seen):
                continue
            seen.append(face)
            props = self._measure(face)
            key = (
                props["surface"],
                round(props["area"], _KEY_DECIMALS),
                *_round3(props["centroid"]),
                round(props["radius"] if props["radius"] is not None else -1.0, _KEY_DECIMALS),
            )
            measured.append((key, face, props))

        for n, (_key, face, props) in enumerate(sorted(measured, key=lambda t: t[0])):
            fid = f"face_{n:04d}"
            self._faces[fid] = face
            self._info[fid] = FaceInfo(fid=fid, **props)

    @staticmethod
    def _measure(face: TopoDS_Face) -> dict:
        adaptor = BRepAdaptor_Surface(face)
        surface = _SURFACE_NAMES.get(adaptor.GetType(), "other")

        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        com = props.CentreOfMass()
        centroid = (com.X(), com.Y(), com.Z())

        radius: float | None = None
        axis: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        normal: tuple[float, float, float] | None = None

        if surface == "cylinder":
            cyl = adaptor.Cylinder()
            radius = cyl.Radius()
            ax = cyl.Axis()
            axis = (
                (ax.Location().X(), ax.Location().Y(), ax.Location().Z()),
                (ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()),
            )
        elif surface == "cone":
            cone = adaptor.Cone()
            radius = cone.RefRadius()
            ax = cone.Axis()
            axis = (
                (ax.Location().X(), ax.Location().Y(), ax.Location().Z()),
                (ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()),
            )
        elif surface == "plane":
            pln = adaptor.Plane()
            d = pln.Axis().Direction()
            normal = (d.X(), d.Y(), d.Z())
        elif surface == "sphere":
            radius = adaptor.Sphere().Radius()

        return {
            "surface": surface,
            "area": props.Mass(),
            "centroid": centroid,
            "radius": radius,
            "axis": axis,
            "normal": normal,
        }

    # -- queries --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._faces)

    def __contains__(self, fid: object) -> bool:
        return fid in self._faces

    def __iter__(self):
        return iter(self._info.values())

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._info)

    def info(self, fid: str) -> FaceInfo:
        return self._info[fid]

    def face(self, fid: str) -> TopoDS_Face:
        return self._faces[fid]

    def of_type(self, surface: str) -> tuple[FaceInfo, ...]:
        return tuple(f for f in self._info.values() if f.surface == surface)

    def cylinders_of_radius(
        self, radius: float, tol: float = 1e-4
    ) -> tuple[FaceInfo, ...]:
        """Every cylindrical face whose radius matches, within ``tol``.

        Returns all matches rather than one: a through hole is a single cylindrical face
        in some constructions and two coaxial halves in others, and a bolt pattern
        legitimately gives one match per hole. Callers that need exactly one say so.
        """
        return tuple(
            f
            for f in self._info.values()
            if f.is_cylindrical and f.radius is not None and abs(f.radius - radius) <= tol
        )

    def planes_with_normal(
        self, direction: tuple[float, float, float], tol: float = 1e-6
    ) -> tuple[FaceInfo, ...]:
        """Planar faces whose normal is parallel to ``direction``, either way along it."""
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        out = []
        for f in self._info.values():
            if not f.is_planar or f.normal is None:
                continue
            n = np.asarray(f.normal, dtype=float)
            if abs(abs(float(np.dot(n, d))) - 1.0) <= tol:
                out.append(f)
        return tuple(out)

    def nearest_cylinder_radii(self, radius: float, count: int = 3) -> tuple[float, ...]:
        """The distinct cylindrical radii closest to ``radius``.

        The verifier's ``size_exists`` check reports these when a dimension matches
        nothing, so a contradiction comes with the near misses that explain it.
        """
        radii = sorted({round(f.radius, 6) for f in self.of_type("cylinder") if f.radius})
        return tuple(sorted(radii, key=lambda r: abs(r - radius))[:count])


# --- semantic features ---------------------------------------------------------------


@dataclass(frozen=True)
class SemanticFeature:
    """What a region of the solid *is*, in the vocabulary a machinist would use.

    ``anchor`` and ``axis`` are the bridge to annotation. PLAN.md section 1.1 decided that
    annotation anchors come from here rather than from hidden-line-removal output: we know
    exactly where a bore's axis is because we placed it, so the drawing generator projects
    that 3D point through the view transform instead of trying to recover the
    correspondence from projected edges.
    """

    fid: str
    kind: str
    faces: tuple[str, ...]
    nominal: dict[str, float]
    anchor: tuple[float, float, float]
    axis: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    parent: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in FEATURES_OF_SIZE | SURFACE_FEATURES:
            raise ValueError(f"{self.fid}: unknown feature kind {self.kind!r}")
        if not self.faces:
            raise ValueError(f"{self.fid}: a feature must reference at least one face")

    @property
    def is_feature_of_size(self) -> bool:
        return self.kind in FEATURES_OF_SIZE

    def size(self) -> float | None:
        """The single dimension that makes this a feature of size, if it has one."""
        for key in ("diameter", "width", "thickness"):
            if key in self.nominal:
                return self.nominal[key]
        return None


@dataclass
class BuiltPart:
    """A solid, its exported STEP, its face index, and what it means."""

    family: str
    params: dict[str, Any]
    shape: TopoDS_Shape
    faces: FaceIndex
    features: tuple[SemanticFeature, ...]
    seed: int
    step_path: Path | None = None

    def feature(self, fid: str) -> SemanticFeature:
        for f in self.features:
            if f.fid == fid:
                return f
        raise KeyError(fid)

    def features_of_kind(self, kind: str) -> tuple[SemanticFeature, ...]:
        return tuple(f for f in self.features if f.kind == kind)

    def validate(self) -> None:
        """Every feature points at faces that exist, and no face id is invented.

        Cheap, and it fails loudly at build time rather than as a mysterious KeyError
        three modules downstream in the annotator.
        """
        known = set(self.faces.ids)
        seen_fids: set[str] = set()
        for feature in self.features:
            if feature.fid in seen_fids:
                raise ValueError(f"duplicate feature id {feature.fid!r}")
            seen_fids.add(feature.fid)
            missing = set(feature.faces) - known
            if missing:
                raise ValueError(
                    f"feature {feature.fid!r} references unknown faces {sorted(missing)}"
                )
            if feature.parent is not None and feature.parent not in seen_fids | {
                f.fid for f in self.features
            }:
                raise ValueError(
                    f"feature {feature.fid!r} names a parent {feature.parent!r} that "
                    f"does not exist"
                )
            for name, value in feature.nominal.items():
                if not math.isfinite(value):
                    raise ValueError(
                        f"feature {feature.fid!r} has a non-finite {name}: {value}"
                    )


@runtime_checkable
class PartFamily(Protocol):
    """The interface every part family implements (SPEC.md section 6.1)."""

    name: str

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw one manufacturable parameter set, resampling internally as needed."""
        ...

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        """Build the solid. Raises :class:`UnbuildableParams` if the geometry degenerates."""
        ...

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        """Name the features of the built solid.

        Deliberately separate from :meth:`build`, and deliberately given the
        :class:`FaceIndex` rather than the raw shape: a family must locate its faces by
        *geometry* -- "the cylinder of radius r whose axis is Z" -- and never by
        construction order. Construction order is not stable across a modelling change,
        and an annotation pointing at the wrong face is a ground-truth error that no
        downstream test would catch.
        """
        ...
