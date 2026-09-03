"""Orthographic projection of a solid into 2D view coordinates.

SPEC.md section 7.1 asks for hidden-line removal *and* for a map from each projected edge
back to the B-rep face it came from, calling that map the highest-risk piece of the project.
PLAN.md section 1.1 took the spec's own fallback instead, and took it from day one rather
than after a slip: HLR draws the linework, and annotation anchors are projected directly
from :class:`~balloonbench.partgen.types.SemanticFeature` through the same transform.

The reason that substitution is sound is visible in :class:`ViewTransform`. OCCT's
``HLRBRep_HLRToShape`` returns edges already flattened into the ``z = 0`` plane of the
projection coordinate system, so a projected edge's view coordinates are literally the
``(X, Y)`` of its points. :meth:`ViewTransform.point` computes the same two numbers for an
arbitrary 3D point by taking dot products against the same basis. Linework and anchors are
therefore in one frame by construction, not by a correspondence we had to recover -- and a
bore's centre mark lands on the bore because we placed the bore, which is a far stronger
guarantee than matching a projected edge to a face and hoping.

The cost is that we cannot say "this specific projected line is the top of the flange". We
never need to: every annotation is anchored to a feature, and features know their own
geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

__all__ = [
    "STANDARD_DIRECTIONS",
    "Polyline",
    "ProjectedView",
    "ViewTransform",
    "project",
    "project_standard",
]

#: Chord deflection for discretising a projected curve, in millimetres of model space. A
#: circle discretised at deflection d has a chord error of d, so 0.02 mm keeps a bore
#: visually round at 600 DPI, where one pixel is about 0.04 mm on the sheet.
DEFLECTION = 0.02

#: The six standard orthographic directions, as ``(view direction, up)``. The view direction
#: points *from the object toward the viewer*, which is the convention OCCT's projector uses
#: and the opposite of a camera "look at" vector -- getting this backwards mirrors every
#: view, which is an error that survives a long time because a symmetric part looks fine
#: either way.
STANDARD_DIRECTIONS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError(f"zero-length direction {v}")
    return a / n


@dataclass(frozen=True)
class ViewTransform:
    """An orthonormal basis mapping model space to a view's 2D plane.

    >>> t = ViewTransform.from_direction((0, 0, 1), (0, 1, 0))
    >>> [round(v, 6) for v in t.point((3.0, 4.0, 99.0))]
    [3.0, 4.0]

    The third component is dropped, not stored: depth plays no part in an orthographic
    drawing, and keeping it would invite someone to sort by it and reintroduce a
    view-dependent ordering into ground truth.
    """

    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    xdir: tuple[float, float, float]
    ydir: tuple[float, float, float]

    @classmethod
    def from_direction(
        cls, direction: tuple[float, float, float], up: tuple[float, float, float]
    ) -> ViewTransform:
        d = _unit(direction)
        u = _unit(up)
        # Gram-Schmidt: an ``up`` that is not perpendicular to the view direction is a
        # reasonable thing for a caller to pass, and silently accepting it would give a
        # non-orthonormal basis in which distances no longer survive projection.
        u = u - np.dot(u, d) * d
        norm = np.linalg.norm(u)
        if norm < 1e-9:
            raise ValueError(f"up {up} is parallel to view direction {direction}")
        u = u / norm
        x = np.cross(u, d)
        return cls(
            direction=tuple(d),
            up=tuple(u),
            xdir=tuple(x),
            ydir=tuple(u),
        )  # type: ignore[arg-type]

    def point(self, p: tuple[float, float, float]) -> tuple[float, float]:
        """Project one model-space point into view coordinates."""
        v = np.asarray(p, dtype=float)
        return (float(np.dot(v, self.xdir)), float(np.dot(v, self.ydir)))

    def points(self, ps) -> np.ndarray:
        """Project many points at once. Returns an ``(n, 2)`` array."""
        m = np.asarray(ps, dtype=float).reshape(-1, 3)
        return m @ np.column_stack([self.xdir, self.ydir])

    def direction_2d(self, v: tuple[float, float, float]) -> tuple[float, float]:
        """Project a *direction* (an axis, a normal) rather than a position.

        Identical arithmetic to :meth:`point`, but named separately because the result is
        not a location and must never be used as one -- an axis direction projected as if
        it were a point is a bug that produces plausible-looking leader lines.
        """
        return self.point(v)

    def ax2(self) -> gp_Ax2:
        """The OCCT projection frame this transform corresponds to."""
        return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*self.direction), gp_Dir(*self.xdir))

    def depth(self, p: tuple[float, float, float]) -> float:
        """Distance along the view direction. Used to decide near/far, never for layout."""
        return float(np.dot(np.asarray(p, dtype=float), self.direction))


@dataclass(frozen=True)
class Polyline:
    """One projected edge, discretised, in view coordinates.

    ``kind`` records what the curve was in 3D. The renderer uses it to decide line weight
    and, for a circle, to emit a true arc rather than a chain of chords -- a visibly
    faceted bolt hole is one of the tells that a drawing was synthesised.
    """

    points: tuple[tuple[float, float], ...]
    kind: str = "other"
    layer: str = "visible"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def length(self) -> float:
        return sum(
            math.dist(a, b) for a, b in zip(self.points, self.points[1:], strict=False)
        )


#: The HLR result compounds we consume, and the layer each becomes. Smooth edges
#: (``Rg1LineVCompound``, the tangent seam where a fillet meets a face) are deliberately
#: dropped: a real drawing does not show them, and including them is an instant tell.
_HLR_LAYERS: tuple[tuple[str, str], ...] = (
    ("VCompound", "visible"),
    ("OutLineVCompound", "visible"),
    ("HCompound", "hidden"),
    ("OutLineHCompound", "hidden"),
)


@dataclass
class ProjectedView:
    """The linework of one orthographic view, in view coordinates (millimetres)."""

    name: str
    transform: ViewTransform
    lines: tuple[Polyline, ...] = field(default_factory=tuple)

    def of_layer(self, layer: str) -> tuple[Polyline, ...]:
        return tuple(p for p in self.lines if p.layer == layer)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Extent of the visible linework, ``(xmin, ymin, xmax, ymax)``.

        Hidden lines are excluded on purpose. They can only ever fall inside the visible
        silhouette of a solid, so including them cannot widen the box -- but a stray
        hidden-line artefact could, and a view scaled to fit an artefact is a view that
        does not fit the part.
        """
        visible = self.of_layer("visible") or self.lines
        if not visible:
            raise ValueError(f"view {self.name!r} projected to no linework at all")
        boxes = np.array([p.bounds for p in visible])
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].min()),
            float(boxes[:, 2].max()),
            float(boxes[:, 3].max()),
        )

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x1 - x0, y1 - y0)


_CURVE_NAMES: dict[object, str] = {
    GeomAbs_CurveType.GeomAbs_Line: "line",
    GeomAbs_CurveType.GeomAbs_Circle: "circle",
    GeomAbs_CurveType.GeomAbs_Ellipse: "ellipse",
    GeomAbs_CurveType.GeomAbs_BSplineCurve: "bspline",
    GeomAbs_CurveType.GeomAbs_BezierCurve: "bezier",
}


def _discretise(edge, deflection: float) -> tuple[tuple[float, float], ...]:
    """Sample a projected edge into a polyline in the ``z = 0`` projection plane."""
    curve = BRepAdaptor_Curve(edge)
    kind = _CURVE_NAMES.get(curve.GetType(), "other")
    if kind == "line":
        # A straight edge needs exactly two points. Letting the deflection algorithm
        # sample it would put redundant collinear vertices into every DXF we emit.
        p0 = curve.Value(curve.FirstParameter())
        p1 = curve.Value(curve.LastParameter())
        return ((p0.X(), p0.Y()), (p1.X(), p1.Y()))

    sampler = GCPnts_QuasiUniformDeflection(curve, deflection)
    if not sampler.IsDone() or sampler.NbPoints() < 2:
        p0 = curve.Value(curve.FirstParameter())
        p1 = curve.Value(curve.LastParameter())
        return ((p0.X(), p0.Y()), (p1.X(), p1.Y()))
    return tuple(
        (p.X(), p.Y())
        for p in (sampler.Value(i) for i in range(1, sampler.NbPoints() + 1))
    )


def _edge_kind(edge) -> str:
    return _CURVE_NAMES.get(BRepAdaptor_Curve(edge).GetType(), "other")


def project(
    shape: TopoDS_Shape,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    *,
    name: str = "view",
    deflection: float = DEFLECTION,
    hidden: bool = True,
) -> ProjectedView:
    """Project ``shape`` orthographically and remove hidden lines.

    ``hidden`` controls whether the hidden edges are *drawn*, not whether hidden-line
    removal runs. ``Hide()`` always runs, and it is the expensive half of the algorithm --
    two to three seconds on a housing against ten milliseconds for the rest. Skipping it
    for views that do not show hidden detail is therefore tempting and wrong: without it,
    ``VCompound`` returns every edge in the model, so the far wall of a bore is drawn as a
    solid line indistinguishable from the near one. The result still looks like linework,
    which is why the shortcut survives review; it is a wireframe, not a drawing.
    """
    transform = ViewTransform.from_direction(direction, up)

    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(transform.ax2()))
    algo.Update()
    algo.Hide()

    to_shape = HLRBRep_HLRToShape(algo)
    lines: list[Polyline] = []
    for accessor, layer in _HLR_LAYERS:
        if layer == "hidden" and not hidden:
            continue
        try:
            compound = getattr(to_shape, accessor)()
        except Exception:  # noqa: BLE001 - an empty result raises rather than returning None
            continue
        if compound is None or compound.IsNull():
            continue
        explorer = TopExp_Explorer(compound, TopAbs_EDGE)
        while explorer.More():
            edge = TopoDS.Edge_s(explorer.Current())
            explorer.Next()
            if BRep_Tool.Degenerated_s(edge):
                continue
            pts = _discretise(edge, deflection)
            if len(pts) < 2:
                continue
            lines.append(Polyline(points=pts, kind=_edge_kind(edge), layer=layer))

    return ProjectedView(name=name, transform=transform, lines=tuple(lines))


def project_standard(
    shape: TopoDS_Shape,
    view: str,
    *,
    hidden: bool = True,
    deflection: float = DEFLECTION,
) -> ProjectedView:
    """Project one of the six named standard views."""
    if view not in STANDARD_DIRECTIONS:
        raise KeyError(
            f"unknown standard view {view!r}; known: {sorted(STANDARD_DIRECTIONS)}"
        )
    direction, up = STANDARD_DIRECTIONS[view]
    return project(shape, direction, up, name=view, hidden=hidden, deflection=deflection)
