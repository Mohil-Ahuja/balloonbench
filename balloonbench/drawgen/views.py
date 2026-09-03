"""Sheet geometry, view selection, section cutting, and the transform chain.

The whole of BalloonBench's ground-truth-by-construction claim reduces to one thing being
right: **a single chain of transforms from model space to pixels, used by everything.**

    model (mm, 3D)
      -> ViewTransform          orthographic basis, project.py
    view (mm, 2D)
      -> ViewPlacement.to_sheet scale and translate onto the sheet
    sheet (mm, origin bottom-left)
      -> SheetLayout.to_pixel   flip Y, apply DPI
    pixels

CLAUDE.md requires that bounding boxes are computed at annotation-placement time and
transformed by *the same* transform the rasteriser uses, never recovered from rendered
pixels. :meth:`ViewPlacement.point3d` and :meth:`SheetLayout.to_pixel` are that transform.
There is deliberately no second implementation of it anywhere in the package.

The Y flip in :meth:`SheetLayout.to_pixel` is the step worth stating out loud. Drawing
space, like every CAD system, has Y increasing upward from a bottom-left origin. Image
space has Y increasing downward from top-left. A bbox that skips the flip is still inside
the image and still looks plausible -- it is mirrored about the sheet's horizontal centre
line, which on a roughly symmetric drawing lands on ink often enough that a naive "does it
contain ink" test passes. That is why the acceptance test checks ink at the *annotation's
own* location rather than anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from balloonbench.drawgen.project import Polyline, ProjectedView, ViewTransform, project

__all__ = [
    "PROJECTION_ANGLES",
    "SHEET_SIZES",
    "STANDARD_SCALES",
    "SheetLayout",
    "ViewPlacement",
    "ViewSpec",
    "layout_for_part",
    "section_view",
    "view_plan",
]

#: ISO 216 sheet sizes in millimetres, landscape.
SHEET_SIZES: dict[str, tuple[float, float]] = {
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
    "A2": (594.0, 420.0),
    "A1": (841.0, 594.0),
    "A0": (1189.0, 841.0),
}

#: Preferred drawing scales, as ``(numerator, denominator)`` meaning drawing:real. ISO 5455
#: enumerates exactly these ratios; anything else on a sheet reads as a mistake even when
#: the geometry is right.
STANDARD_SCALES: tuple[tuple[int, int], ...] = (
    (10, 1), (5, 1), (2, 1), (1, 1), (1, 2), (1, 5), (1, 10), (1, 20), (1, 50), (1, 100),
)

PROJECTION_ANGLES: tuple[str, ...] = ("first_angle", "third_angle")

#: Margin from the sheet edge to the border, and the extra binding margin on the left.
SHEET_MARGIN = 10.0
BINDING_MARGIN = 20.0
#: Height of the title block strip, reserved out of the drawing area.
TITLE_BLOCK_HEIGHT = 40.0
TITLE_BLOCK_WIDTH = 180.0


@dataclass(frozen=True)
class ViewSpec:
    """A view to draw: which way to look, what to call it, and how to treat it."""

    name: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    #: Position relative to the primary view, in *third-angle* terms. First angle mirrors
    #: it. ``(0, 0)`` marks the primary view itself.
    relative: tuple[int, int] = (0, 0)
    hidden: bool = True
    #: When set, the view is a section cut by the plane through ``section_origin`` with
    #: this normal, and the retained half is the one the normal points away from.
    section_normal: tuple[float, float, float] | None = None
    section_letter: str | None = None
    label: str | None = None


#: Which views each part family is drawn in, and in what orientation.
#:
#: These are not the six standard directions applied blindly. A shaft's axis is Z in model
#: space, but a shaft is *always* drawn lying down -- so its elevation uses an up vector
#: that puts the axis horizontal on the sheet. Rotating the solid instead would have been
#: the obvious alternative and is the wrong one: every semantic feature's anchor is in model
#: coordinates, and rotating the model would leave anchors and linework in different frames,
#: which is precisely the failure PLAN.md section 1.1 was written to avoid.
_VIEW_PLANS: dict[str, tuple[ViewSpec, ...]] = {
    "shaft": (
        ViewSpec("elevation", (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0, 0)),
        ViewSpec("end", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1, 0), hidden=True),
    ),
    "flange": (
        ViewSpec("front", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 0), hidden=False),
        ViewSpec(
            "section",
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1, 0),
            hidden=False,
            section_normal=(0.0, 1.0, 0.0),
            section_letter="A",
            label="SECTION A-A",
        ),
    ),
    "plate_bracket": (
        ViewSpec("front", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 0), hidden=True),
        ViewSpec("side", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1, 0), hidden=True),
    ),
    "housing": (
        ViewSpec("front", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0, 0)),
        ViewSpec("top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 1)),
        ViewSpec("right", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1, 0)),
    ),
    "valve_body": (
        ViewSpec("front", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0, 0)),
        ViewSpec("top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 1)),
    ),
}

#: Used when a family has no explicit plan. Front plus top is the least a drawing can carry
#: and still define a solid.
_DEFAULT_PLAN: tuple[ViewSpec, ...] = (
    ViewSpec("front", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0, 0)),
    ViewSpec("top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0, 1)),
)


def view_plan(family: str) -> tuple[ViewSpec, ...]:
    return _VIEW_PLANS.get(family, _DEFAULT_PLAN)


# --- section cutting -------------------------------------------------------------------


def _bbox(shape) -> tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box.Get()


def section_view(
    shape,
    spec: ViewSpec,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    deflection: float | None = None,
) -> tuple[ProjectedView, tuple[Polyline, ...]]:
    """Cut ``shape`` with the spec's plane, project the remainder, and hatch the cut.

    Returns the projected view and the hatch lines, kept separate because they are drawn on
    different layers at different weights and because the hatch is generated in view
    coordinates rather than projected from 3D.

    The cut solid is the intersection of the part with a half-space box, so the retained
    half is genuinely solid and the faces created by the cut are real faces of it. Slicing
    with an infinite plane and taking the section curve would have been cheaper, but it
    yields curves rather than regions, and a region is what hatching needs.
    """
    if spec.section_normal is None:
        raise ValueError(f"view {spec.name!r} is not a section")

    normal = np.asarray(spec.section_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    axis = _axis_index(normal)
    lo = np.array(_bbox(shape)[:3], dtype=float)
    hi = np.array(_bbox(shape)[3:], dtype=float)
    pad = float(np.linalg.norm(hi - lo))

    centre = np.asarray(origin, dtype=float)
    # The retained half is everything on the negative-normal side of the plane, which is
    # the side the viewer is on for the section directions in the plans above. Building it
    # as an explicit axis-aligned min/max corner pair -- rather than positioning a cube by
    # arithmetic on the normal -- is what makes the cut land exactly on the plane; the
    # earlier version placed a box that missed the part entirely and produced a section
    # with nothing to hatch.
    box_lo = lo - pad
    box_hi = hi + pad
    if normal[axis] > 0:
        box_hi[axis] = centre[axis]
    else:
        box_lo[axis] = centre[axis]

    half = BRepPrimAPI_MakeBox(gp_Pnt(*box_lo), gp_Pnt(*box_hi)).Shape()

    common = BRepAlgoAPI_Common(shape, half)
    common.Build()
    if not common.IsDone():
        raise RuntimeError(f"section cut failed for view {spec.name!r}")
    cut = common.Shape()

    kwargs = {} if deflection is None else {"deflection": deflection}
    view = project(cut, spec.direction, spec.up, name=spec.name, hidden=spec.hidden, **kwargs)
    hatch = _hatch(cut, view.transform, normal, centre)
    return view, hatch


def _axis_index(normal: np.ndarray) -> int:
    """Which world axis a cutting-plane normal lies along.

    Section planes are restricted to the three principal planes. That is not a shortcut we
    are hiding: SPEC.md section 2.2 puts auxiliary and oblique views out of scope for v1,
    and every section a machinist expects on these five families is principal. An oblique
    normal is therefore a caller error, not a case to approximate.
    """
    for i in range(3):
        if abs(abs(float(normal[i])) - 1.0) < 1e-9:
            return i
    raise ValueError(
        f"section normal {tuple(normal)} is not axis-aligned; oblique sections are out "
        f"of scope for v1"
    )


def _hatch(
    cut_shape,
    transform: ViewTransform,
    normal: np.ndarray,
    plane_point: np.ndarray,
) -> tuple[Polyline, ...]:
    """Hatch lines over the faces the cut created, in view coordinates.

    Faces are selected by geometry -- planar, normal parallel to the cutting plane's, and
    lying in it -- rather than by asking the boolean which faces it made. The boolean's
    answer is an implementation detail; the geometric test is a definition.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    # shapely is a hard dependency, imported here rather than at module scope only to keep
    # the import graph shallow. It is deliberately *not* wrapped in a try/except: a section
    # view with no hatching is not a degraded drawing, it is a wrong one, and silently
    # returning an empty hatch is exactly how that ships unnoticed.
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    polygons = []
    explorer = TopExp_Explorer(cut_shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
            continue
        pln = adaptor.Plane()
        d = pln.Axis().Direction()
        n = np.array([d.X(), d.Y(), d.Z()])
        if abs(abs(float(np.dot(n, normal))) - 1.0) > 1e-6:
            continue
        loc = pln.Location()
        offset = float(np.dot(np.array([loc.X(), loc.Y(), loc.Z()]) - plane_point, normal))
        if abs(offset) > 1e-6:
            continue
        poly = _face_polygon(face, transform)
        if poly is not None and poly.is_valid and poly.area > 1e-6:
            polygons.append(poly)

    if not polygons:
        return ()

    region = unary_union(polygons)
    x0, y0, x1, y1 = region.bounds
    spacing = 3.0
    angle = math.radians(45.0)
    dx, dy = math.cos(angle), math.sin(angle)
    span = math.hypot(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    lines: list[Polyline] = []
    steps = int(span / spacing) + 2
    for i in range(-steps, steps + 1):
        # Offset perpendicular to the hatch direction, then extend far past the region and
        # let shapely trim. Clipping is what makes a hatch follow a pocket outline.
        ox, oy = -dy * i * spacing, dx * i * spacing
        probe = LineString(
            [
                (cx + ox - dx * span, cy + oy - dy * span),
                (cx + ox + dx * span, cy + oy + dy * span),
            ]
        )
        clipped = probe.intersection(region)
        if clipped.is_empty:
            continue
        for part in getattr(clipped, "geoms", [clipped]):
            if part.geom_type != "LineString" or part.length < 1e-6:
                continue
            lines.append(
                Polyline(points=tuple(part.coords), kind="line", layer="hatch")
            )
    return tuple(lines)


def _face_polygon(face, transform: ViewTransform):
    """A planar face as a shapely polygon in view coordinates, holes included."""
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from shapely.geometry import Polygon


    outer_wire = BRepTools.OuterWire_s(face)
    rings: list[list[tuple[float, float]]] = []
    outer: list[tuple[float, float]] | None = None

    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while explorer.More():
        wire = TopoDS.Wire_s(explorer.Current())
        explorer.Next()
        pts = _wire_points(wire, transform)
        if len(pts) < 3:
            continue
        if wire.IsSame(outer_wire):
            outer = pts
        else:
            rings.append(pts)

    if outer is None:
        return None
    try:
        return Polygon(outer, rings).buffer(0)
    except Exception:  # noqa: BLE001 - a self-intersecting projected wire is not fatal
        return None


def _wire_points(wire, transform: ViewTransform) -> list[tuple[float, float]]:
    """Ordered 2D points around a wire, in view coordinates.

    The orientation check is the whole substance of this function. ``BRepTools_WireExplorer``
    walks the edges in connection order, but each edge carries its own parameterisation, and
    an edge whose orientation in the wire is ``REVERSED`` is traversed from its last
    parameter to its first. Sampling every edge in its natural direction therefore emits a
    ring that doubles back on itself at every reversed edge. The resulting polygon is not
    empty and not obviously invalid -- it is a bow-tie -- so hatching it produces a neat
    series of triangular wedges instead of a filled region, which looks deliberate enough to
    survive a glance at a rendered sheet.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopAbs import TopAbs_REVERSED

    pts: list[tuple[float, float]] = []
    explorer = BRepTools_WireExplorer(wire)
    while explorer.More():
        edge = explorer.Current()
        explorer.Next()
        curve = BRepAdaptor_Curve(edge)
        sampler = GCPnts_QuasiUniformDeflection(curve, 0.05)
        if not sampler.IsDone() or sampler.NbPoints() < 2:
            continue
        indices = range(1, sampler.NbPoints() + 1)
        if edge.Orientation() == TopAbs_REVERSED:
            indices = reversed(indices)
        for i in indices:
            p = sampler.Value(i)
            xy = transform.point((p.X(), p.Y(), p.Z()))
            if not pts or math.dist(xy, pts[-1]) > 1e-9:
                pts.append(xy)
    return pts


# --- placement and the transform chain -------------------------------------------------


@dataclass(frozen=True)
class ViewPlacement:
    """One projected view, scaled and positioned on the sheet."""

    spec: ViewSpec
    view: ProjectedView
    #: drawing:real, e.g. 0.5 for a 1:2 sheet.
    scale: float
    #: Sheet-millimetre position of the view's local origin (view coordinate ``(0, 0)``).
    origin: tuple[float, float]
    hatch: tuple[Polyline, ...] = ()

    @property
    def name(self) -> str:
        return self.spec.name

    def to_sheet(self, xy: tuple[float, float]) -> tuple[float, float]:
        """View millimetres to sheet millimetres."""
        return (self.origin[0] + xy[0] * self.scale, self.origin[1] + xy[1] * self.scale)

    def point3d(self, p: tuple[float, float, float]) -> tuple[float, float]:
        """Model-space 3D point straight to sheet millimetres.

        This is the method annotation placement uses to turn a
        :class:`~balloonbench.partgen.types.SemanticFeature` anchor into a position on the
        sheet, and it composes exactly the same two steps the linework went through.
        """
        return self.to_sheet(self.view.transform.point(p))

    @property
    def sheet_bounds(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.view.bounds
        a = self.to_sheet((x0, y0))
        b = self.to_sheet((x1, y1))
        return (a[0], a[1], b[0], b[1])

    def label_origin(self, text_height: float) -> tuple[float, float]:
        """Baseline origin of the view label, centred beneath the view.

        Defined here rather than in the renderer so that the placement solver can reserve
        the space the label will occupy. Computing it in two places is how a caption ends
        up printed through a datum symbol: the annotator sees clear paper where the
        renderer will later put text.
        """
        x0, y0, x1, _y1 = self.sheet_bounds
        return ((x0 + x1) / 2, y0 - text_height * 2.5)


@dataclass(frozen=True)
class SheetLayout:
    """A sheet, its drawing frame, and the views placed on it."""

    size: str
    width: float
    height: float
    scale: tuple[int, int]
    projection: str
    placements: tuple[ViewPlacement, ...]

    @property
    def scale_text(self) -> str:
        return f"{self.scale[0]}:{self.scale[1]}"

    @property
    def frame(self) -> tuple[float, float, float, float]:
        """The border rectangle, ``(x0, y0, x1, y1)`` in sheet millimetres."""
        return (
            BINDING_MARGIN,
            SHEET_MARGIN,
            self.width - SHEET_MARGIN,
            self.height - SHEET_MARGIN,
        )

    @property
    def drawing_area(self) -> tuple[float, float, float, float]:
        """The frame minus the title block, which views must not overlap."""
        x0, y0, x1, y1 = self.frame
        return (x0, y0 + TITLE_BLOCK_HEIGHT, x1, y1)

    def placement(self, name: str) -> ViewPlacement:
        for p in self.placements:
            if p.name == name:
                return p
        raise KeyError(f"no view {name!r} on this sheet")

    def pixel_size(self, dpi: float) -> tuple[int, int]:
        px = dpi / 25.4
        return (int(round(self.width * px)), int(round(self.height * px)))

    def to_pixel(self, xy: tuple[float, float], dpi: float) -> tuple[float, float]:
        """Sheet millimetres to pixels, with the Y flip into image space.

        >>> layout = SheetLayout('A4', 297.0, 210.0, (1, 1), 'first_angle', ())
        >>> [round(v, 3) for v in layout.to_pixel((0.0, 0.0), 25.4)]
        [0.0, 210.0]
        >>> [round(v, 3) for v in layout.to_pixel((297.0, 210.0), 25.4)]
        [297.0, 0.0]
        """
        px = dpi / 25.4
        return (xy[0] * px, (self.height - xy[1]) * px)

    def bbox_to_pixel(
        self, box: tuple[float, float, float, float], dpi: float
    ) -> tuple[float, float, float, float]:
        """A sheet-space ``(x0, y0, x1, y1)`` box to a pixel-space one.

        The Y flip swaps which corner is the minimum, so the components are re-sorted
        rather than transformed in place. Skipping that re-sort produces a box with
        ``y0 > y1``, which every consumer either rejects or, worse, silently normalises.
        """
        a = self.to_pixel((box[0], box[1]), dpi)
        b = self.to_pixel((box[2], box[3]), dpi)
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def _choose_scale(
    extent: tuple[float, float], available: tuple[float, float]
) -> tuple[int, int]:
    """Largest standard scale at which the views fit the drawing area.

    Largest rather than best-fitting: a drawing is meant to be read, and the convention is
    to fill the sheet at a preferred ratio rather than to pick an arbitrary one that fits
    exactly.
    """
    for num, den in STANDARD_SCALES:
        s = num / den
        if extent[0] * s <= available[0] and extent[1] * s <= available[1]:
            return (num, den)
    return STANDARD_SCALES[-1]


def _grid_offsets(
    relative: tuple[int, int], projection: str
) -> tuple[int, int]:
    """Where a view sits relative to the primary, for the sheet's projection convention.

    Third angle places each view on the side of the primary it is seen from: the top view
    goes above, the right view to the right. First angle places it on the opposite side --
    the top view below, the right view to the left. This single sign flip is the whole
    difference between the two conventions, and getting it backwards produces a drawing
    that is internally consistent and describes a mirrored part. Indian and European
    practice is first angle; US practice is third.
    """
    col, row = relative
    if projection == "first_angle":
        return (-col, -row)
    return (col, row)


def layout_for_part(
    shape,
    family: str,
    *,
    sheet: str = "A3",
    projection: str = "first_angle",
    gap: float = 30.0,
    deflection: float | None = None,
) -> SheetLayout:
    """Project a part into its family's views and place them on a sheet.

    Views are laid out on a grid whose cells are sized by the largest view, which is what
    keeps a multi-view drawing's views aligned -- a top view whose centre does not sit
    under the front view's centre is the single most obvious sign of a drawing that was
    generated rather than drafted.
    """
    if sheet not in SHEET_SIZES:
        raise KeyError(f"unknown sheet size {sheet!r}")
    if projection not in PROJECTION_ANGLES:
        raise KeyError(f"unknown projection {projection!r}")

    width, height = SHEET_SIZES[sheet]
    specs = view_plan(family)

    kwargs = {} if deflection is None else {"deflection": deflection}
    projected: list[tuple[ViewSpec, ProjectedView, tuple[Polyline, ...]]] = []
    for spec in specs:
        if spec.section_normal is not None:
            view, hatch = section_view(shape, spec, deflection=deflection)
        else:
            view = project(shape, spec.direction, spec.up, name=spec.name,
                           hidden=spec.hidden, **kwargs)
            hatch = ()
        projected.append((spec, view, hatch))

    # Column widths and row heights are sized to the views that actually occupy them, not
    # to the largest view on the sheet. Sizing every cell to the largest is simpler and
    # produces the characteristic fault of a generated drawing: a shaft's 25 mm end view
    # marooned a hundred millimetres from its 112 mm elevation, with the alignment still
    # technically correct and the sheet still unmistakably wrong.
    cells: dict[tuple[int, int], tuple[float, float]] = {}
    for spec, view, _h in projected:
        key = _grid_offsets(spec.relative, projection)
        w, h = view.size
        prev = cells.get(key, (0.0, 0.0))
        cells[key] = (max(prev[0], w), max(prev[1], h))

    cols = sorted({c for c, _r in cells})
    rows = sorted({r for _c, r in cells})
    col_w = {c: max((cells[k][0] for k in cells if k[0] == c), default=0.0) for c in cols}
    row_h = {r: max((cells[k][1] for k in cells if k[1] == r), default=0.0) for r in rows}
    span_x = sum(col_w.values()) + (len(cols) - 1) * gap
    span_y = sum(row_h.values()) + (len(rows) - 1) * gap

    ax0, ay0, ax1, ay1 = SheetLayout(
        sheet, width, height, (1, 1), projection, ()
    ).drawing_area
    available = (ax1 - ax0, ay1 - ay0)
    # The gaps between views are sheet millimetres and do not scale with the part, so they
    # are subtracted from the space available rather than scaled along with the geometry.
    gaps = ((len(cols) - 1) * gap, (len(rows) - 1) * gap)
    num, den = _choose_scale(
        (span_x - gaps[0], span_y - gaps[1]),
        (max(1.0, available[0] - gaps[0]), max(1.0, available[1] - gaps[1])),
    )
    scale = num / den

    # Centre the whole arrangement in the drawing area, then place each view at its grid
    # cell's centre. Centring the arrangement rather than each view keeps the alignment
    # between views exact regardless of how much of its cell each one fills.
    total_w = (sum(col_w.values()) * scale) + (len(cols) - 1) * gap
    total_h = (sum(row_h.values()) * scale) + (len(rows) - 1) * gap
    base_x = ax0 + (available[0] - total_w) / 2
    base_y = ay0 + (available[1] - total_h) / 2

    # Running offsets to the centre of each column and row.
    col_centre: dict[int, float] = {}
    acc = base_x
    for c in cols:
        col_centre[c] = acc + col_w[c] * scale / 2
        acc += col_w[c] * scale + gap
    row_centre: dict[int, float] = {}
    acc = base_y
    for r in rows:
        row_centre[r] = acc + row_h[r] * scale / 2
        acc += row_h[r] * scale + gap

    placements: list[ViewPlacement] = []
    for spec, view, hatch in projected:
        col, row = _grid_offsets(spec.relative, projection)
        vx0, vy0, vx1, vy1 = view.bounds
        cx, cy = col_centre[col], row_centre[row]
        origin = (
            cx - (vx0 + vx1) / 2 * scale,
            cy - (vy0 + vy1) / 2 * scale,
        )
        placements.append(
            ViewPlacement(spec=spec, view=view, scale=scale, origin=origin, hatch=hatch)
        )

    return SheetLayout(
        size=sheet,
        width=width,
        height=height,
        scale=(num, den),
        projection=projection,
        placements=tuple(placements),
    )
