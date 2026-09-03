"""Annotation: what to say about each feature, and where on the sheet to put it.

This module owns ground truth. Every :class:`Annotation` it produces carries both the
primitives the renderer draws and the schema fields that describe them, built from the same
numbers in the same place, so the sheet and the JSON cannot disagree. The renderer is
deliberately dumb -- it draws primitives and never decides anything -- because any decision
made at render time would be a decision ground truth did not witness.

Three ideas carry the module.

**Anchors come from features, not from pixels.** A dimension knows the 3D point it refers
to because :class:`~balloonbench.partgen.types.SemanticFeature` recorded it at build time.
:meth:`~balloonbench.drawgen.views.ViewPlacement.point3d` turns that into sheet millimetres
through the same transform the linework went through. Nothing is measured off a render.

**Boxes are computed, never recovered.** The box a labeller would draw around a callout is
the box the text metrics say it occupies, in sheet millimetres, converted to pixels by
:meth:`~balloonbench.drawgen.views.SheetLayout.bbox_to_pixel`. CLAUDE.md requires this;
:mod:`balloonbench.drawgen.text` makes it accurate rather than merely present.

**Placement is scored, not solved.** SPEC.md section 7.3 asks for greedy placement by
decreasing importance with a few local-improvement passes, and explicitly not for an
optimiser -- real drawings are not optimally laid out either, and a drawing whose
annotations are packed perfectly is its own kind of tell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from balloonbench.drawgen.styles import HouseStyle
from balloonbench.drawgen.text import metrics_for
from balloonbench.drawgen.views import SheetLayout

__all__ = [
    "Annotation",
    "OccupancyGrid",
    "Primitive",
    "datum_symbol",
    "diameter_leader",
    "feature_frame",
    "linear_dimension",
    "note_leader",
    "MIN_BOX_EXTENT",
    "place_all",
]


# --- primitives -------------------------------------------------------------------------


@dataclass(frozen=True)
class Primitive:
    """One drawn thing, in sheet millimetres.

    The renderer switches on ``kind`` and draws exactly this; it never computes a position,
    a width or a text string of its own. Keeping the vocabulary this small is what makes
    the guarantee checkable: if a mark appears on a sheet, some annotation put a primitive
    there, and that annotation's bbox is in the ground truth.
    """

    kind: str  # line | arrow | text | box | triangle | circle | dot
    points: tuple[tuple[float, float], ...] = ()
    text: str = ""
    height: float = 0.0
    anchor: str = "left"
    rotation: float = 0.0
    width: float = 0.25
    filled: bool = False
    layer: str = "annotation"

    def translated(self, dx: float, dy: float) -> Primitive:
        return Primitive(
            kind=self.kind,
            points=tuple((x + dx, y + dy) for x, y in self.points),
            text=self.text,
            height=self.height,
            anchor=self.anchor,
            rotation=self.rotation,
            width=self.width,
            filled=self.filled,
            layer=self.layer,
        )


@dataclass
class Annotation:
    """A callout: what it says, what it draws, what it refers to, and where it ended up.

    ``payload`` holds the schema fields for this callout minus ``id`` and ``bbox``, which
    are assigned once placement is final. Splitting it that way means the semantic content
    is fixed before layout runs and cannot be perturbed by it -- a placement pass that
    could change a nominal value would be a ground-truth bug of the worst kind.
    """

    kind: str
    view: str
    text: str
    payload: dict[str, Any]
    #: Sheet-mm point the callout refers to; leaders and extension lines start here.
    anchor: tuple[float, float]
    #: Sheet-mm box of the geometry being referred to, for ``leader_target_bbox``.
    target_box: tuple[float, float, float, float] | None = None
    #: Higher is placed first. Placement is greedy, so this is the priority order in which
    #: callouts claim clear space: datums and critical GD&T before incidental dimensions.
    importance: float = 1.0
    #: Primitives whose position is fixed regardless of where the text lands.
    fixed: tuple[Primitive, ...] = ()
    #: Built once a text position is chosen.
    primitives: tuple[Primitive, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    #: Directions, in radians, the callout may be placed along, in preference order.
    directions: tuple[float, ...] = ()
    #: A builder called with a chosen text origin; returns (primitives, text bbox).
    builder: Any = None
    #: Sheet-mm outline of the view this callout belongs to. Placement measures its offset
    #: tiers from where a ray leaves this box, so callouts land outside the view rather
    #: than a fixed distance from a feature that may be larger than the offset.
    view_box: tuple[float, float, float, float] | None = None

    def realise(self, origin: tuple[float, float]) -> None:
        prims, box = self.builder(origin)
        self.primitives = tuple(self.fixed) + tuple(prims)
        self.bbox = _ensure_extent(box)


# --- occupancy --------------------------------------------------------------------------


class OccupancyGrid:
    """A coarse map of what part of the sheet is already covered in ink.

    Scoring a candidate position against every line segment on the sheet is quadratic and,
    for a few hundred drawings, slow enough to matter. A grid turns it into a constant-time
    sum over the candidate's own cells. One millimetre cells are finer than any spacing a
    drafting standard asks for, so the approximation never changes a decision that a reader
    would notice.

    Weights differ by what occupies a cell. Part linework is expensive to cross because a
    dimension over the part is genuinely wrong; an existing annotation is more expensive
    still, because two overlapping callouts are unreadable rather than merely untidy; the
    title block is effectively forbidden.
    """

    CELL = 1.0
    WEIGHT_GEOMETRY = 3.0
    WEIGHT_ANNOTATION = 12.0
    WEIGHT_RESERVED = 40.0

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.nx = int(math.ceil(width / self.CELL))
        self.ny = int(math.ceil(height / self.CELL))
        self._grid = np.zeros((self.ny, self.nx), dtype=np.float32)

    # -- filling ------------------------------------------------------------------------

    def _cells(self, box: tuple[float, float, float, float]) -> tuple[slice, slice]:
        x0, y0, x1, y1 = box
        cx0 = max(0, int(math.floor(x0 / self.CELL)))
        cy0 = max(0, int(math.floor(y0 / self.CELL)))
        cx1 = min(self.nx, int(math.ceil(x1 / self.CELL)))
        cy1 = min(self.ny, int(math.ceil(y1 / self.CELL)))
        return (slice(cy0, max(cy0 + 1, cy1)), slice(cx0, max(cx0 + 1, cx1)))

    def add_box(self, box: tuple[float, float, float, float], weight: float) -> None:
        rows, cols = self._cells(box)
        self._grid[rows, cols] += weight

    def add_polyline(
        self, points: tuple[tuple[float, float], ...], weight: float
    ) -> None:
        """Mark the cells a polyline passes through.

        Sampled along each segment at half a cell rather than rasterised exactly: a
        Bresenham walk would be more precise and no more useful, because the grid is
        already an approximation chosen to be finer than the decisions it informs.
        """
        for a, b in zip(points, points[1:], strict=False):
            length = math.dist(a, b)
            steps = max(1, int(length / (self.CELL / 2)))
            for i in range(steps + 1):
                t = i / steps
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                cx = int(x / self.CELL)
                cy = int(y / self.CELL)
                if 0 <= cx < self.nx and 0 <= cy < self.ny:
                    self._grid[cy, cx] += weight

    def add_primitives(self, primitives, weight: float) -> None:
        """Mark ink that is drawn but carries no ground-truth entry.

        Centre marks, bolt-circle phantoms and the cutting-plane line are all drawn on the
        sheet and must therefore be avoided, but none of them is a callout. Without this the
        placement solver treats the space they occupy as empty and lands a dimension on top
        of the cutting-plane line.
        """
        for prim in primitives:
            if prim.kind in ("line", "triangle") and len(prim.points) >= 2:
                self.add_polyline(tuple(prim.points), weight)
            elif prim.kind == "box" and len(prim.points) >= 2:
                (x0, y0), (x1, y1) = prim.points[0], prim.points[1]
                self.add_box((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), weight)
            elif prim.kind == "circle" and len(prim.points) >= 2:
                (cx, cy), (r, _u) = prim.points[0], prim.points[1]
                steps = max(12, int(2 * math.pi * r / (self.CELL / 2)))
                self.add_polyline(
                    tuple(
                        (cx + r * math.cos(2 * math.pi * i / steps),
                         cy + r * math.sin(2 * math.pi * i / steps))
                        for i in range(steps + 1)
                    ),
                    weight,
                )
            elif prim.kind == "text" and prim.points and prim.text:
                w = len(prim.text) * prim.height * 0.7
                x, y = prim.points[0]
                self.add_box((x - w / 2, y - prim.height * 0.4,
                              x + w / 2, y + prim.height), weight)

    def add_layout(self, layout: SheetLayout, text_height: float = 3.5) -> None:
        """Mark part linework, and reserve the title block and every view label."""
        for placement in layout.placements:
            if placement.spec.label:
                ox, oy = placement.label_origin(text_height)
                w = len(placement.spec.label) * text_height * 0.75
                self.add_box(
                    (ox - w / 2, oy - text_height * 0.4, ox + w / 2, oy + text_height * 1.2),
                    self.WEIGHT_ANNOTATION,
                )
            for line in placement.view.lines:
                self.add_polyline(
                    tuple(placement.to_sheet(p) for p in line.points),
                    self.WEIGHT_GEOMETRY,
                )
            for line in placement.hatch:
                self.add_polyline(
                    tuple(placement.to_sheet(p) for p in line.points),
                    self.WEIGHT_GEOMETRY * 0.4,
                )
        fx0, fy0, fx1, _fy1 = layout.frame
        from balloonbench.drawgen.views import TITLE_BLOCK_HEIGHT, TITLE_BLOCK_WIDTH

        self.add_box(
            (fx1 - TITLE_BLOCK_WIDTH, fy0, fx1, fy0 + TITLE_BLOCK_HEIGHT),
            self.WEIGHT_RESERVED,
        )
        # Outside the frame is not a placement option at all.
        self.add_box((0, 0, self.width, fy0), self.WEIGHT_RESERVED)
        self.add_box((0, layout.frame[3], self.width, self.height), self.WEIGHT_RESERVED)
        self.add_box((0, 0, fx0, self.height), self.WEIGHT_RESERVED)
        self.add_box((fx1, 0, self.width, self.height), self.WEIGHT_RESERVED)

    # -- scoring ------------------------------------------------------------------------

    def cost(self, box: tuple[float, float, float, float]) -> float:
        rows, cols = self._cells(box)
        return float(self._grid[rows, cols].sum())

    def occupied_fraction(self, box: tuple[float, float, float, float]) -> float:
        rows, cols = self._cells(box)
        region = self._grid[rows, cols]
        return float((region > 0).mean()) if region.size else 1.0


# --- geometry helpers -------------------------------------------------------------------


def _arrow(tip: tuple[float, float], along: float, style: HouseStyle) -> tuple[Primitive, ...]:
    """An arrowhead at ``tip`` pointing along ``along`` radians.

    ``tick`` is the oblique stroke used instead of an arrowhead in some offices; it points
    the opposite way from an arrow and is drawn across the dimension line rather than along
    it, which is why it is a separate branch rather than a parameter.
    """
    length = style.arrow_length
    if style.arrowhead == "tick":
        a = along + math.radians(45)
        half = length / 2
        return (
            Primitive(
                kind="line",
                points=(
                    (tip[0] - math.cos(a) * half, tip[1] - math.sin(a) * half),
                    (tip[0] + math.cos(a) * half, tip[1] + math.sin(a) * half),
                ),
                width=style.line_dimension,
            ),
        )

    spread = math.radians(9.0)
    back = along + math.pi
    p1 = (
        tip[0] + math.cos(back - spread) * length,
        tip[1] + math.sin(back - spread) * length,
    )
    p2 = (
        tip[0] + math.cos(back + spread) * length,
        tip[1] + math.sin(back + spread) * length,
    )
    return (
        Primitive(
            kind="triangle",
            points=(tip, p1, p2),
            filled=style.arrowhead == "filled",
            width=style.line_dimension,
        ),
    )


def _union(boxes) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = zip(*boxes, strict=True)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _pad(box: tuple[float, float, float, float], m: float) -> tuple[float, float, float, float]:
    return (box[0] - m, box[1] - m, box[2] + m, box[3] + m)


#: Minimum extent, in sheet millimetres, of a box recorded in ground truth.
MIN_BOX_EXTENT = 0.6


def _ensure_extent(
    box: tuple[float, float, float, float], minimum: float = MIN_BOX_EXTENT
) -> tuple[float, float, float, float]:
    """Widen a degenerate box about its own centre.

    What a linear dimension points at is a line segment between two measured points, and a
    segment has no area -- a horizontal dimension's target box is exactly zero millimetres
    tall. The schema rejects that, and rightly: a zero-area box has an IoU of zero against
    every prediction, so recording one would silently make that dimension unscoreable. The
    fix is to give it the extent a labeller would draw, centred on the segment rather than
    grown from one edge, so the box stays symmetric about what it refers to.
    """
    x0, y0, x1, y1 = box
    if x1 - x0 < minimum:
        cx = (x0 + x1) / 2
        x0, x1 = cx - minimum / 2, cx + minimum / 2
    if y1 - y0 < minimum:
        cy = (y0 + y1) / 2
        y0, y1 = cy - minimum / 2, cy + minimum / 2
    return (x0, y0, x1, y1)


# --- annotation builders ------------------------------------------------------------------


def linear_dimension(
    a: tuple[float, float],
    b: tuple[float, float],
    text: str,
    style: HouseStyle,
    *,
    view: str,
    kind: str = "dimension",
    payload: dict[str, Any] | None = None,
    horizontal: bool | None = None,
    importance: float = 1.0,
    target_box: tuple[float, float, float, float] | None = None,
) -> Annotation:
    """A dimension between two sheet-mm points, with extension lines and arrows.

    A dimension whose payload says ``is_basic`` is drawn enclosed in a rectangle, because
    that box *is* how a drawing says "basic". Omitting it would leave the sheet stating an
    ordinary toleranced dimension while ground truth recorded a theoretically exact one --
    a disagreement between image and label, which is the one class of bug this project
    cannot ship.

    ``horizontal`` forces the measured direction; by default the longer span wins, which is
    what a drafter does. The dimension line is offset perpendicular to that direction, and
    the offset is what placement chooses -- so a chain of dimensions in the same direction
    naturally lands on consistent tiers, which is the alignment reward SPEC.md section 7.3
    asks for without needing a separate mechanism.
    """
    if horizontal is None:
        horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])

    metrics = metrics_for(style.font)
    payload = dict(payload or {})

    def build(origin: tuple[float, float]):
        # ``origin`` carries the offset: for a horizontal dimension only its y matters.
        if horizontal:
            level = origin[1]
            p0, p1 = (a[0], level), (b[0], level)
            ext_a = ((a[0], a[1] + math.copysign(style.extension_gap, level - a[1])),
                     (a[0], level + math.copysign(style.extension_overshoot, level - a[1])))
            ext_b = ((b[0], b[1] + math.copysign(style.extension_gap, level - b[1])),
                     (b[0], level + math.copysign(style.extension_overshoot, level - b[1])))
            along_a, along_b = math.pi, 0.0
            text_origin = ((a[0] + b[0]) / 2, level + style.text_height * 0.4)
            rotation = 0.0
            text_anchor = "middle"
        else:
            level = origin[0]
            p0, p1 = (level, a[1]), (level, b[1])
            ext_a = ((a[0] + math.copysign(style.extension_gap, level - a[0]), a[1]),
                     (level + math.copysign(style.extension_overshoot, level - a[0]), a[1]))
            ext_b = ((b[0] + math.copysign(style.extension_gap, level - b[0]), b[1]),
                     (level + math.copysign(style.extension_overshoot, level - b[0]), b[1]))
            along_a, along_b = -math.pi / 2, math.pi / 2
            text_origin = (level + style.text_height * 0.4, (a[1] + b[1]) / 2)
            rotation = 90.0
            text_anchor = "middle"

        prims: list[Primitive] = [
            Primitive("line", ext_a, width=style.line_dimension),
            Primitive("line", ext_b, width=style.line_dimension),
        ]

        if style.text_position == "broken":
            # The text sits *in* the dimension line, so the line is drawn as two stubs
            # with a gap. The gap has to be measured from the same metrics the text is
            # drawn with, or the line runs under the characters.
            half = metrics.width(text, style.text_height) / 2 + style.text_height * 0.4
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            if horizontal:
                prims.append(Primitive("line", (p0, (mid[0] - half, mid[1])),
                                       width=style.line_dimension))
                prims.append(Primitive("line", ((mid[0] + half, mid[1]), p1),
                                       width=style.line_dimension))
                text_origin = (mid[0], mid[1] - style.text_height * 0.35)
            else:
                prims.append(Primitive("line", (p0, (mid[0], mid[1] - half)),
                                       width=style.line_dimension))
                prims.append(Primitive("line", ((mid[0], mid[1] + half), p1),
                                       width=style.line_dimension))
                text_origin = (mid[0] - style.text_height * 0.35, mid[1])
        else:
            prims.append(Primitive("line", (p0, p1), width=style.line_dimension))

        prims.extend(_arrow(p0, along_a, style))
        prims.extend(_arrow(p1, along_b, style))
        prims.append(
            Primitive(
                "text",
                (text_origin,),
                text=text,
                height=style.text_height,
                anchor=text_anchor,
                rotation=rotation,
            )
        )

        box = metrics.text_box(text, style.text_height, text_origin, anchor=text_anchor)
        if payload.get("is_basic"):
            m = style.text_height * 0.3
            frame = _pad(box, m)
            prims.append(
                Primitive("box", ((frame[0], frame[1]), (frame[2], frame[3])),
                          width=style.line_dimension)
            )
            box = frame
        if rotation:
            # A rotated string occupies the transpose of its box about its origin.
            w = box[2] - box[0]
            h = box[3] - box[1]
            box = (
                text_origin[0] - (text_origin[1] - box[1]),
                text_origin[1] - w / 2,
                text_origin[0] - (text_origin[1] - box[1]) + h,
                text_origin[1] + w / 2,
            )
            if payload.get("is_basic"):
                # The frame appended above was built in unrotated coordinates, so it boxes
                # the wrong region once the text turns. Replace it with the transposed box.
                prims = [q for q in prims if q.kind != "box"]
                prims.append(
                    Primitive("box", ((box[0], box[1]), (box[2], box[3])),
                              width=style.line_dimension)
                )
        return tuple(prims), box

    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return Annotation(
        kind=kind,
        view=view,
        text=text,
        payload=payload,
        anchor=mid,
        target_box=_ensure_extent(
            target_box
            or (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        ),
        importance=importance,
        directions=(math.pi / 2, -math.pi / 2) if horizontal else (0.0, math.pi),
        builder=build,
    )


def _leader(
    tip: tuple[float, float],
    landing: tuple[float, float],
    style: HouseStyle,
    *,
    shelf: float,
) -> tuple[tuple[Primitive, ...], tuple[float, float]]:
    """A leader from ``tip`` to ``landing`` with a horizontal shelf, and the text origin.

    The shelf runs away from the feature, so text always sits on the outside of the leader
    rather than back over the part. Returning the text origin rather than the text itself
    keeps this usable for a note, a diameter callout and a control frame alike.
    """
    direction = 1.0 if landing[0] >= tip[0] else -1.0
    shelf_end = (landing[0] + direction * shelf, landing[1])
    prims = (
        Primitive("line", (tip, landing), width=style.line_dimension),
        Primitive("line", (landing, shelf_end), width=style.line_dimension),
        *_arrow(tip, math.atan2(tip[1] - landing[1], tip[0] - landing[0]), style),
    )
    return prims, shelf_end


def diameter_leader(
    centre: tuple[float, float],
    radius: float,
    text: str,
    style: HouseStyle,
    *,
    view: str,
    payload: dict[str, Any] | None = None,
    importance: float = 1.0,
) -> Annotation:
    """A diameter callout on a circle shown as a circle: leader from the rim, text outside.

    The tip is placed on the rim in the direction of the chosen landing, so the leader
    points at the circle rather than at its centre. Pointing at the centre is a common
    generator mistake and reads immediately as wrong, because a diameter is a property of
    the surface.
    """
    metrics = metrics_for(style.font)
    payload = dict(payload or {})

    def build(origin: tuple[float, float]):
        angle = math.atan2(origin[1] - centre[1], origin[0] - centre[0])
        tip = (centre[0] + math.cos(angle) * radius, centre[1] + math.sin(angle) * radius)
        shelf = metrics.width(text, style.text_height)
        prims, shelf_end = _leader(tip, origin, style, shelf=shelf)
        text_origin = (
            shelf_end[0] if shelf_end[0] < origin[0] else origin[0],
            origin[1] + style.text_height * 0.3,
        )
        prims = prims + (
            Primitive("text", (text_origin,), text=text, height=style.text_height),
        )
        box = metrics.text_box(text, style.text_height, text_origin)
        if payload.get("is_basic"):
            box = _pad(box, style.text_height * 0.3)
            prims = prims + (
                Primitive("box", ((box[0], box[1]), (box[2], box[3])),
                          width=style.line_dimension),
            )
        return prims, box

    return Annotation(
        kind="dimension",
        view=view,
        text=text,
        payload=payload,
        anchor=centre,
        target_box=(
            centre[0] - radius, centre[1] - radius,
            centre[0] + radius, centre[1] + radius,
        ),
        importance=importance,
        directions=tuple(math.radians(a) for a in (30, 150, 210, 330, 60, 120, 240, 300)),
        builder=build,
    )


def note_leader(
    tip: tuple[float, float],
    text: str,
    style: HouseStyle,
    *,
    view: str,
    kind: str = "note",
    payload: dict[str, Any] | None = None,
    importance: float = 1.0,
    target_box: tuple[float, float, float, float] | None = None,
) -> Annotation:
    """A leader from a point on the part to a line of text."""
    metrics = metrics_for(style.font)
    payload = dict(payload or {})

    def build(origin: tuple[float, float]):
        prims, shelf_end = _leader(
            tip, origin, style, shelf=metrics.width(text, style.text_height)
        )
        text_origin = (
            min(shelf_end[0], origin[0]),
            origin[1] + style.text_height * 0.3,
        )
        prims = prims + (
            Primitive("text", (text_origin,), text=text, height=style.text_height),
        )
        return prims, metrics.text_box(text, style.text_height, text_origin)

    return Annotation(
        kind=kind,
        view=view,
        text=text,
        payload=payload,
        anchor=tip,
        target_box=target_box,
        importance=importance,
        directions=tuple(math.radians(a) for a in (30, 150, 210, 330, 60, 120, 240, 300)),
        builder=build,
    )


def feature_frame(
    tip: tuple[float, float],
    compartments: tuple[str, ...],
    text: str,
    style: HouseStyle,
    *,
    view: str,
    payload: dict[str, Any],
    importance: float = 2.0,
    target_box: tuple[float, float, float, float] | None = None,
) -> Annotation:
    """A feature control frame: a boxed row of compartments on a leader.

    The bbox is the whole frame, not the text inside it, because the frame is what a
    labeller boxes and what a detector is asked to find. That is also why the compartment
    widths are computed here from the same metrics the renderer uses -- an off-by-a-
    millimetre frame would put the box slightly off the drawn rectangle on every sheet.
    """
    metrics = metrics_for(style.font)
    pad = style.text_height * 0.45
    heights = style.text_height + 2 * pad
    widths = [metrics.width(c, style.text_height) + 2 * pad for c in compartments]

    def build(origin: tuple[float, float]):
        # ``origin`` is the frame's left edge at its vertical centre.
        x = origin[0]
        y0 = origin[1] - heights / 2
        y1 = origin[1] + heights / 2
        prims: list[Primitive] = []
        for compartment, w in zip(compartments, widths, strict=True):
            prims.append(
                Primitive(
                    "box",
                    ((x, y0), (x + w, y1)),
                    width=style.line_dimension,
                )
            )
            prims.append(
                Primitive(
                    "text",
                    ((x + w / 2, origin[1] - style.text_height * 0.38),),
                    text=compartment,
                    height=style.text_height,
                    anchor="middle",
                )
            )
            x += w
        box = (origin[0], y0, x, y1)

        # The leader attaches to whichever end of the frame faces the feature, so it never
        # crosses the frame it belongs to.
        attach = (origin[0], origin[1]) if tip[0] < origin[0] else (x, origin[1])
        prims.append(Primitive("line", (tip, attach), width=style.line_dimension))
        prims.extend(
            _arrow(tip, math.atan2(tip[1] - attach[1], tip[0] - attach[0]), style)
        )
        return tuple(prims), box

    return Annotation(
        kind="geometric_tolerance",
        view=view,
        text=text,
        payload=dict(payload),
        anchor=tip,
        target_box=target_box,
        importance=importance,
        directions=tuple(math.radians(a) for a in (0, 180, 45, 135, 225, 315, 90, 270)),
        builder=build,
    )


def datum_symbol(
    tip: tuple[float, float],
    label: str,
    style: HouseStyle,
    *,
    view: str,
    payload: dict[str, Any],
    importance: float = 3.0,
    target_box: tuple[float, float, float, float] | None = None,
) -> Annotation:
    """A datum feature symbol, in whichever of the two conventions the style uses.

    The filled triangle on a leader to a boxed letter is current practice. The ``-A-``
    form between dashes is the older one, and it is drawn without a triangle because that
    is what those drawings actually look like -- a triangle plus dashes would be a hybrid
    that never existed.
    """
    metrics = metrics_for(style.font)
    pad = style.text_height * 0.45

    def build(origin: tuple[float, float]):
        from balloonbench.drawgen.symbols import datum_label_text

        shown = datum_label_text(
            label, "dashed" if style.datum_style == "dashed" else "boxed"
        )
        w = metrics.width(shown, style.text_height) + 2 * pad
        h = style.text_height + 2 * pad
        x0, y0 = origin[0], origin[1] - h / 2
        box = (x0, y0, x0 + w, y0 + h)

        prims: list[Primitive] = []
        if style.datum_style == "filled_triangle":
            prims.append(Primitive("box", ((x0, y0), (x0 + w, y0 + h)),
                                   width=style.line_dimension))
            # Triangle sitting on the feature, leader up to the box.
            side = style.text_height
            base = (tip[0] - side / 2, tip[1]), (tip[0] + side / 2, tip[1])
            apex = (tip[0], tip[1] + math.copysign(side, origin[1] - tip[1]))
            prims.append(Primitive("triangle", (base[0], base[1], apex), filled=True,
                                   width=style.line_dimension))
            attach = (x0 + w / 2, y0 if origin[1] > tip[1] else y0 + h)
            prims.append(Primitive("line", (apex, attach), width=style.line_dimension))
        else:
            attach = (x0 + w / 2, y0 if origin[1] > tip[1] else y0 + h)
            prims.append(Primitive("line", (tip, attach), width=style.line_dimension))
            prims.extend(_arrow(tip, math.atan2(tip[1] - attach[1], tip[0] - attach[0]), style))

        prims.append(
            Primitive(
                "text",
                ((x0 + w / 2, origin[1] - style.text_height * 0.38),),
                text=shown,
                height=style.text_height,
                anchor="middle",
            )
        )
        return tuple(prims), box

    return Annotation(
        kind="datum",
        view=view,
        text=label,
        payload=dict(payload),
        anchor=tip,
        target_box=target_box,
        importance=importance,
        directions=tuple(math.radians(a) for a in (270, 90, 0, 180, 315, 225, 45, 135)),
        builder=build,
    )


# --- placement ----------------------------------------------------------------------------

#: Offset tiers tried for each direction, as multiples of the style's tier step added to its
#: first offset. Five tiers is enough to clear a crowded view without pushing a callout so
#: far from its feature that the leader stops being readable.
_TIERS = (0, 1, 2, 3, 4)


def place_all(
    annotations: list[Annotation],
    layout: SheetLayout,
    style: HouseStyle,
    *,
    passes: int = 2,
    extra_ink: tuple[Primitive, ...] = (),
) -> list[Annotation]:
    """Place every annotation, greedily by importance, then improve locally.

    Returns the annotations that were placed. One that cannot be placed anywhere inside the
    frame is **dropped rather than forced**: a drawing with nine readable callouts is a
    valid drawing, while one with ten where two overlap illegibly is a ground-truth entry
    that no labeller would agree with. The caller is told how many were dropped so a family
    that systematically overflows its sheet shows up as a number rather than as a mess.
    """
    grid = OccupancyGrid(layout.width, layout.height)
    grid.add_layout(layout, style.text_height)
    grid.add_primitives(extra_ink, OccupancyGrid.WEIGHT_GEOMETRY)

    for ann in annotations:
        if ann.view_box is None:
            try:
                ann.view_box = layout.placement(ann.view).sheet_bounds
            except KeyError:
                ann.view_box = None

    ordered = sorted(annotations, key=lambda a: -a.importance)
    placed: list[Annotation] = []

    for ann in ordered:
        best = _best_position(ann, grid, layout, style)
        if best is None:
            continue
        ann.realise(best)
        assert ann.bbox is not None
        grid.add_box(_pad(ann.bbox, 0.5), OccupancyGrid.WEIGHT_ANNOTATION)
        for prim in ann.primitives:
            if prim.kind in ("line", "triangle") and len(prim.points) >= 2:
                grid.add_polyline(prim.points, OccupancyGrid.WEIGHT_GEOMETRY)
        placed.append(ann)

    for _ in range(passes):
        _improve(placed, layout, style, extra_ink)
    return placed


def _exit_distance(
    origin: tuple[float, float], angle: float, box: tuple[float, float, float, float]
) -> float:
    """How far along ``angle`` from ``origin`` before leaving ``box``.

    The slab method, with the degenerate axis handled: a ray exactly parallel to one pair
    of sides never crosses them, so that axis imposes no bound. Returns zero when the
    origin is already outside, which is the right answer -- there is nothing to clear.
    """
    x0, y0, x1, y1 = box
    if not (x0 <= origin[0] <= x1 and y0 <= origin[1] <= y1):
        return 0.0
    dx, dy = math.cos(angle), math.sin(angle)
    best = math.inf
    for delta, lo, hi, here in (
        (dx, x0, x1, origin[0]),
        (dy, y0, y1, origin[1]),
    ):
        if abs(delta) < 1e-12:
            continue
        edge = hi if delta > 0 else lo
        best = min(best, (edge - here) / delta)
    return 0.0 if best is math.inf else max(0.0, best)


def _candidates(
    ann: Annotation, style: HouseStyle
) -> list[tuple[float, float]]:
    """Candidate text positions, measured from where the ray leaves the view outline.

    Offsetting a fixed distance from the *anchor* is the obvious implementation and is
    wrong: a shaft's diameter dimension is anchored on its axis, so a ten-millimetre offset
    puts the dimension line inside a shaft of any diameter over twenty. Every diameter on
    every turned part comes out drawn through the part. Measuring the offset from the point
    the ray leaves the view's outline instead reproduces what a drafter does -- dimension
    lines go outside the view, in tiers -- and makes the first tier mean the same thing on a
    10 mm shaft and a 315 mm flange.
    """
    out: list[tuple[float, float]] = []
    directions = ann.directions or (math.pi / 2,)
    for tier in _TIERS:
        for angle in directions:
            clear = _exit_distance(ann.anchor, angle, ann.view_box) if ann.view_box else 0.0
            r = clear + style.first_offset + tier * style.tier_step
            out.append((ann.anchor[0] + math.cos(angle) * r,
                        ann.anchor[1] + math.sin(angle) * r))
    return out


def _best_position(
    ann: Annotation,
    grid: OccupancyGrid,
    layout: SheetLayout,
    style: HouseStyle,
) -> tuple[float, float] | None:
    fx0, fy0, fx1, fy1 = layout.frame
    best: tuple[float, tuple[float, float]] | None = None

    for i, origin in enumerate(_candidates(ann, style)):
        prims, box = ann.builder(origin)
        if box[0] < fx0 or box[1] < fy0 or box[2] > fx1 or box[3] > fy1:
            continue
        cost = grid.cost(_pad(box, 0.5))
        # Cost also counts the leader's own path, so a candidate whose leader crosses the
        # part is penalised even when its text lands in clear space. A leader over
        # linework is exactly as unreadable as text over linework.
        for prim in prims:
            if prim.kind == "line" and len(prim.points) >= 2:
                cost += grid.cost(_pad(_union([_seg_box(prim.points)]), 0.2)) * 0.15
        # Earlier candidates are the preferred directions and nearer tiers, so a small
        # tie-break by index keeps placement stable and predictable rather than letting
        # floating-point noise choose between two equal positions.
        cost += i * 0.01
        if best is None or cost < best[0]:
            best = (cost, origin)
        if cost <= i * 0.01:
            break  # clear space in a preferred direction; no better option exists
    return None if best is None else best[1]


def _seg_box(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _improve(
    placed: list[Annotation],
    layout: SheetLayout,
    style: HouseStyle,
    extra_ink: tuple[Primitive, ...] = (),
) -> None:
    """One local-improvement pass: re-place each callout against everything else.

    Greedy placement is order-dependent -- an early low-importance callout can sit where a
    later important one needed to be. Re-placing each in turn against a grid that excludes
    only itself fixes most of that, and two passes is where the improvement stops being
    visible, which is the point SPEC.md section 7.3 makes about not needing an optimiser.
    """
    for i, ann in enumerate(placed):
        grid = OccupancyGrid(layout.width, layout.height)
        grid.add_layout(layout, style.text_height)
        grid.add_primitives(extra_ink, OccupancyGrid.WEIGHT_GEOMETRY)
        for j, other in enumerate(placed):
            if i == j or other.bbox is None:
                continue
            grid.add_box(_pad(other.bbox, 0.5), OccupancyGrid.WEIGHT_ANNOTATION)
        origin = _best_position(ann, grid, layout, style)
        if origin is not None:
            ann.realise(origin)
