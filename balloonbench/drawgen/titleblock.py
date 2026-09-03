"""Sheet border, zone markings, and the title block.

The title block is not decoration. It carries the general tolerance note that governs every
dimension the sheet leaves untoleranced, the units, the scale, and the projection
convention -- and a reader who ignores it will misread the drawing, which is precisely why
a benchmark should include it and why ``TitleBlock`` is a required part of the schema
rather than an optional extra.

The projection symbol deserves its own note. It is the frustum-and-circles glyph whose two
halves swap between first and third angle, and it is the only thing on many real sheets
that says which convention is in force. A model that never learns to read it will mirror
every part it sees from an Indian or European supplier. Drawing it correctly -- and
differently for the two conventions -- is therefore load-bearing, not ornamental.

Everything here is emitted as :class:`~balloonbench.drawgen.annotate.Primitive` objects, the
same vocabulary annotations use, so the renderer has exactly one drawing path.
"""

from __future__ import annotations

from dataclasses import dataclass

from balloonbench.drawgen.annotate import Primitive
from balloonbench.drawgen.styles import HouseStyle
from balloonbench.drawgen.text import metrics_for
from balloonbench.drawgen.views import (
    TITLE_BLOCK_HEIGHT,
    TITLE_BLOCK_WIDTH,
    SheetLayout,
)

__all__ = ["TitleBlockData", "border_primitives", "title_block_primitives"]


@dataclass(frozen=True)
class TitleBlockData:
    """The fields the schema's ``title_block`` records, plus what is only drawn."""

    part_number: str
    revision: str
    material: str
    general_tolerance: str
    surface_finish_default: str | None
    title: str
    scale_text: str
    projection: str
    sheet_size: str
    units: str


def border_primitives(layout: SheetLayout, style: HouseStyle) -> tuple[Primitive, ...]:
    """The drawing frame and the zone letters and numbers along it.

    Zones exist so a revision note can say "see zone C3". They are drawn in the margin
    *outside* the frame, which is where a real sheet puts them and which is not merely a
    convention: the title block occupies the bottom-right of the frame's interior, so zone
    labels drawn inside would print through it. The regular repetition of these marks along
    every edge is also a strong visual cue a detector must learn to ignore, and an easy
    source of false positives for anything trained only on clean synthetic sheets.
    """
    x0, y0, x1, y1 = layout.frame
    prims: list[Primitive] = [
        Primitive("box", ((x0, y0), (x1, y1)), width=style.line_border, layer="border"),
    ]

    # Zone cells divide the frame into a whole number of roughly 50 mm cells.
    cols = max(2, round((x1 - x0) / 50.0))
    rows = max(2, round((y1 - y0) / 50.0))
    cw = (x1 - x0) / cols
    ch = (y1 - y0) / rows
    height = style.text_height * 0.9

    below = y0 / 2
    above = (y1 + layout.height) / 2
    left = x0 / 2
    right = (x1 + layout.width) / 2

    for i in range(cols):
        cx = x0 + (i + 0.5) * cw
        label = str(i + 1)
        for cy in (below, above):
            prims.append(
                Primitive("text", ((cx, cy - height * 0.5),), text=label, height=height,
                          anchor="middle", layer="border")
            )
        if i:
            ex = x0 + i * cw
            prims.append(Primitive("line", ((ex, 0.0), (ex, y0)),
                                   width=style.line_dimension, layer="border"))
            prims.append(Primitive("line", ((ex, y1), (ex, layout.height)),
                                   width=style.line_dimension, layer="border"))

    for j in range(rows):
        cy = y0 + (j + 0.5) * ch
        label = chr(ord("A") + j)
        for cx in (left, right):
            prims.append(
                Primitive("text", ((cx, cy - height * 0.5),), text=label, height=height,
                          anchor="middle", layer="border")
            )
        if j:
            ey = y0 + j * ch
            prims.append(Primitive("line", ((0.0, ey), (x0, ey)),
                                   width=style.line_dimension, layer="border"))
            prims.append(Primitive("line", ((x1, ey), (layout.width, ey)),
                                   width=style.line_dimension, layer="border"))
    return tuple(prims)


def _projection_symbol(
    cx: float, cy: float, size: float, projection: str, style: HouseStyle
) -> tuple[Primitive, ...]:
    """The truncated-cone symbol, oriented for the sheet's projection convention.

    The symbol is a cone seen from its side and from its end. In third angle the end view
    sits on the side of the side view that you would see it from; in first angle it sits on
    the opposite side. Drawing the two circles on the correct side is the entire content of
    the symbol, so the mirror is applied to the layout of the two halves rather than to the
    shapes themselves.
    """
    h = size
    r_big = h * 0.5
    r_small = h * 0.3
    gap = h * 0.35
    prims: list[Primitive] = []

    # Side view: a trapezium, big end toward the circles in third angle.
    flip = -1.0 if projection == "first_angle" else 1.0
    sx = cx - flip * (gap + h * 0.6)
    prims.append(
        Primitive(
            "line",
            (
                (sx - flip * h * 0.5, cy - r_small),
                (sx + flip * h * 0.5, cy - r_big),
                (sx + flip * h * 0.5, cy + r_big),
                (sx - flip * h * 0.5, cy + r_small),
                (sx - flip * h * 0.5, cy - r_small),
            ),
            width=style.line_dimension,
            layer="border",
        )
    )
    ex = cx + flip * (gap + h * 0.6)
    prims.append(Primitive("circle", ((ex, cy), (r_big, 0.0)),
                           width=style.line_dimension, layer="border"))
    prims.append(Primitive("circle", ((ex, cy), (r_small, 0.0)),
                           width=style.line_dimension, layer="border"))
    return tuple(prims)


def title_block_primitives(
    layout: SheetLayout, style: HouseStyle, data: TitleBlockData
) -> tuple[Primitive, ...]:
    """The title block: its ruling, its labelled fields, and the projection symbol."""
    fx0, fy0, fx1, _fy1 = layout.frame
    x0 = fx1 - TITLE_BLOCK_WIDTH
    y0 = fy0
    x1, y1 = fx1, fy0 + TITLE_BLOCK_HEIGHT
    metrics = metrics_for(style.font)

    label_h = style.text_height * 0.62
    value_h = style.text_height

    prims: list[Primitive] = [
        Primitive("box", ((x0, y0), (x1, y1)), width=style.line_border, layer="border")
    ]

    # Three rows; the top one is the title, which is why it is taller than the other two.
    row = TITLE_BLOCK_HEIGHT / 3
    for i in (1, 2):
        prims.append(
            Primitive("line", ((x0, y0 + i * row), (x1, y0 + i * row)),
                      width=style.line_dimension, layer="border")
        )

    def cell(
        col_x: float, col_w: float, row_y: float, label: str, value: str
    ) -> None:
        prims.append(
            Primitive("text", ((col_x + 1.5, row_y + row - label_h - 1.0),),
                      text=label, height=label_h, layer="border")
        )
        # A value wider than its cell is shrunk rather than clipped. Clipping would put
        # text on the sheet that ground truth records in full, which is the one failure a
        # transcription benchmark must never introduce itself.
        h = value_h
        if metrics.width(value, h) > col_w - 3.0:
            h = max(label_h, h * (col_w - 3.0) / metrics.width(value, h))
        prims.append(
            Primitive("text", ((col_x + 1.5, row_y + 1.5),),
                      text=value, height=h, layer="border")
        )

    # Bottom row: part number, revision, sheet size, scale.
    widths = (0.42, 0.16, 0.16, 0.26)
    xs = []
    acc = x0
    for w in widths:
        xs.append((acc, TITLE_BLOCK_WIDTH * w))
        acc += TITLE_BLOCK_WIDTH * w
    for (cx, cw), (label, value) in zip(
        xs,
        (
            ("PART No.", data.part_number),
            ("REV", data.revision),
            ("SIZE", data.sheet_size),
            ("SCALE", data.scale_text),
        ),
        strict=True,
    ):
        cell(cx, cw, y0, label, value)
        if cx > x0:
            prims.append(
                Primitive("line", ((cx, y0), (cx, y0 + row)),
                          width=style.line_dimension, layer="border")
            )

    # Middle row: material, general tolerance, units, projection symbol.
    widths = (0.34, 0.30, 0.16, 0.20)
    xs = []
    acc = x0
    for w in widths:
        xs.append((acc, TITLE_BLOCK_WIDTH * w))
        acc += TITLE_BLOCK_WIDTH * w
    for (cx, cw), (label, value) in zip(
        xs[:3],
        (
            ("MATERIAL", data.material),
            ("GENERAL TOL.", data.general_tolerance),
            ("UNITS", data.units.upper()),
        ),
        strict=True,
    ):
        cell(cx, cw, y0 + row, label, value)
        if cx > x0:
            prims.append(
                Primitive("line", ((cx, y0 + row), (cx, y0 + 2 * row)),
                          width=style.line_dimension, layer="border")
            )
    px, pw = xs[3]
    prims.append(
        Primitive("line", ((px, y0 + row), (px, y0 + 2 * row)),
                  width=style.line_dimension, layer="border")
    )
    prims.extend(
        _projection_symbol(px + pw / 2, y0 + row * 1.5, row * 0.5, layout.projection, style)
    )

    # Top row: the title, and the default surface finish beside it if the style has one.
    cell(x0, TITLE_BLOCK_WIDTH * 0.72, y0 + 2 * row, "TITLE", data.title)
    if data.surface_finish_default:
        sx = x0 + TITLE_BLOCK_WIDTH * 0.72
        prims.append(
            Primitive("line", ((sx, y0 + 2 * row), (sx, y0 + 3 * row)),
                      width=style.line_dimension, layer="border")
        )
        cell(sx, TITLE_BLOCK_WIDTH * 0.28, y0 + 2 * row, "FINISH", data.surface_finish_default)

    return tuple(prims)
