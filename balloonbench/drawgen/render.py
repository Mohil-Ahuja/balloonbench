"""Rendering: primitives to PDF, DXF, PNG and a QA overlay.

This module draws and nothing else. Every position, string and box reaching it was decided
by :mod:`balloonbench.drawgen.annotate`, and the renderer's job is to put ink exactly where
it was told. That division is what makes ground truth trustworthy: a renderer that decided
anything -- where to break a line for text, how wide a frame should be -- would be making
decisions the ground truth never saw.

**[DEVIATION from SPEC.md section 7.4]** The spec's pipeline is ezdxf to SVG to PDF. We draw
the PDF directly with reportlab and emit the DXF as a parallel artifact from the same
primitives. The reason is the bbox guarantee. Going through SVG means text is measured by
whichever backend converts it, and its idea of a string's width has to agree exactly with
the metrics :mod:`balloonbench.drawgen.text` used to compute the ground-truth box. Drawing
directly means both the box and the glyphs come from the same font file read the same way,
so the guarantee holds by construction instead of by coincidence. The DXF artifact the spec
wanted for the vector-hybrid baseline is still produced, from the same primitive list, so
the two outputs cannot describe different drawings.

The raster path is deliberately short: reportlab writes a vector PDF whose page is exactly
the sheet size in millimetres, and pypdfium2 rasterises it at a requested DPI. Because the
page size is the sheet size, pixel coordinates are ``sheet_mm * dpi / 25.4`` -- which is
exactly what :meth:`~balloonbench.drawgen.views.SheetLayout.to_pixel` computes, with no
second scale factor to get wrong.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from balloonbench.drawgen.annotate import Annotation, Primitive
from balloonbench.drawgen.styles import HouseStyle
from balloonbench.drawgen.text import font_path, metrics_for
from balloonbench.drawgen.views import SheetLayout

__all__ = ["build_primitives", "render_dxf", "render_overlay", "render_pdf", "render_png"]

MM_TO_PT = 72.0 / 25.4

#: Dash patterns in millimetres, by layer. Hidden detail is short-dashed and centre lines
#: are long-dash-dot; both are ISO 128 line types, and getting them visibly different is
#: what lets a reader tell a bore's hidden wall from its axis.
_DASH: dict[str, tuple[float, ...]] = {
    "hidden": (2.5, 1.5),
    "centre": (10.0, 1.5, 1.5, 1.5),
}

_LAYER_WIDTH_ATTR: dict[str, str] = {
    "visible": "line_visible",
    "hidden": "line_hidden",
    "centre": "line_centre",
    "hatch": "line_dimension",
}


def build_primitives(
    layout: SheetLayout,
    style: HouseStyle,
    annotations: list[Annotation],
    chrome: tuple[Primitive, ...] = (),
    decorations: tuple[Primitive, ...] = (),
) -> list[Primitive]:
    """Flatten a sheet into one ordered primitive list.

    Order is draw order: part linework first, then hatching, then annotations, then the
    border and title block on top. Drawing the frame last means a view that overruns its
    area is visibly clipped by nothing -- it overlaps the border, which is exactly the
    signal a human reviewer needs at the manual gate.
    """
    prims: list[Primitive] = []
    for placement in layout.placements:
        for line in placement.view.lines:
            width = getattr(style, _LAYER_WIDTH_ATTR.get(line.layer, "line_visible"))
            prims.append(
                Primitive(
                    "line",
                    tuple(placement.to_sheet(p) for p in line.points),
                    width=width,
                    layer=line.layer,
                )
            )
        for line in placement.hatch:
            prims.append(
                Primitive(
                    "line",
                    tuple(placement.to_sheet(p) for p in line.points),
                    width=style.line_dimension * 0.8,
                    layer="hatch",
                )
            )
        prims.extend(_centre_lines(placement, style))
        if placement.spec.label:
            prims.append(
                Primitive(
                    "text",
                    (placement.label_origin(style.text_height),),
                    text=placement.spec.label,
                    height=style.text_height * 1.15,
                    anchor="middle",
                    layer="annotation",
                )
            )

    prims.extend(decorations)
    for ann in annotations:
        prims.extend(ann.primitives)
    prims.extend(chrome)
    return prims


def _centre_lines(placement, style: HouseStyle) -> list[Primitive]:
    """Centre lines through each view, drawn to a small overshoot past the outline.

    A drawing without centre lines reads as a silhouette rather than as a machined part,
    and every circular feature on a real sheet carries one. They are generated from the
    view's own bounds rather than from features so that a view with no cylindrical feature
    still gets the pair a drafter would draw.
    """
    x0, y0, x1, y1 = placement.sheet_bounds
    over = 4.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [
        Primitive("line", ((x0 - over, cy), (x1 + over, cy)),
                  width=style.line_centre, layer="centre"),
        Primitive("line", ((cx, y0 - over), (cx, y1 + over)),
                  width=style.line_centre, layer="centre"),
    ]


# --- PDF ---------------------------------------------------------------------------------


def _register_font(style: HouseStyle) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    name = "bb-" + Path(style.font).stem
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(font_path(style.font))))
    return name


def render_pdf(
    path: Path,
    layout: SheetLayout,
    style: HouseStyle,
    primitives: list[Primitive],
) -> Path:
    """Draw the primitives to a vector PDF whose page is exactly the sheet."""
    from reportlab.pdfgen import canvas

    font = _register_font(style)
    metrics = metrics_for(style.font)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(
        str(path), pagesize=(layout.width * MM_TO_PT, layout.height * MM_TO_PT)
    )
    c.setLineJoin(1)
    c.setLineCap(1)

    for prim in primitives:
        _draw(c, prim, style, font, metrics)

    c.showPage()
    c.save()
    return path


def _draw(c, prim: Primitive, style: HouseStyle, font: str, metrics) -> None:
    dash = _DASH.get(prim.layer)
    c.setLineWidth(prim.width * MM_TO_PT)
    c.setDash([d * MM_TO_PT for d in dash], 0) if dash else c.setDash([], 0)

    if prim.kind == "line" and len(prim.points) >= 2:
        p = c.beginPath()
        p.moveTo(prim.points[0][0] * MM_TO_PT, prim.points[0][1] * MM_TO_PT)
        for x, y in prim.points[1:]:
            p.lineTo(x * MM_TO_PT, y * MM_TO_PT)
        c.drawPath(p)

    elif prim.kind == "box" and len(prim.points) >= 2:
        (x0, y0), (x1, y1) = prim.points[0], prim.points[1]
        c.rect(
            x0 * MM_TO_PT, y0 * MM_TO_PT,
            (x1 - x0) * MM_TO_PT, (y1 - y0) * MM_TO_PT,
            stroke=1, fill=0,
        )

    elif prim.kind == "triangle" and len(prim.points) >= 3:
        p = c.beginPath()
        p.moveTo(prim.points[0][0] * MM_TO_PT, prim.points[0][1] * MM_TO_PT)
        for x, y in prim.points[1:]:
            p.lineTo(x * MM_TO_PT, y * MM_TO_PT)
        p.close()
        c.drawPath(p, stroke=1, fill=1 if prim.filled else 0)

    elif prim.kind == "circle" and len(prim.points) >= 2:
        (cx, cy), (r, _unused) = prim.points[0], prim.points[1]
        c.circle(cx * MM_TO_PT, cy * MM_TO_PT, r * MM_TO_PT, stroke=1, fill=0)

    elif prim.kind == "dot" and prim.points:
        cx, cy = prim.points[0]
        c.circle(cx * MM_TO_PT, cy * MM_TO_PT, prim.width * MM_TO_PT, stroke=0, fill=1)

    elif prim.kind == "text" and prim.points and prim.text:
        size = metrics.scale_for_height(prim.height) * MM_TO_PT
        x, y = prim.points[0][0] * MM_TO_PT, prim.points[0][1] * MM_TO_PT
        c.setDash([], 0)
        c.setFont(font, size)
        if prim.rotation:
            c.saveState()
            c.translate(x, y)
            c.rotate(prim.rotation)
            _text(c, 0.0, 0.0, prim)
            c.restoreState()
        else:
            _text(c, x, y, prim)


def _text(c, x: float, y: float, prim: Primitive) -> None:
    if prim.anchor == "middle":
        c.drawCentredString(x, y, prim.text)
    elif prim.anchor == "right":
        c.drawRightString(x, y, prim.text)
    else:
        c.drawString(x, y, prim.text)


# --- DXF ---------------------------------------------------------------------------------

_DXF_LAYERS: dict[str, tuple[int, str]] = {
    # (AutoCAD colour index, linetype). Colour 7 is the drawing default.
    "visible": (7, "CONTINUOUS"),
    "hidden": (8, "DASHED"),
    "centre": (4, "CENTER"),
    "hatch": (8, "CONTINUOUS"),
    "annotation": (7, "CONTINUOUS"),
    "border": (7, "CONTINUOUS"),
}


def render_dxf(
    path: Path, layout: SheetLayout, style: HouseStyle, primitives: list[Primitive]
) -> Path:
    """Write the same primitives as a DXF, for the vector-hybrid baseline.

    Emitted from the primitive list rather than re-derived from the model, so the DXF and
    the PDF are guaranteed to be the same drawing. A baseline that reads the DXF is
    therefore being scored against ground truth that describes what it is reading.
    """
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    for name, (colour, linetype) in _DXF_LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, color=colour, linetype=linetype)

    for prim in primitives:
        layer = prim.layer if prim.layer in _DXF_LAYERS else "annotation"
        if prim.kind == "line" and len(prim.points) >= 2:
            msp.add_lwpolyline(prim.points, dxfattribs={"layer": layer})
        elif prim.kind == "box" and len(prim.points) >= 2:
            (x0, y0), (x1, y1) = prim.points[0], prim.points[1]
            msp.add_lwpolyline(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                close=True,
                dxfattribs={"layer": layer},
            )
        elif prim.kind == "triangle" and len(prim.points) >= 3:
            msp.add_lwpolyline(prim.points, close=True, dxfattribs={"layer": layer})
        elif prim.kind == "circle" and len(prim.points) >= 2:
            msp.add_circle(prim.points[0], prim.points[1][0], dxfattribs={"layer": layer})
        elif prim.kind == "text" and prim.text:
            entity = msp.add_text(
                prim.text,
                height=prim.height,
                rotation=prim.rotation,
                dxfattribs={"layer": layer},
            )
            align = {"middle": "MIDDLE_CENTER", "right": "BOTTOM_RIGHT"}.get(
                prim.anchor, "BOTTOM_LEFT"
            )
            entity.set_placement(prim.points[0], align=ezdxf.enums.TextEntityAlignment[align])

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


# --- raster ------------------------------------------------------------------------------


def render_png(pdf_path: Path, png_path: Path, dpi: float) -> tuple[int, int]:
    """Rasterise the PDF at ``dpi`` and return the image size in pixels.

    The scale handed to pdfium is relative to 72 DPI because a PDF user-space unit is a
    point. Since the page is the sheet in millimetres, the result is exactly
    ``sheet_mm * dpi / 25.4`` pixels, which is what
    :meth:`~balloonbench.drawgen.views.SheetLayout.pixel_size` predicts -- the assertion in
    the caller is what keeps those two from ever drifting apart.
    """
    import pypdfium2 as pdfium

    # Closed explicitly rather than left to the garbage collector. pdfium holds native
    # handles, and a generation run that leaks one per drawing prints a warning per document
    # at interpreter exit and holds the buffers open for the whole run.
    doc = pdfium.PdfDocument(pdf_path.read_bytes())
    try:
        image = doc[0].render(scale=dpi / 72.0).to_pil().convert("RGB")
    finally:
        doc.close()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    return image.size


#: Overlay colours, chosen to stay distinguishable on a white sheet and from each other
#: when boxes overlap.
_OVERLAY_COLOURS: dict[str, tuple[int, int, int]] = {
    "dimension": (0, 110, 220),
    "geometric_tolerance": (200, 40, 40),
    "datum": (30, 150, 60),
    "note": (150, 90, 200),
    "surface_finish": (200, 130, 0),
    "thread": (0, 150, 150),
}


def render_overlay(
    png_path: Path,
    overlay_path: Path,
    boxes: list[tuple[str, tuple[float, float, float, float], str]],
) -> Path:
    """Draw ground-truth boxes on the render, for the human QA gate.

    This image is the artifact the manual acceptance gate is judged on, so it shows what
    ground truth *claims* rather than a prettied version of it: a box that is off by five
    millimetres must look off by five millimetres here.
    """
    from PIL import Image, ImageDraw

    image = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for kind, box, label in boxes:
        colour = _OVERLAY_COLOURS.get(kind, (120, 120, 120))
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colour, width=2)
        if label:
            draw.text((box[0] + 2, max(0.0, box[1] - 11)), label, fill=colour)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path)
    return overlay_path


def ink_in_box(
    png_path: Path, box: tuple[float, float, float, float], threshold: int = 235
) -> bool:
    """Whether a pixel-space box contains any non-white pixel.

    The acceptance test SPEC.md section 7.4 asks for. A generous threshold, because
    antialiasing on a thin line at 150 DPI leaves grey rather than black, and a strict test
    would fail on a correctly placed box around a fine dimension line.
    """
    from PIL import Image

    with Image.open(png_path) as image:
        w, h = image.size
        x0 = max(0, int(math.floor(box[0])))
        y0 = max(0, int(math.floor(box[1])))
        x1 = min(w, int(math.ceil(box[2])))
        y1 = min(h, int(math.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            return False
        crop = image.convert("L").crop((x0, y0, x1, y1))
        # getextrema() rather than min(getdata()): the same answer, without materialising a
        # pixel sequence, and getdata() is deprecated for removal in Pillow 14.
        darkest, _lightest = crop.getextrema()
        return darkest < threshold


def png_bytes(pdf_path: Path, dpi: float) -> bytes:
    """The PNG for a PDF, in memory. Used by tests that do not want a temp file."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path.read_bytes())
    buf = io.BytesIO()
    try:
        doc[0].render(scale=dpi / 72.0).to_pil().convert("RGB").save(buf, format="PNG")
    finally:
        doc.close()
    return buf.getvalue()
