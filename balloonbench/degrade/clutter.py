"""Semantic clutter: the marks a person leaves on a drawing.

SPEC.md section 8 calls these the highest-value degradations, and it is right for a reason
worth being explicit about. Noise and blur make a drawing *harder to read*; clutter makes it
harder to *interpret*. A revision cloud says some region changed. An OBSOLETE stamp says the
whole sheet is void. And a previous ballooning attempt -- hand-drawn circles with numbers
already on the page -- puts something on the image that looks exactly like the output the
model is being asked to produce. That last one is extremely common in practice and is the
single most confusing thing a vision model encounters on a real sheet.

Two rules govern everything here.

**Clutter never moves a box.** These are marks laid onto the paper, so geometry is
untouched and the ground truth passes through unchanged.

**Clutter that destroys a callout removes it from the ground truth.** A torn corner or a
punch hole takes paper away, and a characteristic that was printed on the missing paper is
no longer on the sheet. Keeping it would assert a callout a reader cannot find -- the
opposite failure from a box containing no ink, and just as wrong. :func:`_remove_covered`
is where that happens, and it is deliberately applied only by the transforms that actually
remove paper; a translucent stamp obscures without destroying and keeps everything.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from balloonbench.degrade.base import Sample

__all__ = [
    "handwritten_note",
    "photocopier_edge",
    "previous_ballooning",
    "punch_holes",
    "red_pen_correction",
    "revision_cloud",
    "stamp",
    "torn_corner",
]

#: The vendored handwriting face, used for anything a person wrote. Never a system font:
#: the reproducibility rule applies to degraded images exactly as it does to clean ones.
HAND_FONT = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "caveat" / "Caveat.ttf"

#: Wording seen on controlled-document stamps. Deliberately includes both statuses that
#: invalidate a sheet and statuses that do not, because telling them apart is a genuine
#: reading-comprehension task and a model that treats every stamp alike will get it wrong.
STAMP_TEXTS: tuple[str, ...] = (
    "APPROVED",
    "CONTROLLED COPY",
    "OBSOLETE",
    "FOR REFERENCE ONLY",
    "UNCONTROLLED COPY",
    "SUPERSEDED",
)

#: Margin notes a checker might leave. Short, imperative, and about the part -- not lorem
#: ipsum, because a model that reads them should find something a person would have written.
NOTE_TEXTS: tuple[str, ...] = (
    "check with QA",
    "as per rev B",
    "confirm fit",
    "see note 3",
    "typ. both ends",
    "verify on CMM",
    "was 0.2",
    "deburr all edges",
)

#: Fraction of a box that must be destroyed before its characteristic is dropped. Below
#: this a callout is damaged but still identifiable, which is exactly the hard case the
#: benchmark should contain rather than quietly delete.
COVERAGE_TO_DROP = 0.55


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(HAND_FONT), size)


def _overlay(sample: Sample) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """A transparent layer to draw on, so translucency composites correctly."""
    layer = Image.new("RGBA", sample.image.size, (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer)


def _composite(sample: Sample, layer: Image.Image, name: str) -> Sample:
    merged = Image.alpha_composite(sample.image.convert("RGBA"), layer).convert("RGB")
    return sample.with_image(merged, name)


def _remove_covered(sample: Sample, regions: list[tuple[float, float, float, float]]) -> Sample:
    """Drop characteristics and datums whose box is mostly inside a destroyed region."""

    def covered(box) -> float:
        total = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
        lost = 0.0
        for region in regions:
            w = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
            h = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
            lost += w * h
        return min(lost / total, 1.0)

    drawing = sample.drawing
    datums = [d for d in drawing.datums if covered(d.bbox) < COVERAGE_TO_DROP]
    labels = {d.label for d in datums}
    kept = [
        c
        for c in drawing.characteristics
        if covered(c.bbox) < COVERAGE_TO_DROP
        and all(ref.label in labels for ref in c.datum_refs)
    ]
    for new_id, c in enumerate(kept, start=1):
        c.id = new_id
    drawing.datums = datums
    drawing.characteristics = kept
    return sample


# --- marks that add ink ---------------------------------------------------------------------


def stamp(sample: Sample, rng: np.random.Generator) -> Sample:
    """A rotated, semi-transparent rubber stamp.

    Drawn on its own layer and rotated there, then composited, so the stamp can sit at an
    angle over the drawing without the drawing being rotated with it. Its alpha is well
    below opaque: a real stamp is ink pressed over ink, and whatever it covers stays partly
    legible -- which is why this transform does not remove anything from the ground truth.
    """
    width, height = sample.image.size
    text = str(rng.choice(STAMP_TEXTS))
    size = int(min(width, height) * float(rng.uniform(0.05, 0.085)))
    font = _font(size)

    box = font.getbbox(text)
    pad = size // 2
    tile = Image.new(
        "RGBA", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(tile)
    colour = (
        (int(rng.integers(150, 205)), int(rng.integers(20, 55)), int(rng.integers(20, 60)))
        if rng.random() < 0.7
        else (int(rng.integers(20, 60)), int(rng.integers(45, 95)), int(rng.integers(130, 190)))
    )
    alpha = int(rng.integers(90, 150))
    draw.rectangle(
        [pad // 2, pad // 2, tile.width - pad // 2, tile.height - pad // 2],
        outline=(*colour, alpha),
        width=max(2, size // 12),
    )
    draw.text((pad - box[0], pad - box[1]), text, font=font, fill=(*colour, alpha))

    tile = tile.rotate(
        float(rng.uniform(-28, 28)), expand=True, resample=Image.Resampling.BICUBIC
    )
    layer, _ = _overlay(sample)
    x = int(rng.uniform(0.08, 0.72) * width)
    y = int(rng.uniform(0.08, 0.78) * height)
    layer.alpha_composite(tile, (x, y))
    return _composite(sample, layer, "stamp")


def revision_cloud(sample: Sample, rng: np.random.Generator) -> Sample:
    """A scalloped cloud around a changed region, with its revision triangle.

    The scallops are arcs around an ellipse rather than a wobbly line, because that is what
    the convention actually is and because a model can learn the shape. The triangle with a
    revision letter beside it is what makes the cloud mean something.
    """
    width, height = sample.image.size
    layer, draw = _overlay(sample)
    colour = (190, 40, 40, 210)

    cx = float(rng.uniform(0.2, 0.8)) * width
    cy = float(rng.uniform(0.2, 0.8)) * height
    rx = float(rng.uniform(0.06, 0.16)) * width
    ry = float(rng.uniform(0.05, 0.13)) * height
    bumps = int(rng.integers(14, 26))
    scallop = max(6.0, min(rx, ry) * 0.34)

    for i in range(bumps):
        angle = 2 * math.pi * i / bumps
        px = cx + rx * math.cos(angle)
        py = cy + ry * math.sin(angle)
        draw.arc(
            [px - scallop, py - scallop, px + scallop, py + scallop],
            start=math.degrees(angle) - 150,
            end=math.degrees(angle) + 30,
            fill=colour,
            width=max(2, int(min(width, height) * 0.0016)),
        )

    letter = str(rng.choice(list("BCDEF")))
    size = int(min(width, height) * 0.028)
    tx, ty = cx + rx * 0.85, cy - ry * 1.05
    draw.polygon(
        [
            (tx, ty - size),
            (tx - size * 0.9, ty + size * 0.7),
            (tx + size * 0.9, ty + size * 0.7),
        ],
        outline=colour,
        width=2,
    )
    draw.text((tx - size * 0.3, ty - size * 0.45), letter, font=_font(size), fill=colour)
    return _composite(sample, layer, "revision_cloud")


def handwritten_note(sample: Sample, rng: np.random.Generator) -> Sample:
    """A margin note, written glyph by glyph with a jittering baseline.

    Each character is placed individually with its own small vertical offset and rotation.
    Drawing the string in one call would give a perfectly level line of a handwriting face,
    which reads as a font rather than as writing; the jitter is what sells it, and SPEC.md
    section 8 asks for it by name.
    """
    width, height = sample.image.size
    layer, draw = _overlay(sample)
    text = str(rng.choice(NOTE_TEXTS))
    size = int(min(width, height) * float(rng.uniform(0.018, 0.03)))
    font = _font(size)
    colour = (
        (25, 35, 90, 235) if rng.random() < 0.5 else (30, 30, 30, 235)
    )

    # Margins, not the middle of the sheet: a note goes where there is room to write.
    if rng.random() < 0.5:
        x = float(rng.uniform(0.02, 0.12)) * width
    else:
        x = float(rng.uniform(0.62, 0.86)) * width
    y = float(rng.uniform(0.08, 0.88)) * height

    for char in text:
        offset = float(rng.normal(0.0, size * 0.07))
        draw.text((x, y + offset), char, font=font, fill=colour)
        x += font.getlength(char) * float(rng.uniform(0.94, 1.06))
    return _composite(sample, layer, "handwritten_note")


def red_pen_correction(sample: Sample, rng: np.random.Generator) -> Sample:
    """A struck-through value with a replacement written above it, in red.

    Aimed at a real callout rather than at empty paper: a correction that lands nowhere is
    only clutter, while one crossing a dimension creates the genuinely hard question of
    which number the sheet now specifies. The characteristic is *kept* in the ground truth,
    because the printed dimension is still what the drawing formally states.
    """
    if not sample.drawing.characteristics:
        return sample
    layer, draw = _overlay(sample)
    target = sample.drawing.characteristics[
        int(rng.integers(0, len(sample.drawing.characteristics)))
    ]
    x0, y0, x1, y1 = target.bbox
    colour = (200, 30, 30, 235)
    thickness = max(2, int((y1 - y0) * 0.12))

    draw.line(
        [(x0 - 4, y1 - (y1 - y0) * 0.45), (x1 + 4, y0 + (y1 - y0) * 0.35)],
        fill=colour,
        width=thickness,
    )
    size = max(10, int((y1 - y0) * 1.1))
    replacement = f"{float(rng.uniform(1, 90)):.1f}"
    draw.text((x0, y0 - size * 1.15), replacement, font=_font(size), fill=colour)
    return _composite(sample, layer, "red_pen_correction")


def previous_ballooning(sample: Sample, rng: np.random.Generator) -> Sample:
    """Hand-drawn balloons with numbers, from an earlier inspection of the same sheet.

    The most valuable single item in this module. The circles are drawn as wobbling
    polygons rather than true ellipses so they read as hand-drawn, and they are placed
    *on* real callouts, because that is where a person would have put them. A model that
    has learned to output balloons will happily report these as its own findings.
    """
    characteristics = sample.drawing.characteristics
    if not characteristics:
        return sample
    layer, draw = _overlay(sample)
    colour = (
        (25, 60, 160, 225) if rng.random() < 0.6 else (200, 40, 40, 225)
    )
    count = int(rng.integers(3, min(9, len(characteristics) + 1)))
    chosen = rng.choice(len(characteristics), size=count, replace=False)

    for n, index in enumerate(sorted(chosen), start=1):
        box = characteristics[int(index)].bbox
        radius = max(9.0, (box[3] - box[1]) * 0.95)
        cx = box[0] - radius * 1.25
        cy = (box[1] + box[3]) / 2
        points = []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            wobble = radius * float(rng.uniform(0.88, 1.12))
            points.append((cx + wobble * math.cos(angle), cy + wobble * math.sin(angle)))
        draw.line([*points, points[0]], fill=colour, width=max(2, int(radius * 0.16)))
        size = max(9, int(radius * 1.25))
        draw.text(
            (cx - radius * 0.4, cy - radius * 0.78), str(n), font=_font(size), fill=colour
        )

    return _composite(sample, layer, "previous_ballooning")


def photocopier_edge(sample: Sample, rng: np.random.Generator) -> Sample:
    """The dark band a copier leaves where the lid did not close over the page edge."""
    width, height = sample.image.size
    array = np.asarray(sample.image, dtype=np.float32)
    band = int(min(width, height) * float(rng.uniform(0.01, 0.035)))
    depth = float(rng.uniform(0.35, 0.75))

    ramp = np.linspace(1.0 - depth, 1.0, band, dtype=np.float32)
    for side in rng.choice(4, size=int(rng.integers(1, 3)), replace=False):
        if side == 0:
            array[:band, :, :] *= ramp[:, None, None]
        elif side == 1:
            array[-band:, :, :] *= ramp[::-1][:, None, None]
        elif side == 2:
            array[:, :band, :] *= ramp[None, :, None]
        else:
            array[:, -band:, :] *= ramp[::-1][None, :, None]

    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    return sample.with_image(image.filter(ImageFilter.GaussianBlur(0.6)), "photocopier_edge")


# --- marks that remove paper ------------------------------------------------------------------


def punch_holes(sample: Sample, rng: np.random.Generator) -> Sample:
    """Filing holes down the binding edge, and sometimes a staple in the corner."""
    width, height = sample.image.size
    layer, draw = _overlay(sample)
    regions: list[tuple[float, float, float, float]] = []

    radius = min(width, height) * 0.012
    x = float(rng.uniform(0.012, 0.03)) * width
    count = int(rng.choice([2, 3, 4]))
    for i in range(count):
        y = height * (i + 1) / (count + 1)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(35, 35, 35, 255))
        regions.append((x - radius, y - radius, x + radius, y + radius))

    if rng.random() < 0.5:
        sx, sy = width * 0.035, height * 0.05
        length = min(width, height) * 0.012
        draw.line([(sx, sy), (sx + length, sy + length * 0.4)], fill=(60, 60, 60, 255), width=3)
        regions.append((sx, sy, sx + length, sy + length))

    out = _composite(sample, layer, "punch_holes")
    return _remove_covered(out, regions)


def torn_corner(sample: Sample, rng: np.random.Generator) -> Sample:
    """A missing corner, torn along a ragged line.

    Paper is genuinely gone, so anything printed there goes with it -- both from the image
    and from the ground truth. The tear line is a jagged polyline rather than a straight
    cut, because a clean diagonal reads as a crop and a crop is not a defect a sheet has.
    """
    width, height = sample.image.size
    layer, draw = _overlay(sample)
    span = min(width, height) * float(rng.uniform(0.06, 0.16))
    corner = int(rng.integers(0, 4))
    cx, cy = (0.0, 0.0) if corner in (0, 3) else (float(width), 0.0)
    if corner >= 2:
        cy = float(height)

    steps = 9
    points = [(cx, cy)]
    for i in range(steps + 1):
        t = i / steps
        jitter = float(rng.uniform(-0.18, 0.18)) * span
        px = cx + (span * (1 - t) + jitter) * (1 if cx == 0 else -1)
        py = cy + (span * t + jitter) * (1 if cy == 0 else -1)
        points.append((px, py))
    draw.polygon(points, fill=(255, 255, 255, 255))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    region = (min(xs), min(ys), max(xs), max(ys))
    out = _composite(sample, layer, "torn_corner")
    return _remove_covered(out, [region])
