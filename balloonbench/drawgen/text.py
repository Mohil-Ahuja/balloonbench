"""Text metrics from the vendored font.

Bounding boxes are the product BalloonBench sells. A box that is a few percent too narrow
still contains ink and still passes a naive check, but it costs real IoU against a model's
prediction, and it does so systematically -- every dimension on every drawing, biased the
same way. So the box around a string must come from the font's own advance widths, not from
a guess like "0.6 times the character height".

The same metrics are used by :mod:`balloonbench.drawgen.annotate` to compute the box and by
:mod:`balloonbench.drawgen.render` to draw the string. That is the point of putting them
here: two implementations of "how wide is this text" would drift, and the drift would be
invisible until someone measured IoU against a hand-labelled sheet.

Widths are cached per font because ``TTFont`` parsing is slow enough to dominate generation
of a few hundred drawings otherwise.
"""

from __future__ import annotations

import functools
from pathlib import Path

__all__ = ["FontMetrics", "font_path", "metrics_for"]

#: Root of the vendored font tree. Nothing outside it may be used: SPEC.md section 3
#: requires reproducible output across machines, and a system font is by definition not.
FONT_ROOT = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"


def font_path(relative: str) -> Path:
    """Resolve a style's font, refusing anything outside the vendored tree."""
    path = (FONT_ROOT / relative).resolve()
    if not path.is_relative_to(FONT_ROOT.resolve()):
        raise ValueError(f"font {relative!r} escapes the vendored font tree")
    if not path.exists():
        raise FileNotFoundError(f"vendored font not found: {path}")
    return path


class FontMetrics:
    """Advance widths and vertical extents for one font, in em-relative units.

    Everything is returned as a multiple of the character height, so a caller scales by its
    own text height rather than re-reading the font at each size.
    """

    def __init__(self, path: Path) -> None:
        from fontTools.ttLib import TTFont

        self.path = path
        font = TTFont(path, lazy=True)
        upem = font["head"].unitsPerEm
        hmtx = font["hmtx"]
        cmap = font.getBestCmap()

        self._advance: dict[str, float] = {}
        for codepoint, glyph in cmap.items():
            try:
                advance = hmtx[glyph][0]
            except KeyError:
                continue
            self._advance[chr(codepoint)] = advance / upem

        os2 = font.get("OS/2")
        self.ascent = (os2.sTypoAscender if os2 else font["hhea"].ascent) / upem
        self.descent = abs(os2.sTypoDescender if os2 else font["hhea"].descent) / upem
        # A glyph the font has no entry for is drawn as .notdef, whose advance is the
        # font's default. Falling back to the width of a digit keeps a box sane if that
        # ever happens; check_env.py exists so that it does not.
        self._fallback = self._advance.get("0", 0.5)
        self.missing: set[str] = set()
        font.close()

    def advance(self, char: str) -> float:
        try:
            return self._advance[char]
        except KeyError:
            self.missing.add(char)
            return self._fallback

    def width(self, text: str, height: float) -> float:
        """Width of ``text`` drawn at character height ``height``, in the same units.

        ``height`` is the *character height* in the drafting sense -- the cap height a
        drafting standard specifies -- not the font's point size. The two differ by the
        font's own proportions, which is why :meth:`scale_for_height` exists and why every
        caller must go through it rather than passing a height to a PDF library as a size.
        """
        size = self.scale_for_height(height)
        return sum(self.advance(c) for c in text) * size

    def scale_for_height(self, height: float) -> float:
        """The font size that renders a capital at ``height``.

        ISO 3098 specifies lettering by character height. A PDF library's font size is the
        em size, which is larger. Using the requested height directly as the size draws
        text noticeably smaller than the standard asks for, and -- worse here -- would make
        every text bounding box disagree with what is drawn.
        """
        return height / self.cap_ratio

    @functools.cached_property
    def cap_ratio(self) -> float:
        """Cap height as a fraction of the em, measured from the font rather than assumed."""
        from fontTools.ttLib import TTFont

        font = TTFont(self.path, lazy=True)
        upem = font["head"].unitsPerEm
        ratio = None
        if "OS/2" in font and getattr(font["OS/2"], "sCapHeight", 0):
            ratio = font["OS/2"].sCapHeight / upem
        else:
            glyf = font.get("glyf")
            cmap = font.getBestCmap()
            name = cmap.get(ord("H"))
            if glyf is not None and name and name in glyf:
                ratio = glyf[name].yMax / upem
        font.close()
        if not ratio:
            # 0.7 is the usual proportion for a technical face; only reached if the font
            # reports neither a cap height nor an 'H' outline.
            ratio = 0.7
        return float(ratio)

    def text_box(
        self, text: str, height: float, origin: tuple[float, float], anchor: str = "left"
    ) -> tuple[float, float, float, float]:
        """The box a string occupies, as ``(x0, y0, x1, y1)`` around a baseline origin.

        The box spans the full ascent and descent, not just the cap height: a dimension
        text containing a comma or a subscripted tolerance has ink below the baseline, and
        a box drawn to the baseline would clip it.
        """
        size = self.scale_for_height(height)
        w = self.width(text, height)
        x = origin[0]
        if anchor == "middle":
            x -= w / 2
        elif anchor == "right":
            x -= w
        return (x, origin[1] - self.descent * size, x + w, origin[1] + self.ascent * size)


@functools.lru_cache(maxsize=8)
def metrics_for(relative: str) -> FontMetrics:
    """Metrics for a vendored font, cached by its relative path."""
    return FontMetrics(font_path(relative))
