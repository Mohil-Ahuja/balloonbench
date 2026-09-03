"""GD&T and dimensioning glyphs, and the text of a feature control frame.

Two things live here, and they are deliberately together.

**The glyph table.** Every geometric characteristic symbol has a Unicode codepoint, and
``scripts/check_env.py`` asserts on every run that the vendored osifont contains all of
them. A missing glyph does not raise -- it renders as a tofu box -- so the mapping and the
coverage check must name the same codepoints. :data:`GTOL_GLYPH` is the single definition
both use.

**Frame composition.** ``Characteristic.raw_text`` in the schema is what a reader would
transcribe off the sheet, and it has to agree exactly with the structured fields beside it,
or ground truth contradicts itself. :func:`feature_control_frame` builds the string from
those fields rather than letting a caller pass one in, so the two cannot drift.

ASME Y14.5 and ISO 1101 are copyrighted. The conventions here are implemented from their
public description; no table, figure or wording is reproduced. Symbol *codepoints* are from
the Unicode standard, which is not the standards bodies' text.
"""

from __future__ import annotations

__all__ = [
    "DIM_GLYPH",
    "GTOL_GLYPH",
    "MODIFIER_GLYPH",
    "datum_label_text",
    "feature_control_frame",
    "format_value",
    "tolerance_text",
]

#: Geometric characteristic symbols. Keys are the schema's ``GtolSymbol`` values, so a new
#: symbol in the schema that is not added here fails loudly at render rather than drawing
#: an empty frame.
GTOL_GLYPH: dict[str, str] = {
    "straightness": "⏤",
    "flatness": "⏥",
    "circularity": "○",
    "cylindricity": "⌭",
    "profile_surface": "⌒",
    "perpendicularity": "⟂",
    "parallelism": "∥",
    "angularity": "∠",
    "position": "⌖",
    "concentricity": "◎",
    "symmetry": "⌯",
    "circular_runout": "↗",
    "total_runout": "⌰",
}

#: Material condition and frame modifiers. ``RFS`` -- regardless of feature size -- is the
#: default in current practice and is written by *omitting* a modifier, so it is
#: deliberately absent: emitting a glyph for it would put a symbol on the sheet that a
#: modern drawing does not carry.
MODIFIER_GLYPH: dict[str, str] = {
    "MMC": "Ⓜ",
    "LMC": "Ⓛ",
    "free_state": "Ⓕ",
    "projected": "Ⓟ",
}

#: Dimensioning symbols that prefix or suffix a value.
DIM_GLYPH: dict[str, str] = {
    "diameter": "⌀",
    "radius": "R",
    "counterbore": "⌴",
    "countersink": "⌵",
    "depth": "↧",
    "square": "□",
    "arc": "⌒",
    "degree": "°",
    "plus_minus": "±",
    "micro": "µ",
}


def format_value(value: float, decimals: int, *, trailing_zeros: bool = True) -> str:
    """Format a dimension value in a house style's number convention.

    >>> format_value(44.0, 2)
    '44.00'
    >>> format_value(44.0, 2, trailing_zeros=False)
    '44'
    >>> format_value(-0.0, 2)
    '0.00'

    ``trailing_zeros`` is a real split in practice: ISO drawings tend to write ``44``,
    ASME-influenced ones ``44.00``, and the difference changes what a model has to parse.
    The negative-zero case is not hypothetical -- a symmetric tolerance computed as
    ``-upper`` produces ``-0.0``, which would otherwise print as ``-0.00`` on the sheet.
    """
    text = f"{value + 0.0:.{decimals}f}"
    if not trailing_zeros and "." in text:
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-"):
            text = "0"
    return text


def tolerance_text(
    nominal: float,
    upper: float | None,
    lower: float | None,
    style: str,
    *,
    decimals: int = 2,
    tol_decimals: int = 2,
    trailing_zeros: bool = True,
    prefix: str = "",
    fit_class: str | None = None,
) -> str:
    """The dimension text as it appears on the sheet, in one of the schema's tol styles.

    >>> tolerance_text(44, 0.05, -0.05, "bilateral", decimals=0, tol_decimals=2)
    '44±0.05'
    >>> tolerance_text(44, 0.05, 0.0, "unilateral", decimals=0, tol_decimals=2)
    '44 +0.05/-0.00'
    >>> tolerance_text(44, 0.05, -0.05, "limit", decimals=2, tol_decimals=2)
    '44.05/43.95'
    >>> tolerance_text(44, 0.025, 0.0, "fit", decimals=0, fit_class="H7", prefix='⌀')
    '⌀44 H7'
    >>> tolerance_text(44, None, None, "basic", decimals=0)
    '44'

    ``upper`` and ``lower`` are **deviations from nominal**, signed, matching the schema.
    The limit style is the one place they are turned into absolute values, and it is
    exactly the style that a naive extractor reports as a nominal of 44.05 -- which is why
    the benchmark samples it.
    """

    def fmt(v: float, d: int) -> str:
        return format_value(v, d, trailing_zeros=trailing_zeros)

    def signed(v: float, d: int, *, zero_sign: str) -> str:
        """A deviation with its sign always written.

        A zero deviation still carries a sign on a drawing: ``+0.05 / -0`` says the
        tolerance is one-sided, and dropping the minus turns it into something a reader
        cannot tell apart from a missing lower limit. ``zero_sign`` is therefore the
        direction the *other* deviation is not, not the sign of the number.
        """
        if abs(v) < 10.0 ** -(d + 1):
            return zero_sign + fmt(abs(v), d)
        return ("+" if v > 0 else "-") + fmt(abs(v), d)

    base = prefix + fmt(nominal, decimals)

    if style == "basic":
        return base
    if style == "general":
        return base
    if style == "fit":
        if not fit_class:
            raise ValueError("fit style needs a fit_class")
        return f"{base} {fit_class}"

    if upper is None or lower is None:
        raise ValueError(f"{style} style needs both deviations")

    if style == "bilateral":
        # A bilateral callout is only honest when the deviations are symmetric; an
        # asymmetric pair written as a single +/- would misstate the tolerance zone.
        if abs(upper + lower) > 1e-9:
            hi = signed(upper, tol_decimals, zero_sign="+")
            lo = signed(lower, tol_decimals, zero_sign="-")
            return f"{base} {hi}/{lo}"
        return f"{base}{DIM_GLYPH['plus_minus']}{fmt(abs(upper), tol_decimals)}"
    if style == "unilateral":
        hi = signed(upper, tol_decimals, zero_sign="+")
        lo = signed(lower, tol_decimals, zero_sign="-")
        return f"{base} {hi}/{lo}"
    if style == "limit":
        hi = fmt(nominal + upper, max(decimals, tol_decimals))
        lo = fmt(nominal + lower, max(decimals, tol_decimals))
        return f"{prefix}{hi}/{lo}"
    raise ValueError(f"unknown tolerance style {style!r}")


def datum_label_text(label: str, style: str = "boxed") -> str:
    """A datum feature symbol's text.

    >>> datum_label_text('A')
    'A'
    >>> datum_label_text('A', 'dashed')
    '-A-'

    The ``dashed`` form is the older convention. It still turns up on drawings that have
    been revised rather than redrawn for decades, which is precisely the population a
    benchmark should contain, so it is one of the sampled house-style axes.
    """
    if style == "dashed":
        return f"-{label}-"
    return label


def feature_control_frame(
    symbol: str,
    value: float,
    *,
    zone_prefix: str = "",
    material_modifier: str | None = None,
    datum_refs: tuple[tuple[str, str | None], ...] = (),
    decimals: int = 3,
    trailing_zeros: bool = True,
    separator: str = "|",
) -> str:
    """Compose the readable text of a feature control frame.

    >>> feature_control_frame('position', 0.25, zone_prefix='⌀',
    ...                       material_modifier='MMC',
    ...                       datum_refs=(('A', None), ('B', 'MMC')), decimals=2)
    '⌖|⌀0.25Ⓜ|A|BⓂ'
    >>> feature_control_frame('flatness', 0.05, decimals=2)
    '⏥|0.05'

    The compartment order -- characteristic, then tolerance with its modifier, then datum
    references in precedence order -- is the order a reader scans, and the order the
    schema's fields are validated in. ``separator`` exists because the frame is drawn as
    boxes, not pipes; the pipe is how the same frame is written into ``raw_text`` so a
    string comparison against a model's output is meaningful.
    """
    if symbol not in GTOL_GLYPH:
        raise KeyError(f"no glyph for geometric characteristic {symbol!r}")

    tol = zone_prefix + format_value(value, decimals, trailing_zeros=trailing_zeros)
    if material_modifier:
        if material_modifier not in MODIFIER_GLYPH:
            raise KeyError(f"no glyph for modifier {material_modifier!r}")
        tol += MODIFIER_GLYPH[material_modifier]

    parts = [GTOL_GLYPH[symbol], tol]
    for label, modifier in datum_refs:
        parts.append(label + (MODIFIER_GLYPH[modifier] if modifier else ""))
    return separator.join(parts)


def frame_compartments(
    symbol: str,
    value: float,
    *,
    zone_prefix: str = "",
    material_modifier: str | None = None,
    datum_refs: tuple[tuple[str, str | None], ...] = (),
    decimals: int = 3,
    trailing_zeros: bool = True,
) -> tuple[str, ...]:
    """The same frame, split into the compartments the renderer draws as boxes.

    Kept beside :func:`feature_control_frame` and sharing its arguments so the drawn frame
    and the transcribed ``raw_text`` can never disagree about content.
    """
    return tuple(
        feature_control_frame(
            symbol,
            value,
            zone_prefix=zone_prefix,
            material_modifier=material_modifier,
            datum_refs=datum_refs,
            decimals=decimals,
            trailing_zeros=trailing_zeros,
            separator="\x00",
        ).split("\x00")
    )
