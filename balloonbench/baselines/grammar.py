"""A recursive-descent parser for the strings written on an engineering drawing.

SPEC.md section 11 is explicit that this must be a real parser rather than regex soup, and
the reason becomes obvious the moment the strings get interesting. ``⌀44 +0.05/-0.00`` and
``⌀44.05/43.95`` differ in what the two numbers *mean*, not in how they look; ``4× ⌀12↧20``
nests a count, a diameter and a depth; a feature control frame is a sequence of compartments
whose legality depends on the symbol in the first one. Each of those is a grammar with
structure, and a pattern that matches one of them will quietly mis-read the next.

The parser is used by two baselines -- ``vector_hybrid`` at M6 and the detector's OCR
pipeline at M9 -- so its output is the schema's field names rather than a private structure.
The two baselines must not be able to disagree about what a string means.

**On accepting variants.** Real drawings write the same thing many ways: ``Ø``, ``⌀`` and
``DIA``; ``±`` and ``+/-``; ``4X``, ``4x`` and ``4×``; ``M`` in a circle, ``(M)`` and
``(MMC)``. All are accepted, because a parser that only read our own renderer's output would
be tested in a circle and useless on a real sheet. What is *not* accepted is anything
ambiguous: the parser raises rather than guessing, because a wrong parse enters the results
as a confident answer while a refusal is visibly a refusal and can be routed to the model.

**On what it does not do.** It reads a string. It has no idea where the string sat on the
sheet, which view it annotated, or whether the value was drawn inside a box -- so ``view``,
``bbox`` and ``is_basic`` are the caller's to supply. A basic dimension is a rectangle around
the number, not a mark in the text, and inferring one from the text would be inventing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParseError",
    "ParsedCallout",
    "normalise",
    "parse_callout",
    "parse_many",
]


class ParseError(ValueError):
    """The string is not a callout this grammar recognises.

    Carries the offset it gave up at, which is what makes a failure actionable: "position
    7" points at the character, and the ``vector_hybrid`` baseline logs it before handing
    the string to a model.
    """

    def __init__(self, message: str, text: str, position: int) -> None:
        super().__init__(f"{message} at position {position} in {text!r}")
        self.text = text
        self.position = position


@dataclass
class ParsedCallout:
    """What a string says, in the ground-truth schema's field names."""

    kind: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def as_payload(self) -> dict[str, Any]:
        """The characteristic fields, ready to be merged with a view and a bbox."""
        return {"kind": self.kind, "raw_text": self.raw_text, **self.fields}


# --- normalisation ---------------------------------------------------------------------

#: Every way a drawing writes a symbol, mapped to the one the schema and the renderer use.
#: The ASCII spellings are not hypothetical: ``%%c`` is what AutoCAD stores for a diameter
#: sign, and a DXF exported from a 1990s seat is full of them.
_ALIASES: tuple[tuple[str, str], ...] = (
    ("%%c", "⌀"), ("%%C", "⌀"), ("Ø", "⌀"), ("ø", "⌀"), ("⏀", "⌀"),
    ("%%d", "°"), ("%%D", "°"), ("DEG", "°"),
    ("%%p", "±"), ("%%P", "±"), ("+/-", "±"), ("+-", "±"), ("±", "±"),
    ("(M)", "Ⓜ"), ("(MMC)", "Ⓜ"), ("(L)", "Ⓛ"), ("(LMC)", "Ⓛ"),
    ("(S)", ""), ("(RFS)", ""),  # RFS is the default; writing it changes nothing.
    ("×", "X"), ("x", "X"), ("*", "X"),
    ("THRU ALL", "THRU"), ("THROUGH", "THRU"),
    ("C'BORE", "⌴"), ("CBORE", "⌴"), ("SPOTFACE", "⌴"), ("SF", "⌴"),
    ("C'SINK", "⌵"), ("CSK", "⌵"), ("CSINK", "⌵"),
    ("DEEP", "↧"), ("DP", "↧"),
    ("DIA", "⌀"),
)

#: Geometric characteristic symbols, by glyph and by the shop abbreviations people write
#: when a font lacks the glyph.
_GTOL_NAMES: dict[str, str] = {
    "⏥": "flatness", "FLAT": "flatness", "FLATNESS": "flatness",
    "⏤": "straightness", "STR": "straightness", "STRAIGHTNESS": "straightness",
    "○": "circularity", "◯": "circularity", "CIRC": "circularity",
    "ROUND": "circularity", "ROUNDNESS": "circularity", "CIRCULARITY": "circularity",
    "⌭": "cylindricity", "CYL": "cylindricity", "CYLINDRICITY": "cylindricity",
    "⌓": "profile_surface", "PROF": "profile_surface", "PROFILE": "profile_surface",
    "⊥": "perpendicularity", "PERP": "perpendicularity",
    "PERPENDICULARITY": "perpendicularity", "SQUARENESS": "perpendicularity",
    "∥": "parallelism", "//": "parallelism", "PARA": "parallelism",
    "PARALLELISM": "parallelism",
    "∠": "angularity", "ANG": "angularity", "ANGULARITY": "angularity",
    "⌖": "position", "POS": "position", "TP": "position", "POSITION": "position",
    "TRUE POSITION": "position",
    "◎": "concentricity", "CONC": "concentricity", "CONCENTRICITY": "concentricity",
    "⌯": "symmetry", "SYM": "symmetry", "SYMMETRY": "symmetry",
    "↗": "circular_runout", "RUNOUT": "circular_runout",
    "CIRCULAR RUNOUT": "circular_runout",
    "⌰": "total_runout", "TOTAL RUNOUT": "total_runout", "TIR": "total_runout",
}

_MODIFIERS = {"Ⓜ": "MMC", "Ⓛ": "LMC"}

#: Prefixes that say what kind of size a number is.
_DIM_PREFIX: dict[str, tuple[str, dict[str, Any]]] = {
    "S⌀": ("diameter", {"spherical": True}),
    "SR": ("radius", {"spherical": True}),
    "⌀": ("diameter", {}),
    "R": ("radius", {}),
    "□": ("linear", {"square": True}),
}


def normalise(text: str) -> str:
    """Fold the spelling variants together, leaving the structure alone.

    Case is *not* folded: ``M12`` is a metric thread and ``m`` is a millimetre-ish noise
    character, and datum letters are upper case by definition. Only whitespace is collapsed,
    because a callout wrapped across two lines by the renderer means the same as one on a
    single line.
    """
    out = text.replace("\r", "\n")
    for source, target in _ALIASES:
        out = out.replace(source, target)
    return re.sub(r"\s+", " ", out).strip()


# --- the cursor ------------------------------------------------------------------------


class _Cursor:
    """A position in the string, with the small vocabulary the productions need."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<cursor {self.text[self.pos:]!r}>"

    @property
    def rest(self) -> str:
        return self.text[self.pos :]

    def done(self) -> bool:
        return self.pos >= len(self.text)

    def skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1

    def peek(self, n: int = 1) -> str:
        return self.text[self.pos : self.pos + n]

    def accept(self, literal: str) -> bool:
        self.skip_space()
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def accept_any(self, literals) -> str | None:
        # Longest first, so "S⌀" is never read as "S" followed by a diameter.
        for literal in sorted(literals, key=len, reverse=True):
            if self.accept(literal):
                return literal
        return None

    def fail(self, message: str) -> None:
        raise ParseError(message, self.text, self.pos)


_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)")
_FRACTION = re.compile(r"(\d+)\s*-\s*(\d+)/(\d+)|(\d+)/(\d+)")


def _number(cursor: _Cursor, *, allow_fraction: bool = True) -> float | None:
    """A number, in any of the forms a drawing writes one.

    Fractions come first because ``1/4`` would otherwise read as the number 1 followed by an
    unexpected slash, and an imperial drawing is full of them. A comma decimal is accepted
    only between digits, where it cannot be a separator.
    """
    cursor.skip_space()
    if allow_fraction:
        match = _FRACTION.match(cursor.rest)
        if match:
            cursor.pos += match.end()
            if match.group(1):
                whole, num, den = int(match[1]), int(match[2]), int(match[3])
                return whole + num / den
            return int(match[4]) / int(match[5])

    rest = cursor.rest
    comma = re.match(r"[+-]?\d+,\d+", rest)
    if comma:
        cursor.pos += comma.end()
        return float(comma.group(0).replace(",", "."))

    match = _NUMBER.match(rest)
    if not match:
        return None
    cursor.pos += match.end()
    return float(match.group(0))


def _decimals_of(text: str) -> int:
    match = re.search(r"\d+\.(\d+)", text)
    return len(match.group(1)) if match else 0


# --- feature control frames ------------------------------------------------------------

_DATUM = re.compile(r"[A-Z]{1,2}(?![A-Za-z])")


def _parse_frame(cursor: _Cursor, symbol: str) -> ParsedCallout:
    """``symbol [⌀] value [modifier] [datum [modifier]]...``

    The compartment order is the grammar. A modifier belongs to whatever precedes it -- the
    tolerance in the first compartment, or the datum it follows -- which is why they are
    consumed inside their own production rather than swept up at the end.
    """
    fields: dict[str, Any] = {"gtol_symbol": symbol}

    # A frame is often transcribed with its compartment dividers intact -- "| ⌖ | ⌀0.05 |
    # A |" -- and the bars carry no meaning the structure does not already have.
    cursor.accept("|")
    diametral = cursor.accept("⌀")
    value = _number(cursor, allow_fraction=False)
    if value is None:
        cursor.fail("a feature control frame needs a tolerance value")
    if value <= 0:
        cursor.fail("a tolerance zone must be positive")
    fields["gtol_value"] = value
    fields["gtol_zone"] = "diametral" if diametral else "linear"

    modifier = cursor.accept_any(_MODIFIERS)
    fields["material_modifier"] = _MODIFIERS[modifier] if modifier else None

    refs: list[dict[str, Any]] = []
    while True:
        cursor.skip_space()
        cursor.accept("|")
        cursor.skip_space()
        match = _DATUM.match(cursor.rest)
        if not match:
            break
        cursor.pos += match.end()
        ref_modifier = cursor.accept_any(_MODIFIERS)
        refs.append(
            {
                "label": match.group(0),
                "modifier": _MODIFIERS[ref_modifier] if ref_modifier else None,
            }
        )
        if len(refs) > 3:
            cursor.fail("a frame references at most three datums")
    fields["datum_refs"] = refs

    cursor.skip_space()
    if not cursor.done():
        cursor.fail("unexpected text after the frame")
    return ParsedCallout(kind="geometric_tolerance", fields=fields)


# --- dimensions ------------------------------------------------------------------------


def _parse_tolerance(cursor: _Cursor, nominal: float, text: str) -> dict[str, Any]:
    """Whichever of the four tolerance forms follows the nominal, as signed deviations.

    Deviations, always, even for the limit form -- the schema's R4 says so and the reason is
    that the renderer decides display, not storage. ``44.05/43.95`` is stored as +0.05 and
    -0.05, which is the same statement about the part.
    """
    cursor.skip_space()

    if cursor.accept("±"):
        value = _number(cursor, allow_fraction=False)
        if value is None:
            cursor.fail("± needs a value")
        return {"upper_tol": abs(value), "lower_tol": -abs(value), "tol_style": "bilateral"}

    # Note the emptiness check: ``"" in "+-"`` is true in Python, so a callout that
    # ends after its nominal would otherwise be read as the start of a deviation.
    if cursor.peek() and cursor.peek() in "+-":
        upper = _number(cursor, allow_fraction=False)
        if upper is None:
            cursor.fail("a deviation needs a value")
        cursor.skip_space()
        cursor.accept("/")
        lower = _number(cursor, allow_fraction=False)
        if lower is None:
            cursor.fail("a two-sided deviation needs both values")
        if upper < lower:
            upper, lower = lower, upper
        style = "bilateral" if abs(upper) == abs(lower) and upper > 0 else "unilateral"
        return {"upper_tol": upper, "lower_tol": lower, "tol_style": style}

    # A fit class: a letter for the deviation and a number for the IT grade. Upper case is
    # a hole, lower case a shaft, and the case is the whole distinction -- H7 and h7 are
    # different parts of the same fit.
    match = re.match(
        r"([A-Za-z]{1,2}\d{1,2})(?:/([A-Za-z]{1,2}\d{1,2}))?(?![\w.])", cursor.rest
    )
    if match and not _looks_like_a_word(match.group(0)):
        cursor.pos += match.end()
        return {"fit_class": match.group(1), "tol_style": "fit"}

    if cursor.accept("/"):
        # Limit form: the second number is the lower limit.
        lower_limit = _number(cursor, allow_fraction=False)
        if lower_limit is None:
            cursor.fail("a limit dimension needs its second value")
        upper, lower = max(nominal, lower_limit), min(nominal, lower_limit)
        middle = (upper + lower) / 2
        return {
            "nominal": round(middle, max(3, _decimals_of(text))),
            "upper_tol": round(upper - middle, 6),
            "lower_tol": round(lower - middle, 6),
            "tol_style": "limit",
        }

    return {"tol_style": "general"}


#: Words that would otherwise read as a fit class. ``H7`` is a fit; ``THRU`` is not, and
#: neither is the ``X2`` in a note about two of something.
_WORDS = {"THRU", "TYP", "REF", "MAX", "MIN", "X"}


def _looks_like_a_word(token: str) -> bool:
    return token.upper() in _WORDS


def _starts_a_size(cursor: _Cursor) -> bool:
    """Whether a size prefix and a number follow, without consuming either."""
    rest = cursor.rest.lstrip()
    for prefix in sorted(_DIM_PREFIX, key=len, reverse=True):
        if rest.startswith(prefix) and re.match(r"\s*[.\d]", rest[len(prefix) :]):
            return True
    return False


def _parse_dimension(cursor: _Cursor) -> ParsedCallout:
    """``[count X] [prefix] nominal [tolerance] [depth] [counterbore] [notes]``"""
    fields: dict[str, Any] = {}
    notes: list[str] = []

    prefix = cursor.accept_any(_DIM_PREFIX)
    dim_type, extra = _DIM_PREFIX.get(prefix or "", ("linear", {}))
    notes_from_prefix = {k: v for k, v in extra.items()}

    nominal = _number(cursor)
    if nominal is None:
        cursor.fail("a dimension needs a value")
    fields["nominal"] = nominal

    if cursor.accept("°"):
        dim_type = "angular"
    fields["dim_type"] = dim_type

    fields.update(_parse_tolerance(cursor, nominal, cursor.text))
    if cursor.accept("°"):
        # ``30 ±0.5°`` puts the degree sign after the tolerance, which is common enough.
        fields["dim_type"] = "angular"

    # Trailing feature notes. These are recorded rather than parsed into structure: the
    # schema has no depth field, and inventing one here would be a schema change made in
    # the wrong file.
    while not cursor.done():
        cursor.skip_space()
        if cursor.accept("↧"):
            depth = _number(cursor)
            if depth is None:
                cursor.fail("a depth needs a value")
            notes.append(f"depth {depth:g}")
            continue
        if cursor.accept("⌴") or cursor.accept("⌵"):
            notes.append("counterbore" if cursor.text[cursor.pos - 1] == "⌴" else "countersink")
            continue
        trailing = re.match(r"(\d+)\s*X(?![\w.])", cursor.rest)
        if trailing:
            # "⌀6 THRU 4X" -- the count written after the feature rather than before it.
            cursor.pos += trailing.end()
            fields.setdefault("count", int(trailing.group(1)))
            continue
        if notes and _starts_a_size(cursor):
            # A compound callout: a hole and its counterbore in one string. The second size
            # is recorded as a note rather than split into a characteristic of its own,
            # because on the sheet it is one callout with one balloon number.
            #
            # Only *after* a feature note, and only when a digit follows the prefix. Both
            # guards earn their keep: without the first, "⌀44 ⌀44" would parse as a
            # compound feature instead of the nonsense it is; without the second, the R of
            # "44 REF" would be read as a radius.
            second = cursor.accept_any(_DIM_PREFIX)
            value = _number(cursor)
            if value is None:
                cursor.fail("a second size needs a value")
            notes.append(f"{second}{value:g}")
            continue
        word = re.match(r"[A-Za-z][A-Za-z0-9'\-]*", cursor.rest)
        if not word:
            cursor.fail("unexpected text after the dimension")
        token = word.group(0).upper()
        cursor.pos += word.end()
        if token == "THRU":
            notes.append("through")
        elif token == "TYP":
            notes.append("typical")
        elif token == "REF":
            fields["is_reference"] = True
        else:
            notes.append(word.group(0))

    if notes_from_prefix.get("spherical"):
        notes.append("spherical")
    if notes_from_prefix.get("square"):
        notes.append("square")
    if notes:
        fields["notes"] = ", ".join(notes)
    return ParsedCallout(kind="dimension", fields=fields)


# --- threads and surface finish ---------------------------------------------------------

_METRIC_THREAD = re.compile(
    r"M(\d+(?:\.\d+)?)\s*(?:X\s*(\d+(?:\.\d+)?))?\s*(?:-\s*(\d[A-Za-z]{1,2}))?"
    r"(?:\s*-\s*(LH))?$"
)
_UNIFIED_THREAD = re.compile(
    r"(?:(\d+)\s*-\s*)?(?:(\d+)/(\d+)|(\d+(?:\.\d+)?))\s*-\s*(\d+)\s*"
    r"(UNC|UNF|UNEF|UN|NPT|NPTF)(?:\s*-\s*(\d[AB]))?$"
)


def _parse_thread(text: str) -> ParsedCallout | None:
    metric = _METRIC_THREAD.match(text)
    if metric:
        fields: dict[str, Any] = {
            "dim_type": "thread",
            "nominal": float(metric.group(1)),
        }
        notes = []
        if metric.group(2):
            notes.append(f"pitch {metric.group(2)}")
        if metric.group(3):
            notes.append(f"class {metric.group(3)}")
        if metric.group(4):
            notes.append("left hand")
        if notes:
            fields["notes"] = ", ".join(notes)
        return ParsedCallout(kind="thread", fields=fields)

    unified = _UNIFIED_THREAD.match(text)
    if unified:
        if unified.group(2):
            diameter = int(unified.group(2)) / int(unified.group(3))
            if unified.group(1):
                diameter += int(unified.group(1))
        else:
            diameter = float(unified.group(4))
        notes = [f"{unified.group(5)} tpi", unified.group(6)]
        if unified.group(7):
            notes.append(f"class {unified.group(7)}")
        return ParsedCallout(
            kind="thread",
            fields={
                "dim_type": "thread",
                "nominal": diameter,
                "notes": ", ".join(notes),
            },
        )
    return None


_SURFACE = re.compile(
    r"(?:Ra|RA|ra|Rz|RZ)\s*(\d+(?:[.,]\d+)?)(?:\s*(?:µm|um|μm))?$"
)
_SURFACE_MICROINCH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:µin|uin|microinch)$")


def _parse_surface_finish(text: str) -> ParsedCallout | None:
    match = _SURFACE.match(text)
    if match:
        return ParsedCallout(
            kind="surface_finish",
            fields={"notes": f"{text[:2].capitalize()} {match.group(1).replace(',', '.')}"},
        )
    match = _SURFACE_MICROINCH.match(text)
    if match:
        return ParsedCallout(
            kind="surface_finish", fields={"notes": f"{match.group(1)} microinch"}
        )
    return None


# --- the entry point ---------------------------------------------------------------------


def parse_callout(text: str, *, kind_hint: str | None = None) -> ParsedCallout:
    """Parse one callout string.

    ``kind_hint`` lets a caller that already knows what it is looking at -- a detector that
    localised a feature control frame, say -- skip the dispatch. It narrows the grammar
    rather than overriding it: a hint of ``geometric_tolerance`` on a string that is plainly
    a dimension still raises, because a confidently wrong parse is worse than a refusal.
    """
    original = text
    normalised = normalise(text)
    if not normalised:
        raise ParseError("empty callout", original, 0)

    reference = normalised.startswith("(") and normalised.endswith(")")
    if reference:
        normalised = normalised[1:-1].strip()

    # A leading count belongs to whatever follows it -- "4X ⌀12" is a diameter and "4X M6"
    # is a thread -- so it is taken off before the dispatch rather than inside one branch.
    count: int | None = None
    lead = re.match(r"(\d+)\s*X\s*(?=[^\d])", normalised)
    if lead:
        count = int(lead.group(1))
        normalised = normalised[lead.end() :].strip()
        if not normalised:
            raise ParseError("a count with nothing counted", original, 0)

    symbol = _leading_gtol(normalised)
    if symbol is not None and kind_hint in (None, "geometric_tolerance"):
        cursor = _Cursor(normalised)
        cursor.pos = symbol[1]
        parsed = _parse_frame(cursor, symbol[0])
        parsed.raw_text = original
        return parsed

    if kind_hint in (None, "thread"):
        thread = _parse_thread(normalised)
        if thread is not None:
            thread.raw_text = original
            if count is not None:
                thread.fields["count"] = count
            return thread

    if kind_hint in (None, "surface_finish"):
        finish = _parse_surface_finish(normalised)
        if finish is not None:
            finish.raw_text = original
            return finish

    if kind_hint in (None, "dimension"):
        cursor = _Cursor(normalised)
        parsed = _parse_dimension(cursor)
        parsed.raw_text = original
        if reference:
            parsed.fields["is_reference"] = True
            parsed.fields.setdefault("tol_style", "general")
        if count is not None:
            parsed.fields["count"] = count
        return parsed

    raise ParseError("not a callout this grammar recognises", original, 0)


def _leading_gtol(text: str) -> tuple[str, int] | None:
    """The geometric symbol at the front, and where it ends.

    Longest match wins, so ``TOTAL RUNOUT`` is never read as the word ``TOTAL`` followed by
    a runout symbol, and a bare ``|`` frame separator before it is skipped.
    """
    stripped = text.lstrip("| ")
    offset = len(text) - len(stripped)
    for name in sorted(_GTOL_NAMES, key=len, reverse=True):
        if stripped.startswith(name):
            after = stripped[len(name) :]
            # A symbol must be followed by a separator, not by more letters: "POSITIONAL"
            # is a word in a note, not a positional tolerance.
            if after[:1].isalpha():
                continue
            return _GTOL_NAMES[name], offset + len(name)
    return None


def parse_many(texts: list[str]) -> tuple[list[ParsedCallout], list[ParseError]]:
    """Parse a batch, keeping the failures rather than dropping them.

    The failures are the interesting half for ``vector_hybrid``: they are exactly the
    strings that get handed to a model, and their count is the number that says how much of
    a drawing the parser could do alone.
    """
    parsed: list[ParsedCallout] = []
    failed: list[ParseError] = []
    for text in texts:
        try:
            parsed.append(parse_callout(text))
        except ParseError as exc:
            failed.append(exc)
    return parsed, failed
