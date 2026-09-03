"""ISO 286 limits and fits, tabulated and cross-checked.

SPEC.md section 6.3 is explicit that a fit callout's tolerance band must come from an
IT-grade table rather than a guess, because a plausible-looking but wrong band is exactly
the kind of detail that makes a benchmark look synthetic to an engineer who opens it.

Two things are worth understanding before reading the code.

**Grade and letter are independent.** A fit code such as ``H7`` or ``g6`` is a letter plus
a number. The number selects an *IT grade*, which fixes the width of the tolerance zone.
The letter selects a *fundamental deviation*, which fixes where that zone sits relative to
the basic size. ``H`` is a hole whose lower deviation is zero; ``h`` is a shaft whose upper
deviation is zero. Uppercase is an internal feature (a hole), lowercase an external one (a
shaft) -- the only place in this codebase where letter case carries meaning.

**Why tables and not just formulas.** ISO derives IT grades from a standard tolerance
factor ``i`` that grows as the cube root of size, and derives most fundamental deviations
from similar power laws. Those formulas are implemented here, in
:func:`standard_tolerance_factor` and the ``_formula_*`` helpers. But the values ISO
actually publishes are the computed values *rounded*, and the rounding is not a rule we
can reproduce -- it rounds to convenient numbers, differently at different sizes (at
400-500 mm, IT11 computes to 389 and is published as 400). Reverse-engineering that
rounding would be a rabbit hole for no benefit. So the published values are the authority,
and the formulas serve as an independent cross-check: ``tests/test_fits.py`` asserts every
tabulated value agrees with its formula to within the rounding margin, which catches a
transposed digit in the tables without pretending the formula is exact.

Scope: sizes up to 500 mm; grades IT1 to IT16; letters d, e, f, g, h, js, k, m, n and p in
both cases. The interference letters (r through z) and the very loose letters (a, b, c)
are omitted because ISO subdivides the size ranges for them above 50 mm and 120 mm
respectively, and none of them appear on the parts BalloonBench generates. Asking for one
raises :class:`FitError` rather than returning a quietly wrong band.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "COMMON_HOLE_FITS",
    "COMMON_SHAFT_FITS",
    "SUPPORTED_LETTERS",
    "FitError",
    "Limits",
    "fit_limits",
    "it_grade",
    "parse_fit",
    "size_range",
    "standard_tolerance_factor",
]

MICRON = 1e-3  # one micrometre, in millimetres


class FitError(ValueError):
    """A fit designation ISO 286 does not define, or that is outside our scope."""


# --- size ranges --------------------------------------------------------------------

#: Upper bounds of the ISO 286 principal size ranges, in mm. The first range is
#: 0 < D <= 3; each later range runs from the previous bound up to this one.
_RANGE_BOUNDS: tuple[float, ...] = (
    3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400, 500,
)
_N_RANGES = len(_RANGE_BOUNDS)


def _range_index(diameter_mm: float) -> int:
    if diameter_mm <= 0:
        raise FitError(f"diameter must be positive, got {diameter_mm}")
    for idx, high in enumerate(_RANGE_BOUNDS):
        if diameter_mm <= high:
            return idx
    raise FitError(
        f"diameter {diameter_mm} mm exceeds the {_RANGE_BOUNDS[-1]} mm limit of this "
        f"implementation"
    )


def size_range(diameter_mm: float) -> tuple[float, float]:
    """The ISO 286 size range containing ``diameter_mm``, as ``(low, high)``.

    Ranges are open at the bottom and closed at the top, so 30 mm exactly belongs to the
    18-30 range, not to 30-50. That boundary matters more than it looks: parts built from
    preferred number series land on range boundaries far more often than arbitrary sizes
    do, so an off-by-one here would mis-tolerance a large fraction of generated features.
    """
    idx = _range_index(diameter_mm)
    return (0.0 if idx == 0 else _RANGE_BOUNDS[idx - 1], _RANGE_BOUNDS[idx])


def _geometric_mean_size(diameter_mm: float) -> float:
    """The nominal size ISO uses for a range: the geometric mean of its bounds.

    The first range is the exception -- its lower bound is zero, which would give a mean
    of zero -- so ISO uses sqrt(1 x 3) for it.
    """
    low, high = size_range(diameter_mm)
    return math.sqrt(max(low, 1.0) * high)


def standard_tolerance_factor(diameter_mm: float) -> float:
    """The factor ``i``, in micrometres, that IT5 and coarser are multiples of.

    The cube-root term reflects that process and measurement error scale sublinearly with
    part size; the small linear term covers thermal and elastic effects.
    """
    d = _geometric_mean_size(diameter_mm)
    return 0.45 * d ** (1.0 / 3.0) + 0.001 * d


#: Multiples of ``i`` defining each IT grade from IT5 up. Each grade is about 1.6x its
#: predecessor, so the series repeats by a factor of 10 every five grades.
_IT_MULTIPLES: dict[int, float] = {
    5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100, 12: 160,
    13: 250, 14: 400, 15: 640, 16: 1000,
}


# --- IT grade tables ----------------------------------------------------------------
#
# Published IT values in micrometres, one entry per size range in _RANGE_BOUNDS order.
# IT5 to IT11 are tabulated; IT12 and coarser are derived, because from IT12 up the series
# repeats exactly by a factor of ten every five grades. That identity is a property of the
# standard rather than a shortcut, and tests/test_fits.py checks the derived grades
# against published reference rows as well as against the formula.

_IT_TABLE_UM: dict[int, tuple[int, ...]] = {
    5:  (4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27),
    6:  (6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40),
    7:  (10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63),
    8:  (14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97),
    9:  (25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155),
    10: (40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250),
    # IT11 is tabulated rather than derived: the ten-fold identity below holds for every
    # grade from IT12 up, but IT11 at 3-6 mm is published as 75, not the 80 that 10 x IT6
    # would give.
    11: (60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400),
}

_MIN_TABULATED_GRADE = 5
_MAX_GRADE = 16


def it_grade(diameter_mm: float, grade: int) -> float:
    """Width of the IT``grade`` tolerance zone at ``diameter_mm``, in **millimetres**.

    >>> round(it_grade(44.0, 7), 4)   # the familiar 25 micron band of a 44 H7 bore
    0.025
    >>> round(it_grade(44.0, 11), 4)
    0.16
    """
    if not 1 <= grade <= _MAX_GRADE:
        raise FitError(f"IT{grade} is outside the IT1 to IT{_MAX_GRADE} range covered here")

    idx = _range_index(diameter_mm)

    if grade in _IT_TABLE_UM:
        return _IT_TABLE_UM[grade][idx] * MICRON
    if grade > 11:
        # From IT12 up the series repeats exactly by a factor of ten every five grades.
        return it_grade(diameter_mm, grade - 5) * 10

    # IT1 to IT4 are the gauge grades. The cube-root model does not hold there, so ISO
    # defines IT1 directly and interpolates IT2 to IT4 geometrically between IT1 and IT5
    # to keep the progression smooth. These grades never appear on a BalloonBench part;
    # they exist so a verifier check can still resolve one if a real drawing carries it.
    d = _geometric_mean_size(diameter_mm)
    it1 = 0.8 + 0.020 * d
    if grade == 1:
        return round(it1) * MICRON
    it5 = _IT_TABLE_UM[_MIN_TABULATED_GRADE][idx]
    ratio = (it5 / it1) ** 0.25
    return round(it1 * ratio ** (grade - 1)) * MICRON


# --- fundamental deviation tables ---------------------------------------------------
#
# For a shaft, the fundamental deviation is the UPPER deviation (es, always <= 0) for
# letters that place the zone below the basic size, and the LOWER deviation (ei, always
# >= 0) for letters that place it above. A hole's deviation is derived from the shaft's,
# by the two rules in fit_limits.

#: Shaft upper deviations es, micrometres, for the clearance letters. Never positive.
_SHAFT_ES_UM: dict[str, tuple[int, ...]] = {
    "d": (-20, -30, -40, -50, -65, -80, -100, -120, -145, -170, -190, -210, -230),
    "e": (-14, -20, -25, -32, -40, -50, -60, -72, -85, -100, -110, -125, -135),
    "f": (-6, -10, -13, -16, -20, -25, -30, -36, -43, -50, -56, -62, -68),
    "g": (-2, -4, -5, -6, -7, -9, -10, -12, -14, -15, -17, -18, -20),
    "h": (0,) * _N_RANGES,
}

#: Shaft lower deviations ei, micrometres, for the transition and interference letters.
#: Never negative. ``k`` is the odd one: it is a genuine transition fit only over grades
#: IT4 to IT7, and its deviation collapses to zero outside that window.
_SHAFT_EI_UM: dict[str, tuple[int, ...]] = {
    "k": (0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5),
    "m": (2, 4, 6, 7, 8, 9, 11, 13, 15, 17, 20, 21, 23),
    "n": (4, 8, 10, 12, 15, 17, 20, 23, 27, 31, 34, 37, 40),
    "p": (6, 12, 15, 18, 22, 26, 32, 37, 43, 50, 56, 62, 68),
}

#: Letters this module resolves, in shaft (lowercase) form. ``js`` is handled separately
#: because it is symmetric about the basic size and so has no fundamental deviation.
SUPPORTED_LETTERS: frozenset[str] = (
    frozenset(_SHAFT_ES_UM) | frozenset(_SHAFT_EI_UM) | {"js"}
)

#: Holes K, M and N take the delta correction up to IT8; P and beyond only up to IT7.
_DELTA_MAX_GRADE: dict[str, int] = {"k": 8, "m": 8, "n": 8, "p": 7}


# --- formula counterparts, used only as a cross-check in the tests -------------------


def _formula_it_um(diameter_mm: float, grade: int) -> float:
    return _IT_MULTIPLES[grade] * standard_tolerance_factor(diameter_mm)


def _formula_shaft_es_um(letter: str, diameter_mm: float) -> float:
    d = _geometric_mean_size(diameter_mm)
    return {
        "d": -16 * d**0.44,
        "e": -11 * d**0.41,
        "f": -5.5 * d**0.41,
        "g": -2.5 * d**0.34,
        "h": 0.0,
    }[letter]


def _formula_shaft_ei_um(letter: str, diameter_mm: float) -> float:
    d = _geometric_mean_size(diameter_mm)
    if letter == "k":
        return 0.6 * d ** (1.0 / 3.0)
    if letter == "m":
        return (it_grade(diameter_mm, 7) - it_grade(diameter_mm, 6)) / MICRON
    if letter == "n":
        return 5 * d**0.34
    # ISO gives p as IT7 plus a small tabulated increment with no closed form, so there
    # is nothing to cross-check it against beyond the bound asserted in the tests: an
    # interference fit's deviation must be at least the IT7 width.
    raise KeyError(letter)


# --- public API ---------------------------------------------------------------------

FeatureKind = Literal["hole", "shaft"]


@dataclass(frozen=True)
class Limits:
    """A resolved tolerance band, in millimetres, as signed deviations from the basic size.

    Deviations are stored exactly as the ground-truth schema stores them (SPEC.md section
    4, rule R4): ``upper`` and ``lower`` are offsets from ``nominal``, never absolute
    limits. Whether the drawing shows ``+0.025/0``, ``44.025/44.000`` or ``Ø44 H7`` is the
    renderer's decision, made per house style.
    """

    nominal: float
    upper: float
    lower: float
    designation: str
    grade: int
    kind: FeatureKind

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def max_material(self) -> float:
        """Size at which the feature holds most material: smallest hole, largest shaft.

        The verifier's MMC checks are written against this, so it lives here rather than
        being recomputed with the sign convention guessed at each call site.
        """
        return self.nominal + (self.lower if self.kind == "hole" else self.upper)

    @property
    def least_material(self) -> float:
        return self.nominal + (self.upper if self.kind == "hole" else self.lower)

    def as_limits(self) -> tuple[float, float]:
        """Absolute ``(min, max)`` size, for rendering a tolerance in limit form."""
        return (self.nominal + self.lower, self.nominal + self.upper)

    def __str__(self) -> str:
        return f"{self.nominal:g} {self.designation} ({self.upper:+.3f}/{self.lower:+.3f})"


_FIT_RE = re.compile(r"^([A-Za-z]{1,2})\s*(\d{1,2})$")


def parse_fit(designation: str) -> tuple[str, int, FeatureKind]:
    """Split ``"H7"`` into ``("H", 7, "hole")`` and ``"g6"`` into ``("g", 6, "shaft")``."""
    m = _FIT_RE.match(designation.strip())
    if not m:
        raise FitError(f"{designation!r} is not an ISO 286 fit designation")
    letter, grade_s = m.group(1), m.group(2)
    if letter.isupper():
        kind: FeatureKind = "hole"
    elif letter.islower():
        kind = "shaft"
    else:
        raise FitError(
            f"{designation!r} mixes case; a fit letter is all uppercase (a hole) or all "
            f"lowercase (a shaft)"
        )
    if letter.lower() not in SUPPORTED_LETTERS:
        raise FitError(
            f"fit letter {letter!r} is outside the scope of this implementation; "
            f"supported letters are {sorted(SUPPORTED_LETTERS)}"
        )
    return letter, int(grade_s), kind


def fit_limits(nominal_mm: float, designation: str) -> Limits:
    """Resolve a fit callout such as ``"H7"`` or ``"g6"`` at a given basic size.

    >>> lim = fit_limits(44.0, "H7")
    >>> round(lim.upper, 4), round(lim.lower, 4)
    (0.025, 0.0)
    >>> lim = fit_limits(44.0, "g6")
    >>> round(lim.upper, 4), round(lim.lower, 4)
    (-0.009, -0.025)
    >>> fit_limits(44.0, "P7").upper                      # a press fit is all negative
    -0.017
    """
    letter, grade, kind = parse_fit(designation)
    it = it_grade(nominal_mm, grade)
    idx = _range_index(nominal_mm)
    key = letter.lower()

    if key == "js":
        # Symmetric about the basic size, so there is no fundamental deviation to look up.
        half = it / 2
        return Limits(nominal_mm, half, -half, designation, grade, kind)

    if key in _SHAFT_ES_UM:
        es = _SHAFT_ES_UM[key][idx] * MICRON
        if kind == "shaft":
            return Limits(nominal_mm, es, es - it, designation, grade, kind)
        # A hole of the same letter mirrors the shaft's zone about the basic size. That
        # mirror is what makes a hole-basis and a shaft-basis fit of the same letters give
        # the same clearance. The added zero normalises -0.0 to 0.0, which matters because
        # these deviations are serialised straight into the ground-truth JSON.
        ei = -es + 0.0
        return Limits(nominal_mm, ei + it, ei, designation, grade, kind)

    ei_shaft = _SHAFT_EI_UM[key][idx] * MICRON
    if key == "k" and not 4 <= grade <= 7:
        ei_shaft = 0.0

    if kind == "shaft":
        return Limits(nominal_mm, ei_shaft + it, ei_shaft, designation, grade, kind)

    # For a hole in the transition and interference letters, a plain mirror would make the
    # fit's character drift with grade. ISO corrects that with delta -- the step between
    # this grade and the next finer one -- applied only over the grades where the letter
    # is meant to be used, and never at or below 3 mm, where the correction is defined as
    # zero and the hole zone is the exact mirror of the shaft zone.
    delta = (
        it - it_grade(nominal_mm, grade - 1)
        if idx > 0 and 2 <= grade <= _DELTA_MAX_GRADE[key]
        else 0.0
    )
    es = -ei_shaft + delta
    return Limits(nominal_mm, es, es - it, designation, grade, kind)


#: The fits a sampler should reach for. Everything here is legal ISO; the point of the
#: shortlist is realism -- these are what appears on supplier drawings for the part
#: families BalloonBench generates, and sampling uniformly over all legal fits would
#: produce combinations no process engineer would specify.
COMMON_HOLE_FITS: tuple[str, ...] = (
    "H6", "H7", "H8", "H9", "H11", "G7", "F8", "K7", "N7", "P7", "JS7",
)
COMMON_SHAFT_FITS: tuple[str, ...] = (
    "h6", "h7", "h9", "g6", "f7", "e8", "k6", "m6", "n6", "p6", "js6",
)
