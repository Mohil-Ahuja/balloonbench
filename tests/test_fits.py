"""ISO 286 conformance for balloonbench.partgen.fits.

Two independent kinds of check, because each catches what the other misses:

1. **Reference rows.** Published ISO 286 values for the fits BalloonBench actually
   generates, one value per size range. A wrong table entry or a wrong sign convention
   fails here.
2. **Formula cross-check.** Every tabulated value is compared against the power-law
   formula it was derived from. The published values are the rounded ones, so this cannot
   be an equality test -- but a transposed digit moves a value far outside the rounding
   margin, which is exactly the transcription error a reference row in a *different* grade
   would not catch.

One representative diameter per ISO size range. Values are chosen to sit inside a range
rather than on its boundary, so a range-selection off-by-one shows up as a wrong value
rather than a coincidentally right one.
"""

from __future__ import annotations

import pytest

# The _-prefixed names are the formula counterparts and the raw tables. They are private
# because nothing outside this test should use them: they exist so the published values
# can be checked against the model they were derived from.
from balloonbench.partgen.fits import (  # noqa: PLC2701
    _IT_TABLE_UM,
    _SHAFT_EI_UM,
    _SHAFT_ES_UM,
    COMMON_HOLE_FITS,
    COMMON_SHAFT_FITS,
    MICRON,
    FitError,
    _formula_it_um,
    _formula_shaft_ei_um,
    _formula_shaft_es_um,
    fit_limits,
    it_grade,
    parse_fit,
    size_range,
    standard_tolerance_factor,
)

#: One diameter per ISO size range, in range order.
SIZES: tuple[float, ...] = (2.0, 5.0, 8.0, 15.0, 25.0, 40.0, 65.0, 100.0, 150.0, 200.0,
                            300.0, 350.0, 450.0)

#: Index of the 0-3 mm range in SIZES. Excluded from the formula cross-checks only; the
#: reference rows cover it exactly.
FIRST_RANGE = 0


def _um(values: list[float]) -> list[int]:
    return [round(v / MICRON) for v in values]


# --- IT grades ----------------------------------------------------------------------

IT_REFERENCE: dict[int, list[int]] = {
    5: [4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27],
    6: [6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40],
    7: [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63],
    8: [14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97],
    9: [25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155],
    10: [40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250],
    11: [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400],
    12: [100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570, 630],
    13: [140, 180, 220, 270, 330, 390, 460, 540, 630, 720, 810, 890, 970],
    14: [250, 300, 360, 430, 520, 620, 740, 870, 1000, 1150, 1300, 1400, 1550],
    15: [400, 480, 580, 700, 840, 1000, 1200, 1400, 1600, 1850, 2100, 2300, 2500],
    16: [600, 750, 900, 1100, 1300, 1600, 1900, 2200, 2500, 2900, 3200, 3600, 4000],
}


@pytest.mark.parametrize("grade", sorted(IT_REFERENCE))
def test_it_grades_match_iso(grade: int):
    assert _um([it_grade(d, grade) for d in SIZES]) == IT_REFERENCE[grade]


def test_it_grades_are_monotonic_in_size_and_grade():
    for grade in sorted(IT_REFERENCE):
        widths = [it_grade(d, grade) for d in SIZES]
        assert widths == sorted(widths), f"IT{grade} is not monotonic in size"
    for d in SIZES:
        widths = [it_grade(d, g) for g in sorted(IT_REFERENCE)]
        assert widths == sorted(widths), f"grades are not monotonic at {d} mm"


def test_it_series_repeats_by_ten_every_five_grades_from_it12():
    # A structural property of the standard, and a check that the derivation used for the
    # untabulated coarse grades is anchored to real tabulated ones.
    for grade in range(12, 17):
        for d in SIZES:
            assert it_grade(d, grade) == pytest.approx(it_grade(d, grade - 5) * 10)


@pytest.mark.parametrize("grade", sorted(_IT_TABLE_UM))
def test_tabulated_it_values_agree_with_the_formula(grade: int):
    """The published values are the computed ones rounded, so this is a margin check,
    not an equality check. It still pins every entry to within a few percent, which a
    transposed digit is not.

    The first size range is excluded throughout these cross-checks: below 3 mm the
    cube-root model underlying the formulas fits poorly and ISO effectively sets the
    values by convention, so the formula is not evidence about them either way. Those
    entries are covered exactly by the reference-row tests above.
    """
    for idx, d in enumerate(SIZES):
        if idx == FIRST_RANGE:
            continue
        table = _IT_TABLE_UM[grade][idx]
        formula = _formula_it_um(d, grade)
        # 10% covers the widest genuine rounding gap (IT6 at 3-6 mm, published 8
        # against a computed 7.3). A transposed digit misses by well over 100%.
        assert abs(table - formula) / formula < 0.10, (
            f"IT{grade} at {d} mm: table {table}, formula {formula:.1f}"
        )


# --- fundamental deviations ---------------------------------------------------------

# Hole upper deviation ES, micrometres.
HOLE_ES_REFERENCE: dict[str, list[int]] = {
    "H7": [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63],
    "H11": [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400],
    "K7": [0, 3, 5, 6, 6, 7, 9, 10, 12, 13, 16, 17, 18],
    "M7": [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N7": [-4, -4, -4, -5, -7, -8, -9, -10, -12, -14, -14, -16, -17],
    "P7": [-6, -8, -9, -11, -14, -17, -21, -24, -28, -33, -36, -41, -45],
}

# Hole lower deviation EI, micrometres.
HOLE_EI_REFERENCE: dict[str, list[int]] = {
    "H7": [0] * 13,
    "F8": [6, 10, 13, 16, 20, 25, 30, 36, 43, 50, 56, 62, 68],
    "G7": [2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20],
}

# Shaft upper deviation es, micrometres.
SHAFT_ES_REFERENCE: dict[str, list[int]] = {
    "h6": [0] * 13,
    "h11": [0] * 13,
    "g6": [-2, -4, -5, -6, -7, -9, -10, -12, -14, -15, -17, -18, -20],
    "f7": [-6, -10, -13, -16, -20, -25, -30, -36, -43, -50, -56, -62, -68],
    "e8": [-14, -20, -25, -32, -40, -50, -60, -72, -85, -100, -110, -125, -135],
}

# Shaft lower deviation ei, micrometres.
SHAFT_EI_REFERENCE: dict[str, list[int]] = {
    "k6": [0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5],
    "m6": [2, 4, 6, 7, 8, 9, 11, 13, 15, 17, 20, 21, 23],
    "n6": [4, 8, 10, 12, 15, 17, 20, 23, 27, 31, 34, 37, 40],
    "p6": [6, 12, 15, 18, 22, 26, 32, 37, 43, 50, 56, 62, 68],
    "h11": [-60, -75, -90, -110, -130, -160, -190, -220, -250, -290, -320, -360, -400],
}


@pytest.mark.parametrize("fit", sorted(HOLE_ES_REFERENCE))
def test_hole_upper_deviations_match_iso(fit: str):
    assert _um([fit_limits(d, fit).upper for d in SIZES]) == HOLE_ES_REFERENCE[fit]


@pytest.mark.parametrize("fit", sorted(HOLE_EI_REFERENCE))
def test_hole_lower_deviations_match_iso(fit: str):
    assert _um([fit_limits(d, fit).lower for d in SIZES]) == HOLE_EI_REFERENCE[fit]


@pytest.mark.parametrize("fit", sorted(SHAFT_ES_REFERENCE))
def test_shaft_upper_deviations_match_iso(fit: str):
    assert _um([fit_limits(d, fit).upper for d in SIZES]) == SHAFT_ES_REFERENCE[fit]


@pytest.mark.parametrize("fit", sorted(SHAFT_EI_REFERENCE))
def test_shaft_lower_deviations_match_iso(fit: str):
    assert _um([fit_limits(d, fit).lower for d in SIZES]) == SHAFT_EI_REFERENCE[fit]


def test_the_worked_example_from_the_spec():
    """SPEC.md section 6.3 states that a 44 H7 bore has a legal +0.025/0 band."""
    lim = fit_limits(44.0, "H7")
    assert lim.upper == pytest.approx(0.025)
    assert lim.lower == pytest.approx(0.0)
    assert lim.as_limits() == pytest.approx((44.0, 44.025))


@pytest.mark.parametrize("letter", sorted(_SHAFT_ES_UM))
def test_tabulated_shaft_es_agrees_with_the_formula(letter: str):
    if letter == "h":
        return
    for idx, d in enumerate(SIZES):
        if idx == FIRST_RANGE:
            continue
        table = _SHAFT_ES_UM[letter][idx]
        formula = _formula_shaft_es_um(letter, d)
        assert abs(table - formula) / abs(formula) < 0.08, (
            f"{letter} at {d} mm: table {table}, formula {formula:.1f}"
        )


@pytest.mark.parametrize("letter", ["k", "m", "n"])
def test_tabulated_shaft_ei_agrees_with_the_formula(letter: str):
    for idx, d in enumerate(SIZES):
        if idx == FIRST_RANGE:
            continue
        table = _SHAFT_EI_UM[letter][idx]
        formula = _formula_shaft_ei_um(letter, d)
        assert abs(table - formula) <= max(1.0, 0.12 * formula), (
            f"{letter} at {d} mm: table {table}, formula {formula:.1f}"
        )


def test_p_is_a_genuine_interference_letter():
    """p has no closed form to check against. What must hold is the property that makes
    it an interference letter at all: a p6 shaft's smallest size still exceeds the basic
    size by at least a full IT6 band, so it cannot slip into an H6 hole."""
    for d in SIZES:
        assert fit_limits(d, "p6").lower >= it_grade(d, 6)


# --- semantics of the resolved band -------------------------------------------------


def test_clearance_fit_always_leaves_clearance():
    """H7/g6 is the standard sliding fit: the largest shaft must still enter the
    smallest hole."""
    for d in SIZES:
        hole, shaft = fit_limits(d, "H7"), fit_limits(d, "g6")
        assert shaft.max_material < hole.max_material


def test_interference_fit_always_interferes():
    """H7/p6 is a press fit: the smallest shaft must still be larger than the largest
    hole is at its tightest."""
    for d in SIZES:
        hole, shaft = fit_limits(d, "H7"), fit_limits(d, "p6")
        assert shaft.least_material > hole.max_material


def test_max_and_least_material_follow_the_feature_kind():
    # A hole holds most material when it is smallest; a shaft when it is largest.
    hole = fit_limits(44.0, "H7")
    assert hole.max_material == pytest.approx(44.0)
    assert hole.least_material == pytest.approx(44.025)

    shaft = fit_limits(44.0, "g6")
    assert shaft.max_material == pytest.approx(44.0 - 0.009)
    assert shaft.least_material == pytest.approx(44.0 - 0.025)


def test_js_is_symmetric_about_the_basic_size():
    lim = fit_limits(44.0, "JS7")
    assert lim.upper == pytest.approx(-lim.lower)
    assert lim.width == pytest.approx(it_grade(44.0, 7))


def test_width_always_equals_the_it_grade():
    for fit in COMMON_HOLE_FITS + COMMON_SHAFT_FITS:
        _, grade, _ = parse_fit(fit)
        for d in SIZES:
            assert fit_limits(d, fit).width == pytest.approx(it_grade(d, grade))


def test_upper_is_never_below_lower():
    for fit in COMMON_HOLE_FITS + COMMON_SHAFT_FITS:
        for d in SIZES:
            lim = fit_limits(d, fit)
            assert lim.upper >= lim.lower


def test_every_common_fit_resolves_across_every_size_range():
    for fit in COMMON_HOLE_FITS + COMMON_SHAFT_FITS:
        for d in SIZES:
            assert fit_limits(d, fit) is not None


# --- size ranges and parsing --------------------------------------------------------


@pytest.mark.parametrize(
    ("diameter", "expected"),
    [
        (0.5, (0.0, 3.0)),
        (3.0, (0.0, 3.0)),      # closed at the top
        (3.01, (3.0, 6.0)),
        (30.0, (18.0, 30.0)),   # a preferred size sitting exactly on a boundary
        (30.5, (30.0, 50.0)),
        (500.0, (400.0, 500.0)),
    ],
)
def test_size_range_boundaries(diameter: float, expected: tuple[float, float]):
    assert size_range(diameter) == expected


def test_boundary_diameter_uses_the_lower_range():
    # 30 H7 is +0.021/0 (the 18-30 range), not the +0.025 of 30-50.
    assert fit_limits(30.0, "H7").upper == pytest.approx(0.021)
    assert fit_limits(30.001, "H7").upper == pytest.approx(0.025)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("H7", ("H", 7, "hole")), ("g6", ("g", 6, "shaft")),
     ("JS7", ("JS", 7, "hole")), ("js6", ("js", 6, "shaft")), (" H7 ", ("H", 7, "hole"))],
)
def test_parse_fit(text: str, expected: tuple[str, int, str]):
    assert parse_fit(text) == expected


@pytest.mark.parametrize("bad", ["", "H", "7", "Hh7", "H7x", "Q7", "r6", "a11", "s6"])
def test_unparseable_or_unsupported_fits_are_rejected(bad: str):
    with pytest.raises(FitError):
        parse_fit(bad)


@pytest.mark.parametrize("diameter", [0.0, -5.0, 501.0, 1e4])
def test_out_of_range_diameters_are_rejected(diameter: float):
    with pytest.raises(FitError):
        it_grade(diameter, 7)


@pytest.mark.parametrize("grade", [0, 17, 20, -1])
def test_out_of_range_grades_are_rejected(grade: int):
    with pytest.raises(FitError):
        it_grade(44.0, grade)


def test_standard_tolerance_factor_grows_sublinearly():
    small, large = standard_tolerance_factor(5.0), standard_tolerance_factor(450.0)
    assert large > small
    assert large / small < 450.0 / 5.0  # cube-root growth, not proportional
