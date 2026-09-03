"""Preferred numbers and stock sizes.

SPEC.md section 6.3: do not sample uniformly at random, because that produces
unmanufacturable nonsense that will make the benchmark look synthetic to any engineer who
opens it. Real parts are dimensioned from a small vocabulary of numbers, and the reason is
economic rather than aesthetic -- tooling, stock, fasteners and gauges all exist only at
those sizes.

The Renard series (R5, R10, R20, R40) are geometric progressions covering a decade in 5,
10, 20 or 40 steps, rounded to convenient values. A designer reaching for "about 40 mm"
picks 40, not 41.3, because 40 is where the reamer, the bar stock and the bearing bore
already are. Sampling from these series is the cheapest single thing that makes a
generated drawing look real.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

__all__ = [
    "BOLT_CIRCLE_COUNTS",
    "CLEARANCE_HOLES",
    "METRIC_THREADS",
    "PLATE_THICKNESSES",
    "R10",
    "R20",
    "R40",
    "renard",
    "sample_renard",
    "snap_to_series",
]


# One decade of each Renard series, as published. These are the geometric values rounded
# to convenient numbers, and -- exactly as with the ISO 286 tables in fits.py -- the
# rounding is by convention rather than by a rule we can reproduce: the R10 step above 6.3
# is published as 8.00 where the geometry gives 7.94. Using computed values instead would
# put 39.8 and 12.6 on drawings, and no engineer writes those. tests/test_preferred.py
# cross-checks every entry against the geometric progression it approximates.
_RENARD_DECADES: dict[int, tuple[float, ...]] = {
    5: (1.00, 1.60, 2.50, 4.00, 6.30),
    10: (1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00),
    20: (1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
         3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00),
    40: (1.00, 1.06, 1.12, 1.18, 1.25, 1.32, 1.40, 1.50, 1.60, 1.70,
         1.80, 1.90, 2.00, 2.12, 2.24, 2.36, 2.50, 2.65, 2.80, 3.00,
         3.15, 3.35, 3.55, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.30,
         5.60, 6.00, 6.30, 6.70, 7.10, 7.50, 8.00, 8.50, 9.00, 9.50),
}


def _renard_decade(steps: int) -> tuple[float, ...]:
    return _RENARD_DECADES[steps]


def renard(series: int, low: float, high: float) -> tuple[float, ...]:
    """Values of the R``series`` progression within ``[low, high]``, inclusive.

    >>> renard(10, 10, 100)
    (10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0)
    """
    if series not in _RENARD_DECADES:
        raise ValueError(f"R{series} is not a Renard series; use 5, 10, 20 or 40")
    if low <= 0 or high < low:
        raise ValueError(f"invalid range [{low}, {high}]")

    decade = _renard_decade(series)
    values: list[float] = []
    exponent = math.floor(math.log10(low))
    while True:
        scale = 10.0**exponent
        batch = [round(v * scale, 6) for v in decade]
        if batch[0] > high:
            break
        values.extend(v for v in batch if low <= v <= high)
        exponent += 1
    # The top of a decade is the bottom of the next, so include the closing value.
    top = round(10.0**exponent, 6)
    if low <= top <= high:
        values.append(top)
    return tuple(sorted(set(values)))


#: The three series used across the part families. R10 is the workhorse for diameters;
#: R20 gives finer choice where a designer would have it; R40 is used for lengths.
R10: tuple[float, ...] = renard(10, 1.0, 1000.0)
R20: tuple[float, ...] = renard(20, 1.0, 1000.0)
R40: tuple[float, ...] = renard(40, 1.0, 1000.0)


def snap_to_series(value: float, series: tuple[float, ...]) -> float:
    """The series value nearest ``value``.

    Used after a geometric constraint forces a dimension off the series -- snap back,
    then re-check the constraint, rather than leaving an arbitrary number on the drawing.
    """
    if not series:
        raise ValueError("empty series")
    idx = bisect.bisect_left(series, value)
    if idx == 0:
        return series[0]
    if idx == len(series):
        return series[-1]
    below, above = series[idx - 1], series[idx]
    return below if value - below <= above - value else above


def sample_renard(
    rng: np.random.Generator,
    low: float,
    high: float,
    series: int = 10,
) -> float:
    """Draw one preferred value from ``[low, high]``.

    Uniform over the series entries, which is log-uniform over size -- the right bias,
    since a supplier's part mix is spread over orders of magnitude rather than over a
    linear size axis.
    """
    values = renard(series, low, high)
    if not values:
        raise ValueError(f"no R{series} value lies in [{low}, {high}]")
    return float(rng.choice(values))


#: Metric coarse threads: nominal diameter -> pitch. The pairing is fixed by ISO 261, so a
#: generated thread callout such as ``M12x1.75`` is only credible if the pitch matches.
METRIC_THREADS: dict[float, float] = {
    3.0: 0.5, 4.0: 0.7, 5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5, 12.0: 1.75,
    14.0: 2.0, 16.0: 2.0, 20.0: 2.5, 24.0: 3.0, 30.0: 3.5, 36.0: 4.0, 42.0: 4.5,
    48.0: 5.0,
}

#: Medium-series clearance hole for each bolt size (ISO 273). A bolt pattern whose holes
#: are not a recognised clearance size reads as wrong immediately.
CLEARANCE_HOLES: dict[float, float] = {
    3.0: 3.4, 4.0: 4.5, 5.0: 5.5, 6.0: 6.6, 8.0: 9.0, 10.0: 11.0, 12.0: 13.5,
    14.0: 15.5, 16.0: 17.5, 20.0: 22.0, 24.0: 26.0, 30.0: 33.0, 36.0: 39.0,
}

#: Bolt counts that appear on real circular patterns. Odd counts other than 3 are
#: vanishingly rare because a flange is normally symmetric about two axes.
BOLT_CIRCLE_COUNTS: tuple[int, ...] = (4, 6, 8, 12, 16)

#: Stocked plate and flat bar thicknesses, mm.
PLATE_THICKNESSES: tuple[float, ...] = (
    3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0,
)
