"""Preferred numbers and standard fastener tables."""

from __future__ import annotations

import math

import numpy as np
import pytest

from balloonbench.partgen.preferred import (
    _RENARD_DECADES,  # noqa: PLC2701
    BOLT_CIRCLE_COUNTS,
    CLEARANCE_HOLES,
    METRIC_THREADS,
    PLATE_THICKNESSES,
    R10,
    renard,
    sample_renard,
    snap_to_series,
)


@pytest.mark.parametrize("series", sorted(_RENARD_DECADES))
def test_published_renard_values_track_the_geometric_progression(series: int):
    """Same argument as the ISO 286 cross-check in test_fits: the published values are a
    rounded geometric progression, so they cannot be tested for equality against it, but
    every entry must sit within the rounding margin. A mistyped entry does not."""
    decade = _RENARD_DECADES[series]
    assert len(decade) == series
    for k, value in enumerate(decade):
        ideal = 10 ** (k / series)
        assert abs(value - ideal) / ideal < 0.03, f"R{series}[{k}] = {value} vs {ideal:.4f}"


@pytest.mark.parametrize("series", sorted(_RENARD_DECADES))
def test_renard_decades_are_strictly_increasing(series: int):
    decade = _RENARD_DECADES[series]
    assert list(decade) == sorted(decade)
    assert len(set(decade)) == len(decade)


def test_r10_reproduces_the_familiar_sequence():
    assert renard(10, 10, 100) == (
        10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
    )


def test_finer_series_contain_coarser_ones():
    # R20 refines R10, which refines R5. A value a designer could pick from the coarse
    # series must still be available in the fine one.
    assert set(renard(10, 1, 100)) <= set(renard(20, 1, 100))
    assert set(renard(5, 1, 100)) <= set(renard(10, 1, 100))


def test_renard_spans_multiple_decades():
    values = renard(10, 5.0, 500.0)
    assert min(values) >= 5.0
    assert max(values) <= 500.0
    assert 10.0 in values and 100.0 in values


@pytest.mark.parametrize("bad", [(7, 1, 10), (10, 0, 10), (10, -1, 5), (10, 10, 1)])
def test_renard_rejects_bad_arguments(bad: tuple[int, float, float]):
    with pytest.raises(ValueError):
        renard(*bad)


def test_snap_to_series_picks_the_nearest_value():
    assert snap_to_series(43.0, R10) == 40.0
    assert snap_to_series(47.0, R10) == 50.0
    assert snap_to_series(0.001, R10) == R10[0]
    assert snap_to_series(1e9, R10) == R10[-1]


def test_snap_to_series_breaks_ties_downward():
    """45 sits exactly between the R10 values 40 and 50. Which way it goes matters less
    than that it always goes the same way: the sampler snaps repeatedly, and a tie
    resolved by floating-point chance would make a seed non-reproducible."""
    assert snap_to_series(45.0, R10) == 40.0
    assert snap_to_series(45.0, R10) == snap_to_series(45.0, R10)


def test_snap_to_series_is_idempotent():
    for value in R10:
        assert snap_to_series(value, R10) == value


def test_sample_renard_only_returns_series_values_in_range():
    rng = np.random.default_rng(0)
    allowed = set(renard(10, 20.0, 200.0))
    for _ in range(200):
        assert sample_renard(rng, 20.0, 200.0, series=10) in allowed


def test_sample_renard_is_reproducible_from_a_seed():
    a = [sample_renard(np.random.default_rng(4), 10, 100) for _ in range(3)]
    assert len(set(a)) == 1  # same seed, same draw


def test_sample_renard_rejects_an_empty_window():
    # Nothing in R10 lies strictly between 10.1 and 12.4.
    with pytest.raises(ValueError):
        sample_renard(np.random.default_rng(0), 10.1, 12.4, series=10)


def test_thread_pitches_grow_with_diameter():
    diameters = sorted(METRIC_THREADS)
    pitches = [METRIC_THREADS[d] for d in diameters]
    assert pitches == sorted(pitches)


def test_thread_pitch_is_a_sane_fraction_of_diameter():
    # A coarse metric thread's pitch is roughly a tenth of its diameter; anything far
    # from that is a typo in the table.
    for diameter, pitch in METRIC_THREADS.items():
        assert 0.05 < pitch / diameter < 0.20, f"M{diameter:g} x {pitch}"


def test_clearance_holes_are_larger_than_their_bolt_but_not_by_much():
    for bolt, hole in CLEARANCE_HOLES.items():
        assert hole > bolt, f"M{bolt:g} clearance {hole} does not clear the bolt"
        assert hole - bolt <= 0.15 * bolt + 1.0, f"M{bolt:g} clearance {hole} is loose"


def test_every_clearance_hole_has_a_thread_pitch():
    assert set(CLEARANCE_HOLES) <= set(METRIC_THREADS)


def test_bolt_counts_are_even_and_ascending():
    assert list(BOLT_CIRCLE_COUNTS) == sorted(BOLT_CIRCLE_COUNTS)
    assert all(c % 2 == 0 for c in BOLT_CIRCLE_COUNTS)


def test_plate_thicknesses_are_ascending_and_distinct():
    assert list(PLATE_THICKNESSES) == sorted(set(PLATE_THICKNESSES))


def test_log_uniform_sampling_spreads_across_decades():
    """A supplier's part mix spans orders of magnitude, so sampling must not pile up at
    the top of the range the way a linear-uniform draw would."""
    rng = np.random.default_rng(1)
    draws = [sample_renard(rng, 10.0, 1000.0) for _ in range(600)]
    below_100 = sum(1 for d in draws if d < 100)
    assert 0.35 < below_100 / len(draws) < 0.65, (
        f"{below_100}/{len(draws)} draws below 100 mm; the draw is not log-uniform"
    )
    assert math.isclose(min(draws), 10.0) or min(draws) < 20.0
