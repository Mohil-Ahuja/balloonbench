"""Milestone 3 acceptance tests for ``degrade``.

PLAN.md states M3's gate in one line: *bounding boxes survive every geometric transform and
still contain ink*. That is the only property of this module that cannot be checked by
looking at an image. A profile that produces a beautifully convincing photocopy while
leaving its ground truth behind is worse than one that does nothing, because it silently
poisons every number the benchmark later reports -- the JSON still validates, the picture
still looks right, and the boxes point at blank paper.

So the tests here are almost entirely about the correspondence between the two, not about
whether an effect looks like a scanner. Three groups:

* the warps themselves, checked as mathematics -- ``forward`` and ``inverse`` must actually
  be inverses, or the pixels and the boxes are moving by different amounts;
* the sample bookkeeping -- dropped characteristics renumbered, image size kept in step,
  the caller's drawing never mutated;
* the gate, end to end, over every family and every profile.

``ink_in_box`` is reused from ``drawgen`` deliberately. If the ink test were reimplemented
here with a different threshold, the two milestones could disagree about whether the same
box is populated, and the disagreement would be a property of the test suite rather than of
the data.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from balloonbench.degrade import PROFILES, Sample, Transform, apply_transforms, degrade
from balloonbench.degrade.base import SampleDestroyed, _clip, _map_box
from balloonbench.degrade.geometry import Barrel, Fold, Homography, _solve_homography
from balloonbench.drawgen.generate import generate_drawing
from balloonbench.drawgen.render import ink_in_box
from balloonbench.partgen.registry import load_families
from balloonbench.schema import Drawing

FAMILIES = sorted(load_families())
PROFILE_NAMES = sorted(PROFILES)

#: Same resolution ``test_drawgen`` uses, for the same reason: high enough that a one-pixel
#: dimension line still darkens its box, low enough to keep a full sweep bearable.
TEST_DPI = 150.0

#: The gate sweep. Every family times every profile, with a distinct seed per cell.
GATE_SEED = 4100


@pytest.fixture(scope="module")
def flange(tmp_path_factory):
    """One rendered drawing, shared. Generation is the expensive part, degradation is not."""
    out = tmp_path_factory.mktemp("flange")
    return generate_drawing("flange", 7, out, dpi=TEST_DPI)


# --- warps are invertible ------------------------------------------------------------------


def _roundtrip(warp, size) -> float:
    """Largest distance a grid of points moves under ``inverse(forward(p))``."""
    xs, ys = np.meshgrid(
        np.linspace(0, size[0], 9), np.linspace(0, size[1], 9)
    )
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    back = warp.inverse(warp.forward(pts))
    return float(np.abs(back - pts).max())


def test_homography_forward_and_inverse_agree():
    src = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=float)
    dst = src + np.array([[12, -7], [-9, 4], [6, 11], [-3, -5]], dtype=float)
    warp = Homography(_solve_homography(src, dst))
    assert _roundtrip(warp, (800, 600)) < 1e-6


def test_homography_actually_moves_the_corners_where_asked():
    """The solve is the piece everything geometric rests on, so check it directly."""
    src = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=float)
    dst = np.array([[2, 1], [98, -3], [103, 77], [-1, 82]], dtype=float)
    mapped = Homography(_solve_homography(src, dst)).forward(src)
    assert np.abs(mapped - dst).max() < 1e-9


def test_barrel_inverse_converges():
    """The radial inverse is iterative, so its accuracy is a claim that needs a number."""
    warp = Barrel(k=0.06, size=(1200, 900))
    assert _roundtrip(warp, (1200, 900)) < 0.05


def test_fold_is_exactly_invertible():
    warp = Fold(position=400.0, vertical=True, amplitude=5.0, width=40.0, size=(900, 700))
    assert _roundtrip(warp, (900, 700)) < 1e-9


def test_fold_leaves_the_far_side_of_the_sheet_alone():
    """A crease is local. If it moved a callout at the other edge it would be a global warp
    wearing a fold's name, and the boxes there would be moved for no physical reason."""
    warp = Fold(position=100.0, vertical=True, amplitude=6.0, width=30.0, size=(900, 700))
    far = np.array([[800.0, 350.0]])
    assert np.abs(warp.forward(far) - far).max() < 1e-6


def test_barrel_bows_an_edge_so_corners_alone_would_understate_the_box():
    """Why ``_map_box`` samples along the edges instead of mapping four corners.

    Under a radial map the midpoint of an edge leaves the straight line between the mapped
    corners. If that bulge fell outside a corners-only bound, the box would cut through ink
    it is supposed to contain -- so the test asserts the sampled bound is the larger one.
    """
    warp = Barrel(k=-0.09, size=(1000, 1000))
    box = np.array([100.0, 100.0, 900.0, 900.0])
    corners = np.array([[100, 100], [900, 100], [900, 900], [100, 900]], dtype=float)
    mapped = warp.forward(corners)
    corner_bound = (
        mapped[:, 0].min(), mapped[:, 1].min(), mapped[:, 0].max(), mapped[:, 1].max()
    )
    sampled = _map_box(box, warp.forward)
    assert sampled[0] <= corner_bound[0] and sampled[1] <= corner_bound[1]
    assert sampled[2] >= corner_bound[2] and sampled[3] >= corner_bound[3]
    assert (sampled[2] - sampled[0]) > (corner_bound[2] - corner_bound[0]) + 1.0


# --- sample bookkeeping --------------------------------------------------------------------


def test_clip_rejects_a_sliver():
    assert _clip((10.0, 10.0, 10.4, 60.0), 100, 100) is None
    assert _clip((10.0, 10.0, 40.0, 60.0), 100, 100) == (10.0, 10.0, 40.0, 60.0)


def test_load_does_not_mutate_the_callers_drawing(flange):
    before = flange.drawing.model_dump_json()
    sample = Sample.load(flange.paths["png"], flange.drawing)
    sample.drawing.characteristics.pop()
    sample.drawing.drawing_id = "mutated"
    assert flange.drawing.model_dump_json() == before


def test_image_size_follows_the_image(flange):
    """The schema rejects a box outside the image, so a transform that changed the raster's
    size without telling the drawing would produce ground truth that no longer validates."""
    sample = Sample.load(flange.paths["png"], flange.drawing)
    width, height = sample.size
    padded = Image.new("RGB", (width + 40, height + 40), "white")
    padded.paste(sample.image, (0, 0))
    padded_sample = sample.with_image(padded, "pad")
    assert list(padded_sample.drawing.image_size) == [width + 40, height + 40]


def test_dropped_characteristics_are_renumbered_contiguously(flange):
    """R1 wants ids 1..n with no gaps. Dropping a cropped-off callout would leave a hole,
    and the degraded drawing has to keep validating -- it is the same schema."""
    sample = Sample.load(flange.paths["png"], flange.drawing)
    width, height = sample.size

    # Push everything a long way right; only the leftmost callouts stay on the sheet.
    def shift(pts: np.ndarray) -> np.ndarray:
        out = np.asarray(pts, dtype=float).copy()
        out[:, 0] += width * 0.55
        return out

    out = sample.map_boxes(shift, name="shift", size=(width, height))
    assert len(out.drawing.characteristics) < len(flange.drawing.characteristics)
    ids = [c.id for c in out.drawing.characteristics]
    assert ids == list(range(1, len(ids) + 1))
    Drawing.model_validate(out.drawing.model_dump())


def test_a_box_that_loses_most_of_its_area_is_dropped_not_clamped(flange):
    """Clamping would report a region that is not where the callout is. The sampler in
    ``partgen`` rejects-and-resamples for the same reason; here the honest answer is that
    the degraded sheet no longer shows that callout."""
    sample = Sample.load(flange.paths["png"], flange.drawing)
    width, height = sample.size

    def off_sheet(pts: np.ndarray) -> np.ndarray:
        out = np.asarray(pts, dtype=float).copy()
        out[:, 0] += width
        return out

    with pytest.raises(SampleDestroyed, match="nothing left to label"):
        sample.map_boxes(off_sheet, name="off", size=(width, height))


def test_a_transforms_probability_is_honoured(flange):
    """Zero probability must mean never, not seldom -- a profile's optional steps are how
    its population is described, and a step that fires anyway makes the description wrong."""

    def marker(sample: Sample, rng: np.random.Generator) -> Sample:
        return sample.with_image(sample.image, "ran")

    sample = Sample.load(flange.paths["png"], flange.drawing)
    never = Transform(name="never", fn=marker, probability=0.0)
    always = Transform(name="always", fn=marker, probability=1.0)
    rng = np.random.default_rng(0)
    assert apply_transforms(sample, (never,) * 20, rng).applied == ()
    assert apply_transforms(sample, (always,), rng).applied == ("ran",)


# --- profiles ------------------------------------------------------------------------------


def test_unknown_profile_is_refused(flange):
    with pytest.raises(KeyError, match="unknown degradation profile"):
        degrade(flange.paths["png"], flange.drawing, "shredded", 1)


def test_clean_is_a_real_baseline(flange):
    """``clean`` must run the same code path and change nothing, so a metric measured on it
    is a baseline for the degraded conditions rather than for the generator's own output."""
    sample = degrade(flange.paths["png"], flange.drawing, "clean", 3)
    assert sample.applied == ()
    assert sample.drawing.provenance.degradation_profile == "clean"
    assert len(sample.drawing.characteristics) == len(flange.drawing.characteristics)
    original = Image.open(flange.paths["png"]).convert("RGB")
    assert np.array_equal(np.asarray(sample.image), np.asarray(original))


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_a_seed_reproduces_a_degraded_sample(flange, profile):
    """CLAUDE.md: all randomness flows from an explicit seed. That has to hold through the
    degradation as well, or a benchmark condition cannot be rebuilt from its manifest."""
    a = degrade(flange.paths["png"], flange.drawing, profile, 99)
    b = degrade(flange.paths["png"], flange.drawing, profile, 99)
    assert a.applied == b.applied
    assert a.drawing.model_dump_json() == b.drawing.model_dump_json()
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_different_seeds_give_different_samples(flange, profile):
    if profile == "clean":
        pytest.skip("the control condition is deliberately deterministic")
    a = degrade(flange.paths["png"], flange.drawing, profile, 1)
    b = degrade(flange.paths["png"], flange.drawing, profile, 2)
    assert not np.array_equal(np.asarray(a.image), np.asarray(b.image))


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_the_profile_is_recorded_and_the_drawing_still_validates(flange, profile):
    sample = degrade(flange.paths["png"], flange.drawing, profile, 12)
    assert sample.drawing.provenance.degradation_profile == profile
    assert list(sample.image.size) == list(sample.drawing.image_size)
    Drawing.model_validate(sample.drawing.model_dump())


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_degradation_keeps_most_of_the_ground_truth(flange, profile):
    """Some loss is the point -- a torn corner really does take callouts with it -- but a
    profile that throws away half a sheet is measuring the dropper, not the model."""
    sample = degrade(flange.paths["png"], flange.drawing, profile, 21)
    kept = len(sample.drawing.characteristics)
    assert kept >= 0.7 * len(flange.drawing.characteristics)


def _assert_boxes_have_ink(sample: Sample, png_path) -> None:
    width, height = sample.size
    boxes = [(f"datum {d.label}", d.bbox) for d in sample.drawing.datums]
    boxes += [(f"characteristic {c.id}", c.bbox) for c in sample.drawing.characteristics]
    for label, box in boxes:
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, (
            f"{label} box {box} outside {width}x{height}"
        )
        assert ink_in_box(png_path, box), f"{label} box {box} contains no ink"


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_milestone_3_gate_on_one_family(flange, profile, tmp_path):
    """The gate itself, on the family with the most callouts. The sweep over every family
    is the slow test below; this one runs by default so the property is never unguarded."""
    sample = degrade(flange.paths["png"], flange.drawing, profile, 55)
    path = tmp_path / f"{profile}.png"
    sample.image.save(path)
    _assert_boxes_have_ink(sample, path)


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_milestone_3_gate(family, tmp_path):
    """PLAN.md's M3 gate: boxes survive every geometric transform and still contain ink.

    Run over every family and every profile, because the geometry that breaks a box is a
    property of where the callouts are -- a shaft's stacked leaders and a plate's edge
    dimensions fail differently under the same warp.
    """
    bundle = generate_drawing(family, GATE_SEED, tmp_path, dpi=TEST_DPI)
    for index, profile in enumerate(PROFILE_NAMES):
        sample = degrade(bundle.paths["png"], bundle.drawing, profile, GATE_SEED + index)
        path = tmp_path / f"{family}_{profile}.png"
        sample.image.save(path)
        _assert_boxes_have_ink(sample, path)
        Drawing.model_validate(sample.drawing.model_dump())
        path.unlink(missing_ok=True)
