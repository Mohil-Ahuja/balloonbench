"""Milestone 4 acceptance tests for ``evalkit``.

PLAN.md's M4 gate: *metrics reproduce known values on synthetic perturbations*. That phrasing
decides the shape of this file. A metric cannot be tested against a model's output, because
nobody knows what the right answer is; it can be tested against a deliberately damaged copy
of the ground truth, where the right answer is arithmetic. So every test here starts from a
perfect prediction -- the ground truth itself -- applies one known, counted change, and
asserts the exact fraction that must come back.

The perfect prediction is the fixed point, and it is checked first. If evaluating a drawing
against itself does not report flawless, no other number in the harness means anything.

One deliberate omission: there is no test that a good model scores well and a bad one badly.
That would be a test of the perturbation, not of the metric. What is tested is that each
metric moves by exactly the amount the change implies, and that changes it should be blind
to -- a unit conversion, a few pixels of box jitter -- move it not at all.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from balloonbench.drawgen.generate import generate_drawing
from balloonbench.evalkit.matching import (
    MatchConfig,
    bbox_iou,
    match,
    no_bbox_config,
    semantic_distance,
    to_mm,
)
from balloonbench.evalkit.metrics import (
    aggregate,
    datum_graph_distance,
    error_costs,
    evaluate_drawing,
)
from balloonbench.evalkit.prediction import Prediction, parse_prediction
from balloonbench.evalkit.report import markdown_table, write_report
from balloonbench.partgen.registry import load_families
from balloonbench.schema import Characteristic, DatumRef, Drawing

TEST_DPI = 150.0

#: Families register on import of the registry loader, which the generator relies on.
load_families()


@pytest.fixture(scope="module")
def gt(tmp_path_factory) -> Drawing:
    """A generated flange: ten characteristics, two datums, several tolerances with frames.

    Real generator output rather than a hand-written fixture, so the metrics are exercised
    against the same shape of data the benchmark will actually score.
    """
    out = tmp_path_factory.mktemp("eval")
    return generate_drawing("flange", 7, out, dpi=TEST_DPI, write_artifacts=False).drawing


@pytest.fixture
def perfect(gt) -> Prediction:
    return Prediction.from_ground_truth(gt)


def _dimensions(prediction: Prediction) -> list[Characteristic]:
    return [c for c in prediction.characteristics if c.kind == "dimension"]


def _gtols(prediction: Prediction) -> list[Characteristic]:
    return [c for c in prediction.characteristics if c.kind == "geometric_tolerance"]


# --- the fixed point ----------------------------------------------------------------------


def test_the_ground_truth_scores_perfectly_against_itself(gt, perfect):
    score = evaluate_drawing(gt, perfect)
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)
    assert score.false_positives == score.false_negatives == 0
    assert score.exact_drawing is True
    assert score.graph_distance == 0.0
    assert score.cost == 0.0
    for name, rate in score.tier2.items():
        assert rate.value in (1.0, None), f"{name} is not perfect on identical input"


def test_the_drawing_has_enough_content_to_test_against(gt):
    """Guards the fixture, not the code. If the generator ever produced a flange without
    geometric tolerances, half the tests below would silently pass by vacuity."""
    assert len(gt.characteristics) >= 6
    assert len(_gtols(Prediction.from_ground_truth(gt))) >= 2
    assert any(c.datum_refs for c in gt.characteristics)


# --- tier 1: detection --------------------------------------------------------------------


@pytest.mark.parametrize("dropped", [1, 2, 3])
def test_dropping_callouts_moves_recall_by_exactly_that_much(gt, perfect, dropped):
    n = len(gt.characteristics)
    perfect.characteristics = perfect.characteristics[dropped:]
    score = evaluate_drawing(gt, perfect)
    assert score.false_negatives == dropped
    assert score.recall == (n - dropped) / n
    assert score.precision == 1.0, "dropping ground truth cannot create a false positive"


def test_inventing_callouts_moves_precision_by_exactly_that_much(gt, perfect):
    n = len(gt.characteristics)
    invented = 3
    width, height = gt.image_size
    for i in range(invented):
        perfect.characteristics.append(
            Characteristic(
                id=1000 + i,
                kind="dimension",
                view="front",
                # Far from anything real, so it cannot be matched to a true callout.
                bbox=[width - 60.0 - i, height - 40.0, width - 20.0 - i, height - 10.0],
                dim_type="linear",
                nominal=999.0 + i,
            )
        )
    score = evaluate_drawing(gt, perfect)
    assert score.false_positives == invented
    assert score.recall == 1.0
    assert score.precision == n / (n + invented)


def test_a_malformed_prediction_is_a_false_positive_and_is_reported_as_one(gt):
    """A flatness with a datum reference violates R2 and cannot be loaded. It must still be
    counted -- a model cannot improve its precision by emitting output that will not parse."""
    payload = {
        "drawing_id": gt.drawing_id,
        "units": gt.units,
        "characteristics": [
            {
                "id": 1,
                "kind": "geometric_tolerance",
                "view": "front",
                "bbox": [10.0, 10.0, 60.0, 30.0],
                "gtol_symbol": "flatness",
                "gtol_value": 0.02,
                "datum_refs": [{"label": "A", "modifier": None}],
            }
        ],
    }
    prediction = parse_prediction(payload)
    assert len(prediction.malformed) == 1
    assert "form tolerance" in prediction.malformed[0].reason

    score = evaluate_drawing(gt, prediction)
    assert score.malformed == 1
    assert score.false_positives == 1
    assert score.n_pred == 1


def test_recall_breaks_out_by_kind_and_symbol(gt, perfect):
    target = _gtols(perfect)[0]
    perfect.characteristics = [c for c in perfect.characteristics if c is not target]
    score = evaluate_drawing(gt, perfect)

    symbol = score.by_symbol[target.gtol_symbol]
    assert symbol.hits == symbol.total - 1
    kinds = score.by_kind["geometric_tolerance"]
    assert kinds.hits == kinds.total - 1
    assert score.by_kind["dimension"].value == 1.0


# --- matching behaviour -------------------------------------------------------------------


def test_small_box_jitter_does_not_break_matching(gt, perfect):
    """Boxes move under degradation and under a detector's own imprecision. A few pixels
    must not turn a correct extraction into a miss plus a false alarm."""
    for c in perfect.characteristics:
        c.bbox = [c.bbox[0] + 2, c.bbox[1] - 2, c.bbox[2] + 2, c.bbox[3] - 2]
    score = evaluate_drawing(gt, perfect)
    assert score.recall == 1.0 and score.precision == 1.0


def test_boxes_moved_far_away_stop_matching(gt, perfect):
    for c in perfect.characteristics:
        c.bbox = [10.0, 10.0, 40.0, 25.0]
    score = evaluate_drawing(gt, perfect)
    assert score.true_positives <= 1, "disjoint boxes must not match in bbox mode"


def test_no_bbox_mode_recovers_what_bbox_mode_rejects(gt, perfect):
    """SPEC.md section 10.1's point: bbox-free evaluation is a different, easier task. The
    same prediction that fails to localise scores full recall on content alone."""
    for c in perfect.characteristics:
        c.bbox = [10.0, 10.0, 40.0, 25.0]
    strict = evaluate_drawing(gt, perfect)
    lenient = evaluate_drawing(gt, perfect, no_bbox_config())
    assert lenient.recall == 1.0
    assert lenient.recall > (strict.recall or 0.0)


def test_a_kind_mismatch_can_never_match(gt):
    box = [100.0, 100.0, 160.0, 130.0]
    a = Characteristic(
        id=1, kind="dimension", view="front", bbox=box, dim_type="linear", nominal=10.0
    )
    b = Characteristic(
        id=1, kind="geometric_tolerance", view="front", bbox=box,
        gtol_symbol="flatness", gtol_value=0.02,
    )
    assert semantic_distance(a, b) == 1.0
    assert match([a], [b]).n_matched == 0


def test_the_assignment_is_global_not_greedy():
    """Two near-identical callouts must end up on the right ones, not swapped by whichever
    pair the matcher happened to consider first."""
    left = [0.0, 0.0, 40.0, 20.0]
    right = [200.0, 0.0, 240.0, 20.0]
    gt_items = [
        Characteristic(id=1, kind="dimension", view="front", bbox=left,
                       dim_type="diameter", nominal=12.0),
        Characteristic(id=2, kind="dimension", view="front", bbox=right,
                       dim_type="diameter", nominal=12.5),
    ]
    # Predictions arrive in the opposite order, each on the correct box.
    pred = [
        Characteristic(id=1, kind="dimension", view="front", bbox=right,
                       dim_type="diameter", nominal=12.5),
        Characteristic(id=2, kind="dimension", view="front", bbox=left,
                       dim_type="diameter", nominal=12.0),
    ]
    result = match(gt_items, pred)
    assert result.n_matched == 2
    assert {(m.gt_index, m.pred_index) for m in result.matched} == {(0, 1), (1, 0)}


def test_iou_is_the_usual_one():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 5]) == pytest.approx(0.5)


# --- tier 2: correctness ------------------------------------------------------------------


@pytest.mark.parametrize("wrong", [1, 2])
def test_a_wrong_nominal_is_counted_once_each(gt, perfect, wrong):
    dims = _dimensions(perfect)
    total = len(dims)
    for c in dims[:wrong]:
        c.nominal = (c.nominal or 0.0) + 7.0
    score = evaluate_drawing(gt, perfect)
    assert score.tier2["nominal_exact"].total == total
    assert score.tier2["nominal_exact"].value == (total - wrong) / total


def test_a_unit_conversion_changes_nothing(gt, perfect):
    """SPEC.md section 10.2 requires unit normalisation before comparison. A prediction
    given in inches on a millimetre drawing is right, and must score as right."""
    perfect.units = "inch"
    for c in perfect.characteristics:
        for attribute in ("nominal", "upper_tol", "lower_tol", "gtol_value"):
            value = getattr(c, attribute)
            if value is not None:
                setattr(c, attribute, value / 25.4)
    score = evaluate_drawing(gt, perfect)
    assert score.tier2["nominal_exact"].value == 1.0
    assert score.tier2["tolerance_exact"].value == 1.0
    assert score.exact_drawing is True
    assert to_mm(1.0, "inch") == 25.4


def test_within_one_lsd_is_looser_than_exact_but_not_by_much(gt, perfect):
    """An error of one unit in the last place shown passes the tolerant metric and fails the
    exact one; ten units fails both. That gap is the whole point of having two."""
    dims = [c for c in _dimensions(perfect) if c.upper_tol]
    if not dims:
        pytest.skip("this drawing has no toleranced dimension")
    target = dims[0]
    decimals = len(str(target.raw_text).split(".")[-1]) if "." in str(target.raw_text) else 2
    step = 10.0 ** -decimals
    target.upper_tol = (target.upper_tol or 0.0) + step

    score = evaluate_drawing(gt, perfect)
    assert score.tier2["tolerance_exact"].value < 1.0
    assert score.tier2["tolerance_within_1lsd"].value == 1.0


def test_a_wrong_symbol_shows_up_in_symbol_accuracy_only(gt, perfect):
    gtols = _gtols(perfect)
    total = len(gtols)
    target = gtols[0]
    original = target.gtol_symbol
    # Swap to another symbol with the same datum requirement, so the prediction stays a
    # loadable characteristic and the test measures the metric rather than the parser.
    target.gtol_symbol = "position" if original != "position" else "perpendicularity"

    score = evaluate_drawing(gt, perfect)
    assert score.tier2["symbol_accuracy"].value == (total - 1) / total
    assert score.tier2["datum_ref_exact"].value == 1.0


def test_a_dropped_modifier_shows_up_in_modifier_accuracy(gt, perfect):
    gtols = [c for c in _gtols(perfect) if c.material_modifier]
    if not gtols:
        pytest.skip("this drawing has no material modifier")
    total = len(_gtols(perfect))
    gtols[0].material_modifier = None
    score = evaluate_drawing(gt, perfect)
    assert score.tier2["modifier_accuracy"].value == (total - 1) / total
    assert score.tier2["symbol_accuracy"].value == 1.0


def test_reordered_datum_references_are_wrong(gt, perfect):
    """``|A|B|`` and ``|B|A|`` are different reference frames. A set comparison would call
    this correct, which is exactly the silent failure tier 3 exists to catch."""
    targets = [c for c in _gtols(perfect) if len(c.datum_refs) >= 2]
    if not targets:
        pytest.skip("this drawing has no multi-datum frame")
    total = len(_gtols(perfect))
    targets[0].datum_refs = list(reversed(targets[0].datum_refs))
    score = evaluate_drawing(gt, perfect)
    assert score.tier2["datum_ref_exact"].value == (total - 1) / total
    assert score.tier2["symbol_accuracy"].value == 1.0


# --- tier 3: structure --------------------------------------------------------------------


def test_the_datum_graph_distance_of_a_swapped_frame_is_exactly_predictable(gt, perfect):
    """Reversing one two-reference frame changes two edges on each side, so the symmetric
    difference is 4 out of a total graph size of twice the nodes plus twice the edges."""
    targets = [c for c in _gtols(perfect) if len(c.datum_refs) == 2]
    if not targets:
        pytest.skip("this drawing has no two-datum frame")
    targets[0].datum_refs = list(reversed(targets[0].datum_refs))

    nodes = len(gt.datums)
    edges = sum(len(c.datum_refs) for c in gt.characteristics)
    result = match(gt.characteristics, perfect.characteristics)
    distance = datum_graph_distance(gt, perfect, result)
    assert distance == pytest.approx(4 / (2 * (nodes + edges)))


def test_a_renamed_datum_moves_the_graph_distance_and_nothing_in_tier_1(gt, perfect):
    """The failure the tier exists for: every value right, every box right, full detection
    recall, and the reference frames wired to the wrong faces."""
    targets = [c for c in _gtols(perfect) if c.datum_refs]
    if not targets:
        pytest.skip("this drawing has no datum reference")
    targets[0].datum_refs = [
        DatumRef(label="Z", modifier=r.modifier) for r in targets[0].datum_refs
    ]
    score = evaluate_drawing(gt, perfect)
    assert score.recall == 1.0 and score.precision == 1.0
    assert score.graph_distance > 0.0
    assert score.exact_drawing is False


def test_full_drawing_exact_match_is_all_or_nothing(gt, perfect):
    assert evaluate_drawing(gt, perfect).exact_drawing is True
    _dimensions(perfect)[0].nominal = 12345.0
    assert evaluate_drawing(gt, perfect).exact_drawing is False


# --- tier 4: cost -------------------------------------------------------------------------


def test_the_cost_of_a_miss_is_the_configured_weight(gt, perfect):
    weights = error_costs()
    dropped = next(c for c in perfect.characteristics if not c.is_critical)
    perfect.characteristics = [c for c in perfect.characteristics if c is not dropped]
    score = evaluate_drawing(gt, perfect)
    assert score.cost == pytest.approx(weights["weights"]["missed_characteristic"]["cost"])
    assert set(score.cost_breakdown) == {"missed_characteristic"}


def test_a_critical_miss_costs_the_multiplier(gt, perfect):
    table = error_costs()
    critical = [c for c in perfect.characteristics if c.is_critical]
    if not critical:
        pytest.skip("this drawing has no critical characteristic")
    perfect.characteristics = [c for c in perfect.characteristics if c is not critical[0]]
    score = evaluate_drawing(gt, perfect)
    expected = (
        table["weights"]["missed_characteristic"]["cost"] * table["critical_multiplier"]
    )
    assert score.cost == pytest.approx(expected)


def test_ctq_recall_only_counts_critical_characteristics(gt, perfect):
    critical = [c for c in perfect.characteristics if c.is_critical]
    if not critical:
        pytest.skip("this drawing has no critical characteristic")
    perfect.characteristics = [c for c in perfect.characteristics if c is not critical[0]]
    score = evaluate_drawing(gt, perfect)
    assert score.ctq.total == len(critical)
    assert score.ctq.hits == len(critical) - 1


def test_two_errors_on_one_callout_are_charged_twice(gt, perfect):
    table = error_costs()["weights"]
    targets = [
        c for c in _gtols(perfect) if c.datum_refs and c.gtol_symbol == "position"
    ]
    if not targets:
        pytest.skip("this drawing has no position tolerance")
    target = targets[0]
    flag = target.is_critical
    multiplier = error_costs()["critical_multiplier"] if flag else 1.0
    target.gtol_symbol = "perpendicularity"
    target.datum_refs = [DatumRef(label=target.datum_refs[0].label, modifier=None)]

    score = evaluate_drawing(gt, perfect)
    expected = (
        table["wrong_symbol"]["cost"] + table["wrong_datum_refs"]["cost"]
    ) * multiplier
    assert score.cost == pytest.approx(expected)


def test_every_cost_weight_carries_a_written_justification():
    """CLAUDE.md's standing rule about defending choices in the file that makes them, applied
    to the one config a reader is most likely to disagree with."""
    table = error_costs()
    assert table["critical_multiplier_rationale"]
    for name, entry in table["weights"].items():
        assert entry["cost"] > 0, name
        assert len(entry["rationale"]) > 40, f"{name} has no real justification"


# --- aggregation and reporting ------------------------------------------------------------


def test_aggregation_sums_counts_rather_than_averaging_rates(gt, perfect):
    """A two-callout sheet must not outvote a twenty-callout one. Built from two scores with
    very different denominators, where the two answers differ."""
    big = evaluate_drawing(gt, perfect)

    lean = Prediction.from_ground_truth(gt)
    lean.characteristics = lean.characteristics[:2]
    small_gt = gt.model_copy(deep=True)
    small_gt.characteristics = small_gt.characteristics[:2]
    small_gt.datums = []
    small_gt.characteristics = [
        c for c in small_gt.characteristics if not c.datum_refs
    ] or small_gt.characteristics[:1]
    lean.characteristics = [lean.characteristics[0]]
    small = evaluate_drawing(small_gt, lean)

    summary = aggregate([big, small])
    assert summary.true_positives == big.true_positives + small.true_positives
    assert summary.recall == pytest.approx(
        (big.true_positives + small.true_positives)
        / (big.n_gt + small.n_gt)
    )


def test_per_profile_slices_appear_when_a_run_spans_profiles(gt, perfect):
    a = evaluate_drawing(gt, perfect)
    a.profile = "clean"
    b = evaluate_drawing(gt, Prediction.from_ground_truth(gt))
    b.profile = "scan_heavy"
    b.false_negatives += 1
    summary = aggregate([a, b])
    assert set(summary.by_profile) == {"clean", "scan_heavy"}
    assert summary.by_profile["clean"]["detection"]["recall"] == 1.0


def test_a_single_profile_run_does_not_repeat_itself(gt, perfect):
    summary = aggregate([evaluate_drawing(gt, perfect)])
    assert summary.by_profile == {}


def test_the_report_writes_json_markdown_and_plots(gt, perfect, tmp_path):
    scores = []
    for index, profile in enumerate(("clean", "scan_heavy", "phone_photo")):
        prediction = Prediction.from_ground_truth(gt)
        prediction.characteristics = prediction.characteristics[index:]
        score = evaluate_drawing(gt, prediction)
        score.profile = profile
        scores.append(score)

    written = write_report(scores, tmp_path, run_name="unit-test")
    assert written["json"].exists() and written["markdown"].exists()

    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["run"] == "unit-test"
    assert len(payload["drawings"]) == 3
    assert payload["summary"]["detection"]["tp"] == sum(s.true_positives for s in scores)

    table = markdown_table(aggregate(scores))
    assert "Tier 1" in table and "Tier 4" in table
    assert (tmp_path / "f1_by_profile.png").exists()


def test_a_config_can_change_the_threshold_without_touching_the_defaults():
    tight = replace(MatchConfig(), max_cost=0.05)
    assert MatchConfig().max_cost == 0.6
    assert tight.max_cost == 0.05
    assert tight.use_bbox is True
    assert no_bbox_config().use_bbox is False
