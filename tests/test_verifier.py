"""Milestone 7 acceptance tests for the geometry-grounded verifier.

PLAN.md's M7 gate is a number: **false-positive rate below 5% on clean ground truth**, with
injected-error recall reported alongside. SPEC.md section 12.4 explains why that is the
headline and not recall -- *a verifier that flags correct extractions is worse than no
verifier* -- so the gate is asymmetric on purpose, and so is the module it measures.

The false-positive test counts contradictions against ground truth that is correct by
construction: it came out of the same generator that built the solid, so every number on it
is true. Any contradiction there is the verifier being wrong.

The recall tests damage that ground truth in counted ways and ask what the verifier notices.
They compare against the clean report and count only *new* findings, which matters more than
it sounds: an earlier version of this harness counted any finding at all, and scored a
verifier as catching errors it had not noticed, using complaints it had been making about
the undamaged drawing all along.

Some injections are geometrically undetectable and are labelled so rather than counted as
misses. Swapping datum A for datum B between two frames that both constrain the part
adequately produces a drawing that is still internally consistent and still satisfiable by
the solid; no check working from geometry alone can tell the two apart, and pushing it to
try is exactly how the false-positive rate goes up. Catching that error is ``evalkit``'s
job -- its tier-3 datum graph distance has ground truth to compare against, and this does
not.
"""

from __future__ import annotations

import json

import pytest

from balloonbench.drawgen.generate import generate_drawing
from balloonbench.partgen.registry import build_part, load_families
from balloonbench.schema import Drawing
from balloonbench.verifier import BrepIndex, inject, injector_names, verify_drawing
from balloonbench.verifier.brep_index import Candidate
from balloonbench.verifier.checks import datum_dof

FAMILIES = sorted(load_families())
load_families()

#: Kept small in the default suite and swept properly in the slow gate below.
QUICK_SEEDS = 2
GATE_SEEDS = 8

#: PLAN.md's M7 gate.
MAX_FALSE_POSITIVE_RATE = 0.05


@pytest.fixture(scope="module")
def flange(tmp_path_factory):
    out = tmp_path_factory.mktemp("verify")
    bundle = generate_drawing("flange", 5, out, dpi=100, write_artifacts=False)
    return bundle, BrepIndex.from_shape(bundle.part.shape)


def _findings(report) -> set:
    """What the verifier positively asserts is wrong, as comparable keys."""
    return {("contradicted", v.id) for v in report.per_characteristic
            if v.verdict == "contradicted"} | {
        ("defect", d.type, d.characteristic_id) for d in report.drawing_defects
    }


# --- the B-rep index ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_index_measures_the_solid(family):
    part = build_part(family, 5)
    index = BrepIndex.from_shape(part.shape)
    stats = index.stats()
    assert stats["faces"] > 0
    assert stats["usable"] == stats["faces"] - stats["skipped"]
    assert all(extent > 0 for extent in index.envelope)
    assert index.envelope == tuple(sorted(index.envelope, reverse=True))


def test_a_bolt_circle_is_recovered_from_geometry_alone():
    """The pattern is found by measuring the solid, never by reading partgen's parameters.
    Anything else would make the check circular, and would not work on an imported STEP."""
    part = build_part("flange", 5)
    index = BrepIndex.from_shape(part.shape)
    patterns = [p for p in index.hole_patterns() if p.uniform]
    assert patterns, "no bolt circle found on a flange"
    pattern = patterns[0]
    assert pattern.count >= 4
    assert pattern.bolt_circle > 2 * pattern.radius


def test_a_rectangular_pattern_is_not_called_a_bolt_circle():
    """Every rectangle's corners are equidistant from its centre, so distance alone says
    four holes on a plate lie on a circle. Only equal angular spacing makes it one."""
    part = build_part("plate_bracket", 5)
    index = BrepIndex.from_shape(part.shape)
    patterns = index.hole_patterns()
    assert patterns, "no hole pattern found on a bracket"
    assert not any(p.uniform for p in patterns if p.count == 4), (
        "a rectangular four-hole pattern was reported as a bolt circle"
    )


def test_a_drafted_wall_is_found_by_a_diameter_dimension():
    """A cast wall with draft is a cone, not a cylinder. Matching only cylinders
    contradicted the valve body's ⌀100 body wall on every seed."""
    part = build_part("valve_body", 2)
    index = BrepIndex.from_shape(part.shape)
    assert index.cones, "the valve body has no conical face"
    assert any(
        candidate.value > 0 for candidate in index.diameter_candidates(100.0, 1.0)
    )


def test_a_hole_to_edge_distance_is_indexed():
    """The commonest dimension on a plate is neither a plane pair nor an axis pair."""
    part = build_part("plate_bracket", 5)
    index = BrepIndex.from_shape(part.shape)
    assert index.axis_to_plane_distances()


def test_a_tap_drill_is_only_offered_for_a_standard_thread_size():
    """The exemption that lets a ⌀14 callout match a ⌀12 hole must not become a licence to
    excuse any near miss: an earlier ratio-based version waved through a nominal misread as
    125 because the part had a ⌀100 face."""
    part = build_part("housing", 5)
    index = BrepIndex.from_shape(part.shape)
    assert index.tap_drill_for(14.0), "M14 should find its drilled hole"
    assert not index.tap_drill_for(125.0), "125 is not a thread size"
    assert not index.tap_drill_for(100.0)


def test_the_index_reads_a_step_file(tmp_path):
    """The stress test of SPEC.md section 12.5 needs this path: solids nobody here built."""
    part = build_part("shaft", 3, step_dir=tmp_path)
    assert part.step_path is not None
    index = BrepIndex.from_step(part.step_path)
    assert len(index.faces) > 0
    assert index.cylinders


def test_a_candidate_reports_its_own_error():
    candidate = Candidate(value=44.05, kind="diameter")
    assert candidate.error(44.0) == pytest.approx(0.05)


# --- the degrees-of-freedom table --------------------------------------------------------


def test_the_dof_table_is_cumulative_and_capped():
    assert datum_dof.constrained_dof(["planar_face"]) == 3
    assert datum_dof.constrained_dof(["cylindrical_feature"]) == 4
    assert datum_dof.constrained_dof(["planar_face", "planar_face"]) == 5
    assert datum_dof.constrained_dof(["planar_face", "planar_face", "planar_face"]) == 6
    assert datum_dof.constrained_dof(["cylindrical_feature", "planar_face", "axis"]) == 6


def test_a_single_planar_datum_cannot_locate_a_feature():
    """SPEC.md section 12.2's own example of a defect real drawings contain constantly."""
    assert not datum_dof.is_sufficient("position", ["planar_face"])
    assert datum_dof.is_sufficient("position", ["planar_face", "planar_face", "planar_face"])


def test_a_plane_and_a_bore_are_enough_to_locate_a_bolt_pattern():
    """The deliberately lenient case. A circular pattern positioned from its own bore leaves
    rotation about that bore free, which is how flanges are drawn -- flagging it would fire
    on almost every flange ever made, which is how a verifier teaches people to ignore it."""
    assert datum_dof.is_sufficient("position", ["planar_face", "cylindrical_feature"])


def test_an_orientation_tolerance_needs_only_a_direction():
    assert datum_dof.is_sufficient("perpendicularity", ["planar_face"])


def test_runout_needs_something_to_spin_about():
    assert not datum_dof.is_sufficient("circular_runout", ["planar_face"])
    assert datum_dof.is_sufficient("circular_runout", ["cylindrical_feature"])


# --- the report --------------------------------------------------------------------------


def test_a_clean_drawing_produces_no_contradiction(flange):
    bundle, index = flange
    report = verify_drawing(bundle.drawing, index)
    assert report.contradicted == []
    assert report.summary["verified"] > 0
    assert sum(report.summary.values()) == len(bundle.drawing.characteristics)


def test_every_characteristic_gets_a_verdict(flange):
    """Silence would be indistinguishable from approval."""
    bundle, index = flange
    report = verify_drawing(bundle.drawing, index)
    assert {v.id for v in report.per_characteristic} == {
        c.id for c in bundle.drawing.characteristics
    }


def test_the_worst_verdict_wins(flange):
    """A check only contradicts when geometry positively disagrees, so one contradiction
    stands even when three other checks were happy."""
    bundle, index = flange
    damaged = bundle.drawing.model_copy(deep=True)
    target = next(c for c in damaged.characteristics if c.kind == "dimension")
    target.nominal = 12345.0
    report = verify_drawing(damaged, index)
    assert report.verdict_for(target.id) == "contradicted"


def test_the_report_matches_the_documented_shape(flange, tmp_path):
    bundle, index = flange
    report = verify_drawing(bundle.drawing, index)
    path = tmp_path / "verification.json"
    report.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) >= {"drawing_id", "summary", "per_characteristic", "drawing_defects"}
    assert set(payload["summary"]) == {"verified", "contradicted", "unverifiable"}
    for entry in payload["per_characteristic"]:
        assert set(entry) >= {"id", "verdict", "check", "detail", "confidence"}


def test_a_contradiction_carries_the_near_misses_that_explain_it(flange):
    """"No ⌀22 here" and "no ⌀22 here; the part has ⌀20 and ⌀25" are the difference between
    a complaint and a suggested correction."""
    bundle, index = flange
    damaged = bundle.drawing.model_copy(deep=True)
    target = next(
        c for c in damaged.characteristics
        if c.kind == "dimension" and c.dim_type == "diameter"
    )
    target.nominal = (target.nominal or 0.0) * 3
    report = verify_drawing(damaged, index)
    entry = next(v for v in report.per_characteristic if v.id == target.id)
    assert entry.verdict == "contradicted"
    assert "Nearest on the solid" in entry.detail


def test_an_inch_drawing_is_not_contradicted_for_being_in_inches(tmp_path):
    """Values reach a check already converted, so an imperial sheet is not a special case
    anyone has to remember -- forgetting it once would contradict every dimension on it."""
    from balloonbench.drawgen.styles import get_style

    bundle = generate_drawing(
        "flange", 5, tmp_path, dpi=100, style=get_style("imperial_shop"),
        write_artifacts=False,
    )
    if bundle.drawing.units != "inch":
        pytest.skip("this seed did not produce an imperial sheet")
    index = BrepIndex.from_shape(bundle.part.shape)
    assert verify_drawing(bundle.drawing, index).contradicted == []


# --- the injection harness ---------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(injector_names()))
def test_every_injector_damages_something_and_says_what(flange, kind):
    bundle, _ = flange
    damaged, injection = inject(bundle.drawing, kind, 7)
    if injection is None:
        pytest.skip(f"{kind} found nothing to damage on this drawing")
    assert injection.kind == kind
    assert injection.detail
    assert damaged.model_dump_json() != bundle.drawing.model_dump_json()


def test_injection_does_not_touch_the_original(flange):
    """Injecting into the same ground truth repeatedly must not compound, or every rate
    measured after the first is meaningless."""
    bundle, _ = flange
    before = bundle.drawing.model_dump_json()
    for seed in range(5):
        inject(bundle.drawing, "perturb_nominal", seed)
    assert bundle.drawing.model_dump_json() == before


def test_a_damaged_drawing_still_validates(flange):
    """The injections model extraction errors, not corrupt files. A damaged drawing that no
    longer loads would be caught by the schema rather than by the verifier."""
    bundle, _ = flange
    for kind in injector_names():
        damaged, injection = inject(bundle.drawing, kind, 3)
        if injection is None or kind == "undeclared_datum":
            # An undeclared reference is exactly the schema violation R4 exists for; the
            # verifier reports it as a drawing defect, and the schema would refuse to load
            # it. Both are right, and this test is about the others.
            continue
        Drawing.model_validate(damaged.model_dump())


def test_an_unknown_injection_is_refused(flange):
    bundle, _ = flange
    with pytest.raises(KeyError, match="unknown injection"):
        inject(bundle.drawing, "set_it_on_fire", 1)


# --- the checks, each on the error it exists for ------------------------------------------


def test_stripping_a_frame_to_one_planar_datum_is_caught(flange):
    bundle, index = flange
    base = _findings(verify_drawing(bundle.drawing, index))
    damaged, injection = inject(bundle.drawing, "underconstrain_drf", 11)
    if injection is None or injection.benign:
        pytest.skip("this drawing has no frame that becomes insufficient")
    new = _findings(verify_drawing(damaged, index)) - base
    assert any(finding[0] == "defect" and "underconstrained" in finding[1] for finding in new)


def test_referencing_a_datum_the_drawing_never_establishes_is_caught(flange):
    bundle, index = flange
    base = _findings(verify_drawing(bundle.drawing, index))
    damaged, injection = inject(bundle.drawing, "undeclared_datum", 11)
    assert injection is not None
    new = _findings(verify_drawing(damaged, index)) - base
    assert any("undeclared_datum" in str(finding) for finding in new)


def test_a_whole_sheet_in_the_wrong_unit_is_caught(flange):
    """The failure is never one wrong number. It is every number at once, which is what
    makes it detectable by a ratio and undetectable by reading any one of them."""
    bundle, index = flange
    base = _findings(verify_drawing(bundle.drawing, index))
    damaged, injection = inject(bundle.drawing, "unit_confusion", 11)
    assert injection is not None
    new = _findings(verify_drawing(damaged, index)) - base
    assert any("unit_mismatch" in str(finding) for finding in new)


def test_an_over_dimensioned_chain_is_reported_as_a_drawing_defect(flange):
    """Three steps plus the overall length, all toleranced, is unbuildable: two of the four
    constraints fight each other on every part made."""
    bundle, index = flange
    damaged = bundle.drawing.model_copy(deep=True)
    view = damaged.characteristics[0].view
    box = list(damaged.characteristics[0].bbox)
    made = []
    for offset, value in enumerate((20.0, 30.0, 50.0)):
        made.append(
            {
                "id": len(damaged.characteristics) + offset + 1,
                "kind": "dimension",
                "view": view,
                "bbox": [box[0], box[1] + offset, box[2], box[3] + offset],
                "dim_type": "linear",
                "nominal": value,
                "upper_tol": 0.1,
                "lower_tol": -0.1,
                "tol_style": "bilateral",
                "raw_text": f"{value:g} ±0.1",
            }
        )
    from balloonbench.schema import Characteristic

    damaged.characteristics = [
        *damaged.characteristics,
        *[Characteristic.model_validate(item) for item in made],
    ]
    report = verify_drawing(damaged, index)
    assert any(d.type == "overdimensioned_chain" for d in report.drawing_defects)


# --- the gate ----------------------------------------------------------------------------


def _sweep(seeds: range, tmp_path):
    cases = []
    for family in FAMILIES:
        for seed in seeds:
            bundle = generate_drawing(
                family, seed, tmp_path, dpi=100, write_artifacts=False
            )
            index = BrepIndex.from_shape(bundle.part.shape)
            cases.append((bundle.drawing, index))
    return cases


def _false_positive_rate(cases) -> tuple[int, int]:
    contradicted = total = 0
    for drawing, index in cases:
        report = verify_drawing(drawing, index)
        contradicted += len(report.contradicted)
        total += len(drawing.characteristics)
    return contradicted, total


def test_false_positive_rate_on_clean_ground_truth(tmp_path):
    """PLAN.md's M7 gate, on a quick sweep. The full one is the slow test below."""
    contradicted, total = _false_positive_rate(_sweep(range(QUICK_SEEDS), tmp_path))
    assert total > 0
    rate = contradicted / total
    assert rate <= MAX_FALSE_POSITIVE_RATE, (
        f"{contradicted} of {total} correct characteristics were contradicted "
        f"({rate:.1%}); the gate is {MAX_FALSE_POSITIVE_RATE:.0%}"
    )


@pytest.mark.slow
def test_milestone_7_gate(tmp_path):
    """The full sweep: false-positive rate under 5%, and injected-error recall reported.

    Recall is asserted only where an error is geometrically detectable at all. The rest is
    printed for the record, because a number nobody can see is not a measurement.
    """
    cases = _sweep(range(GATE_SEEDS), tmp_path)
    contradicted, total = _false_positive_rate(cases)
    rate = contradicted / total
    print(f"\nfalse positives: {contradicted}/{total} = {rate:.2%}")
    assert rate <= MAX_FALSE_POSITIVE_RATE

    baselines = [(d, i, _findings(verify_drawing(d, i))) for d, i in cases]
    detectable_total = detectable_caught = 0
    for kind in injector_names():
        caught = missed = undetectable = 0
        for offset, (drawing, index, base) in enumerate(baselines):
            damaged, injection = inject(drawing, kind, 1000 + offset)
            if injection is None:
                continue
            if injection.benign:
                undetectable += 1
                continue
            if _findings(verify_drawing(damaged, index)) - base:
                caught += 1
            else:
                missed += 1
        seen = caught + missed
        detectable_total += seen
        detectable_caught += caught
        share = f"{caught / seen:.0%}" if seen else "n/a"
        print(f"{kind:22s} {caught:3d}/{seen:3d} = {share:>4s}  "
              f"({undetectable} undetectable by geometry)")

    assert detectable_total > 0, "no injection was detectable; the harness measures nothing"
    assert detectable_caught / detectable_total >= 0.75, (
        f"recall on geometrically detectable errors is "
        f"{detectable_caught}/{detectable_total}"
    )
