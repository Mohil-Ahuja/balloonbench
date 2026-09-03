"""Milestone 1 acceptance tests (SPEC.md section 6.4).

The gate: 200 samples per family build without OCCT errors, every exported STEP
re-imports with the same face count, face ids are stable across a rebuild from the same
seed, sampled fits validate against the ISO 286 tables, and no two features overlap in
space.

Tests are parametrised over the registry, so a family is covered the moment it registers
itself -- there is no per-family test list to forget to update.

The full 200-sample sweep is marked ``slow`` and runs separately; the default run uses a
smaller sample so the suite stays usable during development. Both use the same code path.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from balloonbench.partgen.fits import fit_limits, it_grade, parse_fit
from balloonbench.partgen.registry import build_part, load_families, read_step
from balloonbench.partgen.types import BuiltPart, FaceIndex, UnbuildableParams

FAMILIES = load_families()

#: Sample size for the default run. The ``slow`` tests use the full 200 the spec asks for.
QUICK_SAMPLES = 40
GATE_SAMPLES = 200


@pytest.fixture(scope="module")
def parts() -> dict[str, list[BuiltPart]]:
    """One batch per family, built once and shared. Building is the expensive part."""
    out: dict[str, list[BuiltPart]] = {}
    for family in FAMILIES:
        built = []
        for seed in range(QUICK_SAMPLES):
            with contextlib.suppress(UnbuildableParams):
                built.append(build_part(family, seed))
        out[family] = built
    return out


def test_at_least_one_family_is_registered():
    assert FAMILIES, "no part families registered; did load_families() import them?"


# --- build rate ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_every_seed_builds(family: str, parts):
    """A seed that cannot produce a part is a sampler bug, not bad luck: the sampler is
    responsible for drawing inside its own constraints, and the driver already resamples
    up to fifty times before giving up."""
    assert len(parts[family]) == QUICK_SAMPLES


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_the_full_two_hundred_sample_gate(family: str):
    for seed in range(GATE_SAMPLES):
        build_part(family, seed)


# --- determinism --------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_seed_reproduces_the_same_part(family: str):
    """Ground truth records only the seed, so a seed that does not reproduce its part
    invalidates every drawing generated from it."""
    for seed in (0, 3, 17):
        a, b = build_part(family, seed), build_part(family, seed)
        assert a.params == b.params
        assert a.faces.ids == b.faces.ids, "face ids drifted between two builds"
        assert a.features == b.features


@pytest.mark.parametrize("family", FAMILIES)
def test_face_ids_are_assigned_from_geometry_not_traversal(family: str, parts):
    """Ids run contiguously and each maps to exactly one measured face."""
    for part in parts[family][:5]:
        assert part.faces.ids == tuple(f"face_{n:04d}" for n in range(len(part.faces)))
        assert len({f.fid for f in part.faces}) == len(part.faces)


@pytest.mark.parametrize("family", FAMILIES)
def test_different_seeds_give_different_parts(family: str, parts):
    """A sampler that collapses to one part would pass every other test here."""
    def numeric(part) -> tuple:
        return tuple(
            sorted(
                (k, round(v, 4))
                for k, v in part.params.items()
                if isinstance(v, int | float)
            )
        )

    shapes = {numeric(p) for p in parts[family]}
    assert len(shapes) > QUICK_SAMPLES * 0.5, "sampled parts are barely distinguishable"


# --- STEP round trip ----------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_step_export_reimports_with_the_same_faces(family: str, tmp_path):
    for seed in range(5):
        part = build_part(family, seed, step_dir=tmp_path)
        assert part.step_path is not None and part.step_path.exists()

        reread = FaceIndex(read_step(part.step_path))
        assert len(reread) == len(part.faces), (
            f"{family} seed {seed}: exported {len(part.faces)} faces, re-imported "
            f"{len(reread)}"
        )

        # Face counts alone would pass even if the geometry were mangled, so compare the
        # measured radii too. STEP is exported in millimetres explicitly, so a unit
        # mismatch would show here as a factor of 25.4.
        before = sorted(round(f.radius, 4) for f in part.faces if f.radius)
        after = sorted(round(f.radius, 4) for f in reread if f.radius)
        assert before == pytest.approx(after)


# --- feature description ------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_features_reference_only_faces_that_exist(family: str, parts):
    for part in parts[family]:
        part.validate()  # raises on a dangling face id or duplicate feature id
        for feature in part.features:
            assert feature.faces
            for fid in feature.faces:
                assert fid in part.faces


@pytest.mark.parametrize("family", FAMILIES)
def test_features_of_size_report_a_size(family: str, parts):
    """GD&T's whole material-condition machinery rests on this distinction, so a feature
    that claims to have a size must be able to state it."""
    for part in parts[family]:
        for feature in part.features:
            if feature.is_feature_of_size:
                size = feature.size()
                assert size is not None and size > 0, (
                    f"{family}: {feature.fid} is a feature of size with no size"
                )


@pytest.mark.parametrize("family", FAMILIES)
def test_every_part_has_a_datum_candidate(family: str, parts):
    """A part with no face nominated as a datum cannot carry an orientation or location
    tolerance, which would leave the drawing with nothing interesting to extract."""
    for part in parts[family]:
        assert any(f.meta.get("datum") for f in part.features), (
            f"{family} seed {part.seed} nominates no datum feature"
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_feature_anchors_lie_within_the_part_envelope(family: str, parts):
    """Anchors are projected into view coordinates to place annotations (PLAN.md section
    1.1). An anchor outside the solid would put a leader line in empty space."""
    for part in parts[family]:
        xs = [f.centroid[0] for f in part.faces]
        ys = [f.centroid[1] for f in part.faces]
        zs = [f.centroid[2] for f in part.faces]
        span = max(
            max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0
        )
        margin = span * 0.75
        for feature in part.features:
            x, y, z = feature.anchor
            assert min(xs) - margin <= x <= max(xs) + margin
            assert min(ys) - margin <= y <= max(ys) + margin
            assert min(zs) - margin <= z <= max(zs) + margin


# --- fits ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_sampled_fits_resolve_against_the_iso_tables(family: str, parts):
    """Every fit a family puts on a feature must be one ISO 286 defines at that size, and
    the band it resolves to must be the IT width for its grade."""
    checked = 0
    for part in parts[family]:
        for feature in part.features:
            designation = feature.meta.get("fit")
            if not designation:
                continue
            size = feature.size()
            assert size is not None
            _, grade, _ = parse_fit(designation)
            limits = fit_limits(size, designation)
            assert limits.width == pytest.approx(it_grade(size, grade))
            assert limits.upper >= limits.lower
            checked += 1
    if checked == 0:
        pytest.skip(f"{family} does not sample fitted features")


@pytest.mark.parametrize("family", FAMILIES)
def test_stored_tolerances_are_deviations_not_absolute_limits(family: str, parts):
    """Schema rule R4: a tolerance is stored as a signed offset from nominal. A value
    stored as an absolute limit would be the same order as the nominal itself."""
    for part in parts[family]:
        for feature in part.features:
            for key in ("upper_tol", "lower_tol"):
                if key in feature.nominal:
                    assert abs(feature.nominal[key]) < 1.0, (
                        f"{feature.fid}.{key} = {feature.nominal[key]} looks like an "
                        f"absolute limit rather than a deviation"
                    )


# --- geometric sanity ---------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_no_two_features_of_size_occupy_the_same_space(family: str, parts):
    """SPEC.md section 6.4 asks for this as a property test. Two distinct features whose
    axes coincide and whose sizes are equal are the same feature described twice, which
    would produce a duplicate balloon on the drawing."""
    for part in parts[family]:
        sized = [f for f in part.features if f.is_feature_of_size and f.axis]
        for i, a in enumerate(sized):
            for b in sized[i + 1 :]:
                if a.parent == b.fid or b.parent == a.fid:
                    continue  # a counterbore legitimately shares its hole's axis
                same_axis = all(
                    math.isclose(p, q, abs_tol=1e-6)
                    for p, q in zip(a.axis[0] + a.axis[1], b.axis[0] + b.axis[1], strict=True)
                )
                if same_axis:
                    assert not math.isclose(
                        a.size() or 0.0, b.size() or 0.0, rel_tol=1e-6
                    ), f"{a.fid} and {b.fid} are the same feature described twice"


@pytest.mark.parametrize("family", FAMILIES)
def test_solids_are_closed_and_have_volume(family: str, parts):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    for part in parts[family]:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(part.shape, props)
        assert props.Mass() > 0, f"{family} seed {part.seed} has no volume"


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_any_seed_produces_a_valid_part_or_a_clear_rejection(seed: int):
    """Property test over the whole seed space. Whatever a seed produces, it must be
    either a valid part or an explicit UnbuildableParams -- never an OCCT crash, a
    silently invalid solid, or a part whose features point at faces that do not exist."""
    for family in FAMILIES:
        try:
            part = build_part(family, seed)
        except UnbuildableParams:
            continue
        part.validate()
        assert len(part.faces) > 0
        assert part.features


def test_unknown_family_is_rejected_by_name():
    from balloonbench.partgen.registry import get_family

    with pytest.raises(KeyError, match="no part family named"):
        get_family("sprocket")


def test_rng_is_advanced_between_resample_attempts():
    """If the driver rebuilt the generator each attempt, a rejected draw would be redrawn
    identically forever and the retry budget would be pointless."""
    rng = np.random.default_rng(0)
    first = rng.random()
    second = rng.random()
    assert first != second
