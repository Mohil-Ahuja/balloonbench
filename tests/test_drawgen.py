"""Milestone 2 acceptance tests for ``drawgen``.

SPEC.md section 7.5 sets four conditions: 200 drawings generate with zero exceptions, every
ground-truth box lies inside the image and contains ink, schema validation passes on all of
them, and a human accepts twenty renders. The first three are here. The fourth cannot be,
and pretending otherwise by asserting some proxy for "looks right" would be worse than
leaving it to a person.

The ink assertion is the load-bearing one. A bounding box that is inside the image and
contains no ink means the sheet and the ground truth disagree about where a callout is --
the exact failure mode CLAUDE.md's rule about computing boxes at placement time exists to
prevent -- and it is invisible to every other check, because the JSON validates, the image
renders, and only overlaying the two reveals it.
"""

from __future__ import annotations

import json

import pytest

from balloonbench.drawgen.annotate import _exit_distance
from balloonbench.drawgen.generate import generate_drawing
from balloonbench.drawgen.project import ViewTransform, project_standard
from balloonbench.drawgen.render import ink_in_box
from balloonbench.drawgen.styles import STYLES, get_style, sample_style, style_names
from balloonbench.drawgen.symbols import (
    GTOL_GLYPH,
    feature_control_frame,
    tolerance_text,
)
from balloonbench.drawgen.text import metrics_for
from balloonbench.drawgen.views import SheetLayout, layout_for_part
from balloonbench.partgen.registry import build_part, load_families
from balloonbench.schema import Drawing, GtolSymbol

FAMILIES = sorted(load_families())

#: DPI for the tests. Low enough to keep the sweep quick, high enough that a thin dimension
#: line still leaves a grey pixel inside its box -- below about 100 the ink test starts
#: failing on correct boxes, which would make it worse than useless.
TEST_DPI = 150.0

QUICK_SEEDS = 3
GATE_DRAWINGS = 200


# --- projection ---------------------------------------------------------------------------


def test_view_transform_is_orthonormal():
    t = ViewTransform.from_direction((0, 0, 1), (0, 1, 0))
    for a in (t.xdir, t.ydir, t.direction):
        assert abs(sum(c * c for c in a) - 1.0) < 1e-9
    assert abs(sum(a * b for a, b in zip(t.xdir, t.ydir, strict=True))) < 1e-9


def test_view_transform_rejects_degenerate_up():
    with pytest.raises(ValueError, match="parallel"):
        ViewTransform.from_direction((0, 0, 1), (0, 0, 1))


def test_view_transform_orthogonalises_a_skew_up():
    """An up vector not perpendicular to the view direction is projected, not rejected.

    Distances have to survive projection or a scaled drawing means nothing, and that only
    holds for an orthonormal basis.
    """
    t = ViewTransform.from_direction((0, 0, 1), (0, 1, 1))
    assert abs(t.ydir[2]) < 1e-9


def test_projection_of_a_box_has_the_right_extent():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10.0, 20.0, 30.0).Shape()
    for view, expected in (("front", (10, 30)), ("top", (10, 20)), ("right", (20, 30))):
        size = project_standard(box, view).size
        assert size == pytest.approx(expected, abs=1e-6)


def test_hidden_line_removal_actually_classifies():
    """A view drawn without hidden detail must still have run the removal.

    Skipping ``Hide()`` returns every edge on the visible layer, so a bore's far wall is
    drawn as a solid line. The test that catches it is that a solid with internal detail has
    strictly fewer visible edges than it has edges in total.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    block = BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, 0), 20.0, 20.0, 20.0).Shape()
    hole = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 4.0, 22.0
    ).Shape()
    cut = BRepAlgoAPI_Cut(block, hole).Shape()

    view = project_standard(cut, "front")
    assert view.of_layer("hidden"), "a blind bore seen from the side must have hidden lines"
    assert len(view.of_layer("visible")) < len(view.lines)


# --- the transform chain -------------------------------------------------------------------


def test_pixel_transform_flips_y():
    layout = SheetLayout("A4", 297.0, 210.0, (1, 1), "first_angle", ())
    assert layout.to_pixel((0.0, 0.0), 25.4) == pytest.approx((0.0, 210.0))
    assert layout.to_pixel((297.0, 210.0), 25.4) == pytest.approx((297.0, 0.0))


def test_bbox_to_pixel_reorders_after_the_flip():
    """The Y flip swaps which corner is minimal, so the result must be re-sorted.

    Without the re-sort the box comes back with ``y0 > y1``: still four plausible numbers,
    still inside the image, and rejected or silently normalised by every consumer.
    """
    layout = SheetLayout("A4", 297.0, 210.0, (1, 1), "first_angle", ())
    x0, y0, x1, y1 = layout.bbox_to_pixel((10.0, 20.0, 30.0, 40.0), 25.4)
    assert x0 < x1 and y0 < y1


def test_pixel_size_matches_the_rasteriser(tmp_path):
    """The predicted image size and the real one must agree, or every box is wrong."""
    bundle = generate_drawing(
        FAMILIES[0], 1, tmp_path, dpi=TEST_DPI, write_artifacts=False
    )
    from PIL import Image

    with Image.open(bundle.paths["png"]) as image:
        assert tuple(bundle.drawing.image_size) == image.size


def test_exit_distance_leaves_the_box():
    box = (0.0, 0.0, 10.0, 10.0)
    assert _exit_distance((5.0, 5.0), 0.0, box) == pytest.approx(5.0)
    assert _exit_distance((5.0, 5.0), 3.141592653589793 / 2, box) == pytest.approx(5.0)
    # A point already outside has nothing to clear.
    assert _exit_distance((50.0, 5.0), 0.0, box) == 0.0


# --- text and symbols ------------------------------------------------------------------------


def test_every_gtol_symbol_in_the_schema_has_a_glyph():
    """A symbol the schema allows but the renderer cannot draw would print a blank frame."""
    from typing import get_args

    assert set(get_args(GtolSymbol)) == set(GTOL_GLYPH)


def test_font_covers_every_glyph_the_renderer_uses():
    from balloonbench.drawgen.symbols import DIM_GLYPH, MODIFIER_GLYPH

    metrics = metrics_for(STYLES[0].font)
    for table in (GTOL_GLYPH, MODIFIER_GLYPH, DIM_GLYPH):
        for name, glyph in table.items():
            metrics.advance(glyph)
            assert glyph not in metrics.missing, f"{name} ({glyph!r}) not in the font"


def test_text_width_grows_with_length_and_height():
    m = metrics_for(STYLES[0].font)
    assert m.width("44", 3.5) < m.width("444", 3.5)
    assert m.width("44", 3.5) < m.width("44", 5.0)


def test_limit_style_states_absolute_limits_not_deviations():
    """The trap the benchmark exists to measure: 44.05/43.95 is not a nominal of 44.05."""
    assert tolerance_text(44, 0.05, -0.05, "limit", decimals=2) == "44.05/43.95"


def test_zero_deviation_keeps_its_sign():
    assert tolerance_text(44, 0.05, 0.0, "unilateral", decimals=0) == "44 +0.05/-0.00"


def test_frame_orders_compartments_symbol_tolerance_datums():
    text = feature_control_frame(
        "position", 0.25, zone_prefix="⌀", material_modifier="MMC",
        datum_refs=(("A", None), ("B", "MMC")), decimals=2,
    )
    assert text.split("|") == ["⌖", "⌀0.25Ⓜ", "A", "BⓂ"]


@pytest.mark.parametrize("name", style_names())
def test_every_style_is_self_consistent(name):
    style = get_style(name)
    assert style.tol_decimals >= style.decimals
    assert sum(style.tolerance_bias.values()) > 0


def test_style_sampling_is_seeded():
    import numpy as np

    a = sample_style(np.random.default_rng(7)).name
    b = sample_style(np.random.default_rng(7)).name
    assert a == b


# --- layout ------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_views_fit_inside_the_drawing_area(family):
    part = build_part(family, 5)
    layout = layout_for_part(part.shape, family, sheet="A3")
    ax0, ay0, ax1, ay1 = layout.drawing_area
    for placement in layout.placements:
        x0, y0, x1, y1 = placement.sheet_bounds
        assert ax0 <= x0 and x1 <= ax1, f"{placement.name} overruns the frame horizontally"
        assert ay0 <= y0 and y1 <= ay1, f"{placement.name} overruns the frame vertically"


@pytest.mark.parametrize("family", FAMILIES)
def test_projection_convention_mirrors_the_layout(family):
    """First and third angle must place the secondary views on opposite sides.

    The two conventions differ by exactly this mirror. A generator that ignored it would
    produce internally consistent drawings describing a mirrored part, and would do so
    without ever raising.
    """
    part = build_part(family, 5)
    first = layout_for_part(part.shape, family, projection="first_angle")
    third = layout_for_part(part.shape, family, projection="third_angle")
    if len(first.placements) < 2:
        pytest.skip(f"{family} is drawn in a single view")

    primary = first.placements[0].name
    other = next(p.name for p in first.placements if p.name != primary)

    def offset(layout):
        a = layout.placement(primary).sheet_bounds
        b = layout.placement(other).sheet_bounds
        return (
            (b[0] + b[2]) / 2 - (a[0] + a[2]) / 2,
            (b[1] + b[3]) / 2 - (a[1] + a[3]) / 2,
        )

    fx, fy = offset(first)
    tx, ty = offset(third)
    assert fx == pytest.approx(-tx, abs=1e-6)
    assert fy == pytest.approx(-ty, abs=1e-6)


def test_section_view_is_hatched():
    """A section without hatching is a wrong drawing, not a plain one."""
    families = [f for f in FAMILIES if any(
        s.section_normal is not None for s in __import__(
            "balloonbench.drawgen.views", fromlist=["view_plan"]
        ).view_plan(f)
    )]
    if not families:
        pytest.skip("no family is drawn with a section view")
    family = families[0]
    part = build_part(family, 5)
    layout = layout_for_part(part.shape, family)
    sections = [p for p in layout.placements if p.spec.section_normal is not None]
    assert sections
    for placement in sections:
        assert placement.hatch, f"{placement.name} was cut but not hatched"


# --- generation ----------------------------------------------------------------------------


def _check_bundle(bundle, png_path) -> None:
    drawing = bundle.drawing
    w, h = drawing.image_size

    boxes = [(f"datum {d.label}", d.bbox) for d in drawing.datums]
    boxes += [(f"characteristic {c.id}", c.bbox) for c in drawing.characteristics]
    for label, box in boxes:
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"{label} box {box} outside {w}x{h}"
        assert ink_in_box(png_path, box), f"{label} box {box} contains no ink"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(QUICK_SEEDS))
def test_drawing_generates_and_its_boxes_contain_ink(family, seed, tmp_path):
    bundle = generate_drawing(family, seed, tmp_path, dpi=TEST_DPI)
    assert bundle.drawing.characteristics, "a drawing with no callouts is not a drawing"
    _check_bundle(bundle, bundle.paths["png"])


@pytest.mark.parametrize("family", FAMILIES)
def test_all_artifacts_are_written(family, tmp_path):
    bundle = generate_drawing(family, 11, tmp_path, dpi=TEST_DPI)
    for key in ("pdf", "png", "dxf", "json", "overlay"):
        assert key in bundle.paths, f"{key} artifact missing"
        assert bundle.paths[key].exists()
        assert bundle.paths[key].stat().st_size > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_the_written_json_revalidates(family, tmp_path):
    """The file on disk must validate, not just the object in memory."""
    bundle = generate_drawing(family, 12, tmp_path, dpi=TEST_DPI)
    Drawing.from_json(bundle.paths["json"])


@pytest.mark.parametrize("family", FAMILIES)
def test_a_seed_reproduces_the_drawing(family, tmp_path):
    """CLAUDE.md: a seed must reproduce a drawing. Ground truth is the thing to compare.

    The JSON carries every decision the generator made -- style, projection, sheet, each
    tolerance presentation, every box -- so two runs agreeing on it means they agreed on
    everything that matters. Comparing PDF bytes instead would fail on the timestamp
    reportlab embeds, which says nothing about reproducibility.
    """
    a = generate_drawing(family, 21, tmp_path / "a", dpi=TEST_DPI)
    b = generate_drawing(family, 21, tmp_path / "b", dpi=TEST_DPI)
    left = json.loads(a.paths["json"].read_text(encoding="utf-8"))
    right = json.loads(b.paths["json"].read_text(encoding="utf-8"))
    left.pop("part_ref"), right.pop("part_ref")  # differs only by output directory
    assert left == right


@pytest.mark.parametrize("family", FAMILIES)
def test_datum_references_resolve(family, tmp_path):
    """No characteristic may reference a datum the sheet does not establish."""
    bundle = generate_drawing(family, 31, tmp_path, dpi=TEST_DPI, write_artifacts=False)
    labels = {d.label for d in bundle.drawing.datums}
    for c in bundle.drawing.characteristics:
        for ref in c.datum_refs:
            assert ref.label in labels


@pytest.mark.parametrize("family", FAMILIES)
def test_basic_dimensions_carry_no_tolerance(family, tmp_path):
    """R5, checked end to end: a located pattern's dimensions must be theoretically exact.

    A basic dimension alongside a positional tolerance is correct; a *toleranced* one is the
    most common error in hand-built GD&T, because it constrains the same variation twice.
    """
    bundle = generate_drawing(family, 32, tmp_path, dpi=TEST_DPI, write_artifacts=False)
    for c in bundle.drawing.characteristics:
        if c.is_basic:
            assert not c.upper_tol and not c.lower_tol


@pytest.mark.parametrize("family", FAMILIES)
def test_raw_text_matches_the_structured_fields(family, tmp_path):
    """What the sheet says and what the JSON says must be the same statement."""
    bundle = generate_drawing(family, 33, tmp_path, dpi=TEST_DPI, write_artifacts=False)
    for c in bundle.drawing.characteristics:
        assert c.raw_text, f"characteristic {c.id} has no transcription"
        if c.kind == "geometric_tolerance":
            assert c.raw_text.startswith(GTOL_GLYPH[c.gtol_symbol])
            for ref in c.datum_refs:
                assert ref.label in c.raw_text
        if c.fit_class:
            assert c.fit_class in c.raw_text


@pytest.mark.slow
def test_milestone_2_gate(tmp_path):
    """SPEC.md section 7.5: 200 drawings, no exceptions, schema valid, every box has ink."""
    per_family = GATE_DRAWINGS // len(FAMILIES)
    checked = 0
    for family in FAMILIES:
        for seed in range(1000, 1000 + per_family):
            bundle = generate_drawing(family, seed, tmp_path, dpi=TEST_DPI)
            _check_bundle(bundle, bundle.paths["png"])
            for path in bundle.paths.values():
                path.unlink(missing_ok=True)
            checked += 1
    assert checked == per_family * len(FAMILIES)
