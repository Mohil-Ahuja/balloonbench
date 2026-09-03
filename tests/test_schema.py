"""SPEC.md section 4 requires a failing case for every schema rule. This is that file.

Each test mutates the valid reference drawing in exactly one way and asserts the
validator rejects it, matching on the message so a rule cannot silently be caught by a
different rule.
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from balloonbench.schema import Drawing, load_json_schema


def _expect_reject(payload: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Drawing.model_validate(payload)


# --- the reference instance ---------------------------------------------------------


def test_valid_drawing_passes_pydantic(valid_drawing):
    d = Drawing.model_validate(valid_drawing)
    assert d.drawing_id == "syn_flange_00417"
    assert [c.id for c in d.characteristics] == [1, 2, 3]


def test_valid_drawing_passes_json_schema(valid_drawing):
    Draft202012Validator(load_json_schema()).validate(valid_drawing)


def test_json_schema_document_is_itself_valid():
    Draft202012Validator.check_schema(load_json_schema())


def test_round_trip_through_json(valid_drawing):
    d = Drawing.model_validate(valid_drawing)
    reparsed = Drawing.model_validate(json.loads(d.model_dump_json()))
    assert reparsed == d


# --- R1: ids unique and contiguous from 1 -------------------------------------------


def test_duplicate_balloon_id_rejected(valid_drawing):
    valid_drawing["characteristics"][2]["id"] = 1
    _expect_reject(valid_drawing, "duplicate balloon ids")


def test_non_contiguous_balloon_ids_rejected(valid_drawing):
    valid_drawing["characteristics"][2]["id"] = 7
    _expect_reject(valid_drawing, "contiguous from 1")


def test_balloon_ids_not_starting_at_one_rejected(valid_drawing):
    for offset, c in enumerate(valid_drawing["characteristics"], start=2):
        c["id"] = offset
    _expect_reject(valid_drawing, "contiguous from 1")


# --- R2: datum references per symbol class ------------------------------------------


def test_position_without_datum_reference_rejected(valid_drawing):
    valid_drawing["characteristics"][1]["datum_refs"] = []
    valid_drawing["characteristics"][1]["material_modifier"] = None
    _expect_reject(valid_drawing, "requires at least one datum reference")


@pytest.mark.parametrize(
    "symbol",
    [
        "position",
        "perpendicularity",
        "parallelism",
        "angularity",
        "concentricity",
        "symmetry",
        "circular_runout",
        "total_runout",
    ],
)
def test_every_datum_requiring_symbol_rejects_empty_refs(valid_drawing, symbol):
    c = valid_drawing["characteristics"][1]
    c["gtol_symbol"] = symbol
    c["datum_refs"] = []
    c["material_modifier"] = None
    c["gtol_zone"] = "linear"
    _expect_reject(valid_drawing, "requires at least one datum reference")


@pytest.mark.parametrize(
    "symbol", ["flatness", "straightness", "circularity", "cylindricity"]
)
def test_form_tolerance_with_datum_reference_rejected(valid_drawing, symbol):
    c = valid_drawing["characteristics"][2]
    c["gtol_symbol"] = symbol
    c["datum_refs"] = [{"label": "A", "modifier": None}]
    _expect_reject(valid_drawing, "must not reference a datum")


def test_reference_to_undeclared_datum_rejected(valid_drawing):
    valid_drawing["characteristics"][1]["datum_refs"] = [
        {"label": "Z", "modifier": None}
    ]
    _expect_reject(valid_drawing, "not declared in datums")


def test_repeated_datum_reference_rejected(valid_drawing):
    valid_drawing["characteristics"][1]["datum_refs"] = [
        {"label": "A", "modifier": None},
        {"label": "A", "modifier": None},
    ]
    _expect_reject(valid_drawing, "repeated datum reference")


# --- R3: material condition modifiers only on features of size ----------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "flatness",
        "straightness",
        "circularity",
        "cylindricity",
        "profile_surface",
        "circular_runout",
        "total_runout",
    ],
)
def test_modifier_on_non_feature_of_size_rejected(valid_drawing, symbol):
    c = valid_drawing["characteristics"][1]
    c["gtol_symbol"] = symbol
    c["material_modifier"] = "MMC"
    if symbol in {"flatness", "straightness", "circularity", "cylindricity"}:
        c["datum_refs"] = []
    _expect_reject(valid_drawing, "cannot carry a MMC modifier")


def test_modifier_on_planar_datum_reference_rejected(valid_drawing):
    # Datum A is a planar face; a planar datum has no size to depart from.
    valid_drawing["characteristics"][1]["datum_refs"][0]["modifier"] = "MMC"
    _expect_reject(valid_drawing, "which has no size")


def test_modifier_on_dimension_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["material_modifier"] = "MMC"
    _expect_reject(valid_drawing, "only meaningful on a geometric")


def test_datum_refs_on_dimension_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["datum_refs"] = [
        {"label": "A", "modifier": None}
    ]
    _expect_reject(valid_drawing, "only meaningful on a geometric tolerance")


# --- R4: deviations are signed and ordered ------------------------------------------


def test_upper_below_lower_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["upper_tol"] = -0.05
    valid_drawing["characteristics"][0]["lower_tol"] = 0.05
    _expect_reject(valid_drawing, "signed deviations from nominal")


def test_limit_style_stores_deviations_not_absolute_limits(valid_drawing):
    # 44.05 / 44.00 displayed, stored as +0.05 / 0.00 about a nominal of 44.
    c = valid_drawing["characteristics"][0]
    c.update(
        tol_style="limit", fit_class=None, nominal=44.0, upper_tol=0.05, lower_tol=0.0
    )
    d = Drawing.model_validate(valid_drawing)
    assert d.characteristics[0].nominal == 44.0
    assert d.characteristics[0].upper_tol == 0.05


def test_fit_class_without_fit_style_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["tol_style"] = "bilateral"
    _expect_reject(valid_drawing, "requires tol_style 'fit'")


# --- R5: basic dimensions are theoretically exact -----------------------------------


def test_basic_dimension_with_tolerance_rejected(valid_drawing):
    c = valid_drawing["characteristics"][0]
    c.update(is_basic=True, tol_style="basic", fit_class=None, upper_tol=0.05)
    _expect_reject(valid_drawing, "theoretically exact")


def test_basic_dimension_with_zero_tolerance_accepted(valid_drawing):
    c = valid_drawing["characteristics"][0]
    c.update(
        is_basic=True, tol_style="basic", fit_class=None, upper_tol=0.0, lower_tol=0.0
    )
    assert Drawing.model_validate(valid_drawing).characteristics[0].is_basic


# --- R6: bboxes lie inside the image and have extent --------------------------------


def test_bbox_outside_image_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["bbox"] = [3400.0, 400.0, 3600.0, 430.0]
    _expect_reject(valid_drawing, "falls outside")


def test_negative_bbox_origin_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["bbox"] = [-5.0, 400.0, 420.0, 430.0]
    _expect_reject(valid_drawing, "falls outside")


def test_inverted_bbox_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["bbox"] = [420.0, 400.0, 300.0, 430.0]
    _expect_reject(valid_drawing, "non-positive extent")


def test_leader_target_bbox_outside_image_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["leader_target_bbox"] = [
        0.0,
        0.0,
        10.0,
        3000.0,
    ]
    _expect_reject(valid_drawing, "falls outside")


def test_datum_bbox_outside_image_rejected(valid_drawing):
    valid_drawing["datums"][0]["bbox"] = [0.0, 2400.0, 100.0, 2600.0]
    _expect_reject(valid_drawing, "falls outside")


def test_bboxes_unchecked_when_image_size_absent(valid_drawing):
    # A hand-labelled real drawing may arrive before the image is registered.
    valid_drawing["image_size"] = None
    valid_drawing["characteristics"][0]["bbox"] = [9e5, 9e5, 9e5 + 10, 9e5 + 10]
    assert Drawing.model_validate(valid_drawing)


# --- kind / field coherence ---------------------------------------------------------


def test_dimension_without_dim_type_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["dim_type"] = None
    _expect_reject(valid_drawing, "requires dim_type")


def test_dimension_without_nominal_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["nominal"] = None
    _expect_reject(valid_drawing, "requires nominal")


def test_geometric_tolerance_without_value_rejected(valid_drawing):
    valid_drawing["characteristics"][1]["gtol_value"] = None
    _expect_reject(valid_drawing, "requires gtol_value")


def test_zero_gtol_value_rejected(valid_drawing):
    valid_drawing["characteristics"][1]["gtol_value"] = 0.0
    _expect_reject(valid_drawing, "greater than 0")


def test_unknown_field_rejected(valid_drawing):
    valid_drawing["characteristics"][0]["confidence"] = 0.9
    _expect_reject(valid_drawing, "Extra inputs are not permitted")


def test_duplicate_datum_labels_rejected(valid_drawing):
    valid_drawing["datums"][1]["label"] = "A"
    _expect_reject(valid_drawing, "duplicate datum labels")


def test_empty_characteristics_rejected(valid_drawing):
    valid_drawing["characteristics"] = []
    _expect_reject(valid_drawing, "at least 1 item")


# --- the two documents must not drift apart -----------------------------------------


def test_pydantic_and_json_schema_agree_on_gtol_symbols():
    js = load_json_schema()
    from typing import get_args

    from balloonbench.schema import GtolSymbol

    json_symbols = set(
        js["$defs"]["characteristic"]["properties"]["gtol_symbol"]["enum"]
    ) - {None}
    assert json_symbols == set(get_args(GtolSymbol))


def test_pydantic_and_json_schema_agree_on_kinds():
    js = load_json_schema()
    from typing import get_args

    from balloonbench.schema import Kind

    assert set(js["$defs"]["characteristic"]["properties"]["kind"]["enum"]) == set(
        get_args(Kind)
    )
