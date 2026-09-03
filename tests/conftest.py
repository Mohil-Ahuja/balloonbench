"""Shared fixtures. The valid drawing here is the reference instance every schema test
mutates: each failure test starts from something that passes, changes exactly one thing,
and asserts it now fails. That keeps the tests honest about *which* rule fired.
"""

from __future__ import annotations

import copy

import pytest

VALID_DRAWING: dict = {
    "drawing_id": "syn_flange_00417",
    "source": "synthetic",
    "part_ref": "parts/syn_flange_00417.step",
    "units": "mm",
    "projection": "first_angle",
    "image_size": [3508, 2480],
    "sheet": {"size": "A3", "scale": "1:1"},
    "title_block": {
        "part_number": "FLG-4417-A",
        "revision": "C",
        "material": "EN-GJS-400-15",
        "general_tolerance": "ISO 2768-mK",
        "surface_finish_default": "Ra 3.2",
    },
    "datums": [
        {
            "label": "A",
            "feature_type": "planar_face",
            "view": "front",
            "bbox": [100.0, 100.0, 140.0, 130.0],
            "geometry_ref": "face_12",
        },
        {
            "label": "B",
            "feature_type": "cylindrical_feature",
            "view": "front",
            "bbox": [200.0, 100.0, 240.0, 130.0],
            "geometry_ref": "face_31",
        },
    ],
    "characteristics": [
        {
            "id": 1,
            "kind": "dimension",
            "view": "front",
            "bbox": [300.0, 400.0, 420.0, 430.0],
            "leader_target_bbox": [310.0, 440.0, 380.0, 470.0],
            "dim_type": "diameter",
            "nominal": 44.0,
            "upper_tol": 0.025,
            "lower_tol": 0.0,
            "tol_style": "fit",
            "fit_class": "H7",
            "is_critical": True,
            "raw_text": "Ø44 H7",
        },
        {
            "id": 2,
            "kind": "geometric_tolerance",
            "view": "front",
            "bbox": [500.0, 400.0, 700.0, 430.0],
            "gtol_symbol": "position",
            "gtol_value": 0.05,
            "gtol_zone": "diametral",
            "material_modifier": "MMC",
            "datum_refs": [
                {"label": "A", "modifier": None},
                {"label": "B", "modifier": "MMC"},
            ],
            "raw_text": "position Ø0.05 (M) A B (M)",
        },
        {
            "id": 3,
            "kind": "geometric_tolerance",
            "view": "front",
            "bbox": [500.0, 500.0, 640.0, 530.0],
            "gtol_symbol": "flatness",
            "gtol_value": 0.02,
            "gtol_zone": "linear",
            "raw_text": "flatness 0.02",
        },
    ],
    "provenance": {
        "generator_version": "0.1.0",
        "generator_seed": 918273,
        "degradation_profile": "clean",
        "house_style": "house_a",
        "labeler": None,
    },
}


@pytest.fixture
def valid_drawing() -> dict:
    """A deep copy, so a test that mutates it cannot poison its neighbours."""
    return copy.deepcopy(VALID_DRAWING)
