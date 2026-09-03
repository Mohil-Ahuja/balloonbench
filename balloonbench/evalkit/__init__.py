"""Matching and metrics: how a prediction is compared with ground truth (SPEC.md section 10)."""

from balloonbench.evalkit.matching import (
    MatchConfig,
    MatchResult,
    match,
    no_bbox_config,
)
from balloonbench.evalkit.metrics import (
    DrawingScore,
    RunSummary,
    aggregate,
    error_costs,
    evaluate_drawing,
)
from balloonbench.evalkit.prediction import Prediction, load_prediction, parse_prediction
from balloonbench.evalkit.report import markdown_table, write_report

__all__ = [
    "DrawingScore",
    "MatchConfig",
    "MatchResult",
    "Prediction",
    "RunSummary",
    "aggregate",
    "error_costs",
    "evaluate_drawing",
    "load_prediction",
    "markdown_table",
    "match",
    "no_bbox_config",
    "parse_prediction",
    "write_report",
]
