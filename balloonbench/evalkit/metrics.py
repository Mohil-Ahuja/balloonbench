"""The four metric tiers of SPEC.md section 10.2.

The tiers exist because a single accuracy number cannot distinguish failures that mean
entirely different things to the person using the output. They are ordered by how much they
presuppose:

* **Tier 1, detection.** Did the model find the callout at all? Precision, recall and F1,
  broken out by kind and by geometric symbol.
* **Tier 2, correctness given a match.** Having found it, did it read it right? Values,
  tolerances, symbols, modifiers and datum sequences, each scored only over the pairs where
  the question applies -- ``modifier_accuracy`` over dimensions would be meaningless, and
  padding it with automatic passes would flatter every model equally.
* **Tier 3, structure.** Is the drawing wired together correctly? A model can score
  perfectly on tiers 1 and 2 and still attach every tolerance to the wrong reference frame,
  which is invisible per-characteristic and catastrophic in practice.
* **Tier 4, cost.** What would the errors have cost? Recall restricted to critical
  characteristics, and a weighted error total whose weights live in ``error_costs.json``
  with a written justification for each one.

Rates are reported as ``(numerator, denominator)`` counts as well as a fraction, because a
tier-2 rate computed over three matched pairs is not the same claim as one computed over
three hundred, and an aggregate over drawings has to sum the counts rather than average the
fractions -- averaging would give a sheet with two callouts the same weight as one with
twenty.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from balloonbench.evalkit.matching import MatchConfig, MatchResult, match, to_mm
from balloonbench.evalkit.prediction import Prediction
from balloonbench.schema import Characteristic, Drawing

__all__ = [
    "COSTS_PATH",
    "DrawingScore",
    "Rate",
    "aggregate",
    "datum_graph_distance",
    "error_costs",
    "evaluate_drawing",
    "weighted_error_cost",
]

COSTS_PATH = Path(__file__).resolve().parent / "error_costs.json"


@lru_cache(maxsize=4)
def error_costs(path: str | None = None) -> dict[str, Any]:
    """The cost weights, with their written justifications, from the config file."""
    return json.loads(Path(path or COSTS_PATH).read_text(encoding="utf-8"))


@dataclass
class Rate:
    """A count-carrying rate. ``value`` is ``None`` when nothing applied."""

    hits: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.hits += int(ok)

    @property
    def value(self) -> float | None:
        return None if self.total == 0 else self.hits / self.total

    def merged(self, other: Rate) -> Rate:
        return Rate(self.hits + other.hits, self.total + other.total)

    def as_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "total": self.total, "value": self.value}


# --- tier 2 helpers ---------------------------------------------------------------------

_NUMBER = re.compile(r"\d+\.(\d+)")


def _decimals(text: str | None) -> int:
    """The most decimal places any number in a callout's text is written to.

    This is what "last significant digit" means on a drawing: ``⌀44.00 ±0.05`` is written to
    two places, so a prediction of 44.01 is out by one unit in the last place shown, while
    on ``⌀44 ±0.5`` the same absolute error is out by ten. Taking the digits from the *text*
    rather than from the float is the point -- 44.00 and 44.0 are the same number and
    different claims about precision.
    """
    if not text:
        return 2
    found = [len(m.group(1)) for m in _NUMBER.finditer(text)]
    return max(found) if found else 0


def _close(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)


def _tolerances_equal(gt: Characteristic, pred: Characteristic, gu: str, pu: str) -> bool:
    return _close(to_mm(gt.upper_tol, gu), to_mm(pred.upper_tol, pu)) and _close(
        to_mm(gt.lower_tol, gu), to_mm(pred.lower_tol, pu)
    )


def _within_one_lsd(gt: Characteristic, pred: Characteristic, gu: str, pu: str) -> bool:
    step = 10.0 ** -_decimals(gt.raw_text)
    limit = to_mm(step, gu) or step
    for attribute in ("upper_tol", "lower_tol"):
        a = to_mm(getattr(gt, attribute), gu)
        b = to_mm(getattr(pred, attribute), pu)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if abs(a - b) > limit + 1e-9:
            return False
    return True


def _datum_refs_equal(gt: Characteristic, pred: Characteristic) -> bool:
    a = [(r.label, r.modifier) for r in gt.datum_refs]
    b = [(r.label, r.modifier) for r in pred.datum_refs]
    return a == b


def _is_perfect(gt: Characteristic, pred: Characteristic, gu: str, pu: str) -> bool:
    """Every scored field agrees. The unit of the full-drawing exact match."""
    if gt.kind != pred.kind or gt.view != pred.view:
        return False
    if gt.kind == "dimension":
        return (
            gt.dim_type == pred.dim_type
            and _close(to_mm(gt.nominal, gu), to_mm(pred.nominal, pu))
            and _tolerances_equal(gt, pred, gu, pu)
            and gt.is_basic == pred.is_basic
            and gt.fit_class == pred.fit_class
        )
    if gt.kind == "geometric_tolerance":
        return (
            gt.gtol_symbol == pred.gtol_symbol
            and _close(to_mm(gt.gtol_value, gu), to_mm(pred.gtol_value, pu))
            and gt.gtol_zone == pred.gtol_zone
            and gt.material_modifier == pred.material_modifier
            and _datum_refs_equal(gt, pred)
        )
    return (gt.raw_text or "") == (pred.raw_text or "")


# --- tier 3 -----------------------------------------------------------------------------


def _graph(
    datums, characteristics, ids
) -> tuple[set[str], set[tuple]]:
    """Nodes and labelled edges of a drawing's reference-frame graph.

    Nodes are datum labels. Edges are ``(characteristic key, position, label, modifier)``
    -- one per datum reference, carrying its position in the sequence, because primary and
    secondary references constrain different degrees of freedom and swapping them is a real
    error that a set-based comparison would score as correct.
    """
    nodes = {d.label for d in datums}
    edges: set[tuple] = set()
    for key, c in zip(ids, characteristics, strict=True):
        if c.kind != "geometric_tolerance":
            continue
        for position, ref in enumerate(c.datum_refs):
            edges.add((key, position, ref.label, ref.modifier))
            nodes.add(ref.label)
    return nodes, edges


def datum_graph_distance(
    gt: Drawing, prediction: Prediction, result: MatchResult
) -> float:
    """Normalised edit distance between the ground-truth and predicted datum graphs.

    General graph edit distance is NP-hard, so this is the restricted form that the problem
    actually calls for: the node correspondence is *given*, because a datum's label is its
    identity -- datum A in the prediction is datum A in the ground truth or it is nothing.
    With the correspondence fixed, the edit distance is just the symmetric difference of the
    node and edge sets, which is exact, cheap, and interpretable as "how many wires are
    wrong".

    Matched characteristics are keyed by their ground-truth id so the two graphs are
    comparable; an unmatched characteristic on either side keys uniquely and so contributes
    its edges as pure insertions or deletions. Normalisation is by the total size of both
    graphs, giving 0 for identical and 1 for wholly disjoint.
    """
    gt_ids = [f"g{c.id}" for c in gt.characteristics]
    pred_keys: list[str] = [f"p{i}" for i in range(len(prediction.characteristics))]
    for m in result.matched:
        pred_keys[m.pred_index] = f"g{gt.characteristics[m.gt_index].id}"

    gt_nodes, gt_edges = _graph(gt.datums, gt.characteristics, gt_ids)
    pred_nodes, pred_edges = _graph(
        prediction.datums, prediction.characteristics, pred_keys
    )

    total = len(gt_nodes) + len(gt_edges) + len(pred_nodes) + len(pred_edges)
    if total == 0:
        return 0.0
    difference = len(gt_nodes ^ pred_nodes) + len(gt_edges ^ pred_edges)
    return difference / total


# --- tier 4 -----------------------------------------------------------------------------


def weighted_error_cost(
    gt: Drawing,
    prediction: Prediction,
    result: MatchResult,
    *,
    costs: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Total cost of this drawing's errors, and the breakdown by error type.

    Each matched pair is charged for every field it got wrong, not only the worst one: a
    callout with the wrong symbol *and* the wrong datum frame is two distinct mistakes for
    the inspector downstream, and charging only the larger would make a model that fails in
    two ways look no worse than one that fails in one.
    """
    table = costs or error_costs()
    weights = {k: v["cost"] for k, v in table["weights"].items()}
    critical = float(table["critical_multiplier"])
    gu, pu = gt.units, prediction.units

    breakdown: dict[str, float] = {}

    def charge(key: str, is_critical: bool) -> None:
        amount = weights[key] * (critical if is_critical else 1.0)
        breakdown[key] = breakdown.get(key, 0.0) + amount

    for index in result.unmatched_gt:
        charge("missed_characteristic", gt.characteristics[index].is_critical)
    for _ in result.unmatched_pred:
        charge("spurious_characteristic", False)
    for _ in prediction.malformed:
        charge("spurious_characteristic", False)
        charge("schema_violation", False)

    for m in result.matched:
        g, p, flag = m.gt, m.pred, m.gt.is_critical
        if g.kind == "dimension":
            if not _close(to_mm(g.nominal, gu), to_mm(p.nominal, pu)):
                charge("wrong_nominal", flag)
            if not _tolerances_equal(g, p, gu, pu):
                charge("wrong_tolerance", flag)
        elif g.kind == "geometric_tolerance":
            if g.gtol_symbol != p.gtol_symbol:
                charge("wrong_symbol", flag)
            if not _close(to_mm(g.gtol_value, gu), to_mm(p.gtol_value, pu)):
                charge("wrong_tolerance", flag)
            if g.material_modifier != p.material_modifier:
                charge("wrong_modifier", flag)
            if not _datum_refs_equal(g, p):
                charge("wrong_datum_refs", flag)
        elif (g.raw_text or "") != (p.raw_text or ""):
            charge("wrong_nominal", flag)

    return sum(breakdown.values()), breakdown


# --- the score --------------------------------------------------------------------------


@dataclass
class DrawingScore:
    """Every tier, for one drawing. Counts, so that drawings can be summed not averaged."""

    drawing_id: str
    profile: str | None = None
    family: str | None = None
    n_gt: int = 0
    n_pred: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    malformed: int = 0

    by_kind: dict[str, Rate] = field(default_factory=dict)
    by_symbol: dict[str, Rate] = field(default_factory=dict)

    tier2: dict[str, Rate] = field(default_factory=dict)

    graph_distance: float = 0.0
    exact_drawing: bool = False

    ctq: Rate = field(default_factory=Rate)
    cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return None if denominator == 0 else self.true_positives / denominator

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "drawing_id": self.drawing_id,
            "profile": self.profile,
            "family": self.family,
            "detection": {
                "n_gt": self.n_gt,
                "n_pred": self.n_pred,
                "tp": self.true_positives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
                "malformed": self.malformed,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "by_kind": {k: v.as_dict() for k, v in sorted(self.by_kind.items())},
                "by_symbol": {k: v.as_dict() for k, v in sorted(self.by_symbol.items())},
            },
            "correctness": {k: v.as_dict() for k, v in sorted(self.tier2.items())},
            "structural": {
                "datum_graph_distance": self.graph_distance,
                "exact_drawing": self.exact_drawing,
            },
            "cost": {
                "ctq_recall": self.ctq.as_dict(),
                "total": self.cost,
                "breakdown": dict(sorted(self.cost_breakdown.items())),
            },
        }


#: Tier-2 questions, and which characteristics each one applies to. Keeping the applicability
#: rule beside the metric is what stops a rate from being computed over pairs where the
#: question is meaningless -- the single easiest way to publish a flattering number.
_TIER2_APPLIES = {
    "nominal_exact": lambda c: c.kind == "dimension",
    "tolerance_exact": lambda c: c.kind == "dimension",
    "tolerance_within_1lsd": lambda c: c.kind == "dimension",
    "symbol_accuracy": lambda c: c.kind == "geometric_tolerance",
    "modifier_accuracy": lambda c: (
        c.kind == "geometric_tolerance" and c.gtol_symbol not in {None}
    ),
    "datum_ref_exact": lambda c: c.kind == "geometric_tolerance",
}


def evaluate_drawing(
    gt: Drawing,
    prediction: Prediction,
    config: MatchConfig | None = None,
    *,
    family: str | None = None,
) -> DrawingScore:
    """Match, then score every tier.

    >>> from balloonbench.evalkit.prediction import Prediction
    >>> # A perfect prediction is the fixed point: nothing missed, nothing invented.
    >>> # (Exercised properly in tests/test_evalkit.py against a generated drawing.)
    """
    result = match(
        gt.characteristics,
        prediction.characteristics,
        config,
        gt_units=gt.units,
        pred_units=prediction.units,
    )
    gu, pu = gt.units, prediction.units

    score = DrawingScore(
        drawing_id=gt.drawing_id,
        profile=gt.provenance.degradation_profile,
        family=family,
        n_gt=len(gt.characteristics),
        n_pred=prediction.n_predicted,
        true_positives=result.n_matched,
        false_positives=len(result.unmatched_pred) + len(prediction.malformed),
        false_negatives=len(result.unmatched_gt),
        malformed=len(prediction.malformed),
    )

    # Tier 1 breakouts. Recall by group: of the ground-truth callouts of this kind or
    # symbol, how many were found? Precision cannot be broken out the same way without
    # deciding which group a wrong prediction belongs to, and that decision would be the
    # metric rather than the model.
    found = {m.gt_index for m in result.matched}
    for index, c in enumerate(gt.characteristics):
        score.by_kind.setdefault(c.kind, Rate()).add(index in found)
        if c.gtol_symbol:
            score.by_symbol.setdefault(c.gtol_symbol, Rate()).add(index in found)
        if c.is_critical:
            score.ctq.add(index in found)

    for name in _TIER2_APPLIES:
        score.tier2[name] = Rate()

    for m in result.matched:
        g, p = m.gt, m.pred
        if _TIER2_APPLIES["nominal_exact"](g):
            score.tier2["nominal_exact"].add(
                _close(to_mm(g.nominal, gu), to_mm(p.nominal, pu))
            )
            score.tier2["tolerance_exact"].add(_tolerances_equal(g, p, gu, pu))
            score.tier2["tolerance_within_1lsd"].add(_within_one_lsd(g, p, gu, pu))
        if _TIER2_APPLIES["symbol_accuracy"](g):
            score.tier2["symbol_accuracy"].add(g.gtol_symbol == p.gtol_symbol)
            score.tier2["modifier_accuracy"].add(g.material_modifier == p.material_modifier)
            score.tier2["datum_ref_exact"].add(_datum_refs_equal(g, p))

    score.graph_distance = datum_graph_distance(gt, prediction, result)
    score.exact_drawing = (
        not result.unmatched_gt
        and not result.unmatched_pred
        and not prediction.malformed
        and all(_is_perfect(m.gt, m.pred, gu, pu) for m in result.matched)
    )

    score.cost, score.cost_breakdown = weighted_error_cost(gt, prediction, result)
    return score


# --- aggregation ------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Every drawing in a run, summed. Rates come from summed counts, never from averaged
    per-drawing rates, so a two-callout sheet does not outvote a twenty-callout one."""

    n_drawings: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    malformed: int = 0
    by_kind: dict[str, Rate] = field(default_factory=dict)
    by_symbol: dict[str, Rate] = field(default_factory=dict)
    tier2: dict[str, Rate] = field(default_factory=dict)
    ctq: Rate = field(default_factory=Rate)
    exact_drawings: int = 0
    mean_graph_distance: float = 0.0
    cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    by_profile: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def precision(self) -> float | None:
        d = self.true_positives + self.false_positives
        return None if d == 0 else self.true_positives / d

    @property
    def recall(self) -> float | None:
        d = self.true_positives + self.false_negatives
        return None if d == 0 else self.true_positives / d

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def exact_drawing_rate(self) -> float | None:
        return None if self.n_drawings == 0 else self.exact_drawings / self.n_drawings

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_drawings": self.n_drawings,
            "detection": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
                "malformed": self.malformed,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "by_kind": {k: v.as_dict() for k, v in sorted(self.by_kind.items())},
                "by_symbol": {k: v.as_dict() for k, v in sorted(self.by_symbol.items())},
            },
            "correctness": {k: v.as_dict() for k, v in sorted(self.tier2.items())},
            "structural": {
                "mean_datum_graph_distance": self.mean_graph_distance,
                "exact_drawings": self.exact_drawings,
                "exact_drawing_rate": self.exact_drawing_rate,
            },
            "cost": {
                "ctq_recall": self.ctq.as_dict(),
                "total": self.cost,
                "breakdown": dict(sorted(self.cost_breakdown.items())),
            },
            "by_profile": self.by_profile,
        }


def aggregate(scores: list[DrawingScore]) -> RunSummary:
    """Sum a run. Per-profile slices come free, and they are the point: SPEC.md section 8
    exists so a result can be reported per degradation condition rather than averaged into
    a number that hides where the model actually breaks."""
    summary = RunSummary(n_drawings=len(scores))
    distances: list[float] = []

    for score in scores:
        summary.true_positives += score.true_positives
        summary.false_positives += score.false_positives
        summary.false_negatives += score.false_negatives
        summary.malformed += score.malformed
        summary.exact_drawings += int(score.exact_drawing)
        summary.cost += score.cost
        distances.append(score.graph_distance)
        summary.ctq = summary.ctq.merged(score.ctq)
        for group, rates in (("by_kind", score.by_kind), ("by_symbol", score.by_symbol)):
            target = getattr(summary, group)
            for key, rate in rates.items():
                target[key] = target.get(key, Rate()).merged(rate)
        for key, rate in score.tier2.items():
            summary.tier2[key] = summary.tier2.get(key, Rate()).merged(rate)
        for key, amount in score.cost_breakdown.items():
            summary.cost_breakdown[key] = summary.cost_breakdown.get(key, 0.0) + amount

    summary.mean_graph_distance = sum(distances) / len(distances) if distances else 0.0

    profiles = sorted({s.profile or "unknown" for s in scores})
    for profile in profiles:
        subset = [s for s in scores if (s.profile or "unknown") == profile]
        if len(subset) == len(scores) and len(profiles) == 1:
            # A single-condition run: the slice would repeat the whole summary.
            break
        inner = aggregate(subset)
        summary.by_profile[profile] = inner.as_dict()

    return summary
