"""Matching predicted characteristics to ground truth before anything can be scored.

A prediction carries no balloon numbers that mean the same thing as the ground truth's, so
every metric downstream rests on a correspondence that has to be established first. SPEC.md
section 10.1 sets the shape: a cost matrix combining a geometric term and a semantic one,
solved as an assignment problem, with matches above a threshold rejected so they become a
false positive and a false negative rather than a bad pair.

Two decisions here shape every number the harness reports.

**The assignment is global, not greedy.** ``scipy.optimize.linear_sum_assignment`` minimises
total cost over the whole sheet. Greedy nearest-match would let one confident prediction
claim a callout that a second prediction fits far better, and the damage compounds: the
displaced prediction then takes a third callout, and a single local error becomes a chain of
them. On a drawing with four ⌀12 holes at four corners this is the difference between "all
four found" and "all four wrong".

**A kind mismatch is fatal, not expensive.** A predicted flatness can never match a
dimension however well their boxes overlap. Making it merely costly would let a high-IoU
pair sneak under the threshold and be scored as a correct detection with wrong content,
which flatters the detection tier by borrowing from the correctness tier.

The ``--no-bbox`` mode of SPEC.md section 10.1 is here as :func:`no_bbox_config`. It is a
genuinely easier task -- there is no localisation to get wrong, and a prediction may be
matched to a callout on the other side of the sheet -- so the two are reported separately
and never averaged together.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment

from balloonbench.schema import Characteristic

__all__ = [
    "MatchConfig",
    "MatchResult",
    "Matched",
    "bbox_iou",
    "cost_matrix",
    "match",
    "no_bbox_config",
    "semantic_distance",
    "to_mm",
]

#: Millimetres per unit, for the unit normalisation SPEC.md section 10.2 requires before any
#: value comparison. A house style may render an inch sheet; a model may answer in either.
_MM_PER_UNIT = {"mm": 1.0, "inch": 25.4}


def to_mm(value: float | None, units: str) -> float | None:
    """A length in the drawing's units, as millimetres.

    Comparisons happen in one unit or they do not happen at all: 0.5 inch and 12.7 mm are
    the same tolerance, and a harness that scored them as different would be measuring the
    house style rather than the model.
    """
    if value is None:
        return None
    try:
        return value * _MM_PER_UNIT[units]
    except KeyError as exc:
        raise ValueError(f"unknown units {units!r}; known: {sorted(_MM_PER_UNIT)}") from exc


@dataclass(frozen=True)
class MatchConfig:
    """Weights and thresholds for the assignment.

    ``max_cost`` is the one to understand. A pair costing more than this is not a match at
    all: the ground-truth callout is counted missed and the prediction spurious. Set it too
    high and a model gets credit for finding something it described wrongly; too low and a
    correct extraction with a sloppy box is punished twice, once as a miss and once as a
    false alarm. The default admits a pair whose box overlaps well but whose content is
    wholly wrong, because "found it, read it wrong" and "did not find it" are different
    failures and the tiers exist to tell them apart.
    """

    w_geo: float = 0.5
    w_sem: float = 0.5
    max_cost: float = 0.6
    #: Below this IoU the boxes are treated as disjoint even in bbox mode, so a prediction
    #: cannot be dragged onto a distant callout by a strong semantic score alone.
    min_iou: float = 0.05

    @property
    def use_bbox(self) -> bool:
        return self.w_geo > 0.0

    def __post_init__(self) -> None:
        if self.w_geo < 0 or self.w_sem <= 0:
            raise ValueError("w_geo must be non-negative and w_sem positive")
        # A kind mismatch and a pair of disjoint boxes both cost exactly 1.0, and both must
        # be unmatchable. A threshold at or above 1.0 would admit them.
        if not 0.0 < self.max_cost < 1.0:
            raise ValueError(f"max_cost must lie in (0, 1), got {self.max_cost}")


def no_bbox_config(max_cost: float = 0.5) -> MatchConfig:
    """Semantic-only matching, for models that cannot produce reliable coordinates.

    The threshold is tighter than in bbox mode on purpose. With no geometry to separate
    them, two callouts that read alike -- ⌀12 and ⌀12.5 on different features -- are
    distinguishable only by content, so the content has to be held to a higher standard.
    """
    return MatchConfig(w_geo=0.0, w_sem=1.0, max_cost=max_cost)


# --- geometry -------------------------------------------------------------------------


def bbox_iou(a, b) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return 0.0 if union <= 0 else intersection / union


# --- semantics ------------------------------------------------------------------------


def _relative(a: float | None, b: float | None) -> float:
    """Relative difference between two values, in ``[0, 1]``.

    Relative rather than absolute because a drawing carries a 0.02 flatness zone and a 315
    bolt-circle diameter on the same sheet. An absolute distance would make every tolerance
    look identical to every other and every diameter look wildly different, so matching
    would be decided entirely by the largest numbers on the page.
    """
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    scale = max(abs(a), abs(b), 1e-6)
    return min(1.0, abs(a - b) / scale)


def _datum_sequence_distance(gt: Characteristic, pred: Characteristic) -> float:
    """How far apart two ordered datum reference sequences are, in ``[0, 1]``.

    Order carries meaning -- ``|A|B|`` and ``|B|A|`` are different reference frames that
    constrain different degrees of freedom -- so the comparison is positional rather than
    set-based, and a modifier difference counts as half a mismatch.
    """
    a, b = gt.datum_refs, pred.datum_refs
    if not a and not b:
        return 0.0
    span = max(len(a), len(b))
    penalty = 0.0
    for i in range(span):
        ref_a = a[i] if i < len(a) else None
        ref_b = b[i] if i < len(b) else None
        if ref_a is None or ref_b is None or ref_a.label != ref_b.label:
            penalty += 1.0
        elif ref_a.modifier != ref_b.modifier:
            penalty += 0.5
    return penalty / span


def _text_distance(a: str | None, b: str | None) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def semantic_distance(
    gt: Characteristic,
    pred: Characteristic,
    *,
    gt_units: str = "mm",
    pred_units: str = "mm",
) -> float:
    """How different two callouts say they are, in ``[0, 1]``.

    Returns exactly ``1.0`` on a kind mismatch, which with any sane threshold makes the pair
    unmatchable. The per-kind weights below are the harness's opinion about what identifies
    a callout: for a dimension the number dominates, for a geometric tolerance the symbol
    does, because two position tolerances of different values are the same *kind of thing*
    in a way that a position and a perpendicularity are not.
    """
    if gt.kind != pred.kind:
        return 1.0

    view = 0.0 if gt.view == pred.view else 1.0

    if gt.kind == "dimension":
        value = _relative(to_mm(gt.nominal, gt_units), to_mm(pred.nominal, pred_units))
        dim_type = 0.0 if gt.dim_type == pred.dim_type else 1.0
        return 0.60 * value + 0.25 * dim_type + 0.15 * view

    if gt.kind == "geometric_tolerance":
        symbol = 0.0 if gt.gtol_symbol == pred.gtol_symbol else 1.0
        value = _relative(to_mm(gt.gtol_value, gt_units), to_mm(pred.gtol_value, pred_units))
        datums = _datum_sequence_distance(gt, pred)
        return 0.50 * symbol + 0.20 * value + 0.20 * datums + 0.10 * view

    # note, surface_finish, thread: the text is the content.
    return 0.85 * _text_distance(gt.raw_text, pred.raw_text) + 0.15 * view


# --- assignment -----------------------------------------------------------------------


def cost_matrix(
    gt: list[Characteristic],
    pred: list[Characteristic],
    config: MatchConfig,
    *,
    gt_units: str = "mm",
    pred_units: str = "mm",
) -> np.ndarray:
    """The ``(len(gt), len(pred))`` cost matrix the assignment minimises over."""
    costs = np.zeros((len(gt), len(pred)), dtype=float)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if g.kind != p.kind:
                # Fatal, not expensive. A perfectly-placed box must not buy a prediction a
                # match to a callout of a different kind: that would score as a correct
                # detection with wrong content, flattering tier 1 at tier 2's expense.
                costs[i, j] = 1.0
                continue
            sem = semantic_distance(g, p, gt_units=gt_units, pred_units=pred_units)
            if not config.use_bbox:
                costs[i, j] = config.w_sem * sem
                continue
            iou = bbox_iou(g.bbox, p.bbox)
            if iou < config.min_iou:
                # Disjoint boxes. Cost 1.0 rather than infinity: the solver still needs a
                # finite matrix, and 1.0 is above any threshold worth using.
                costs[i, j] = 1.0
            else:
                costs[i, j] = config.w_geo * (1.0 - iou) + config.w_sem * sem
    return costs


@dataclass(frozen=True)
class Matched:
    """One accepted correspondence."""

    gt_index: int
    pred_index: int
    cost: float
    iou: float
    semantic: float
    gt: Characteristic
    pred: Characteristic


@dataclass(frozen=True)
class MatchResult:
    matched: tuple[Matched, ...]
    #: Ground-truth indices with no accepted prediction: the misses.
    unmatched_gt: tuple[int, ...]
    #: Prediction indices with no accepted ground truth: the false alarms.
    unmatched_pred: tuple[int, ...]
    config: MatchConfig

    @property
    def n_matched(self) -> int:
        return len(self.matched)


def match(
    gt: list[Characteristic],
    pred: list[Characteristic],
    config: MatchConfig | None = None,
    *,
    gt_units: str = "mm",
    pred_units: str = "mm",
) -> MatchResult:
    """Assign predictions to ground truth and reject the pairs that cost too much.

    >>> from balloonbench.schema import Characteristic
    >>> box = [10.0, 10.0, 60.0, 30.0]
    >>> a = Characteristic(id=1, kind="dimension", view="front", bbox=box,
    ...                    dim_type="diameter", nominal=44.0)
    >>> b = Characteristic(id=1, kind="dimension", view="front", bbox=box,
    ...                    dim_type="diameter", nominal=44.0)
    >>> result = match([a], [b])
    >>> result.n_matched, result.unmatched_gt, result.unmatched_pred
    (1, (), ())
    """
    config = config or MatchConfig()
    if not gt or not pred:
        return MatchResult(
            matched=(),
            unmatched_gt=tuple(range(len(gt))),
            unmatched_pred=tuple(range(len(pred))),
            config=config,
        )

    costs = cost_matrix(gt, pred, config, gt_units=gt_units, pred_units=pred_units)
    rows, cols = linear_sum_assignment(costs)

    matched: list[Matched] = []
    taken_gt: set[int] = set()
    taken_pred: set[int] = set()
    for i, j in zip(rows, cols, strict=True):
        cost = float(costs[i, j])
        if cost > config.max_cost:
            continue
        matched.append(
            Matched(
                gt_index=int(i),
                pred_index=int(j),
                cost=cost,
                iou=bbox_iou(gt[i].bbox, pred[j].bbox),
                semantic=semantic_distance(
                    gt[i], pred[j], gt_units=gt_units, pred_units=pred_units
                ),
                gt=gt[i],
                pred=pred[j],
            )
        )
        taken_gt.add(int(i))
        taken_pred.add(int(j))

    return MatchResult(
        matched=tuple(matched),
        unmatched_gt=tuple(i for i in range(len(gt)) if i not in taken_gt),
        unmatched_pred=tuple(j for j in range(len(pred)) if j not in taken_pred),
        config=config,
    )


def with_threshold(config: MatchConfig, max_cost: float) -> MatchConfig:
    """A copy of ``config`` at a different rejection threshold, for sensitivity sweeps."""
    return replace(config, max_cost=max_cost)
