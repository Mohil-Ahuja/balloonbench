"""Do the dimensions on this drawing agree with each other?

SPEC.md section 12.2 asks for three things: chains that sum to the overall dimension,
tolerances that accumulate consistently, and loops whose arithmetic does not close. All
three are properties of the drawing alone -- no solid is consulted -- which makes this the
check that finds errors a perfect extraction would faithfully reproduce.

The hard part is that the schema records no direction. A dimension has a value and a box,
not an axis, so "chained along an axis" has to be recovered. It is recovered from the
arithmetic instead: a set of dimensions whose values sum to another dimension's value, to
within a whisker, is a chain -- or a coincidence. Which is why this check is bounded hard:

* only chains of two or three parts, because the number of subsets grows exponentially and
  a five-way coincidence on a busy sheet is more likely than a five-way chain;
* only exact sums, within a tenth of the tightest tolerance involved. A chain that closes
  to within a few hundredths is not a chain, it is arithmetic noise;
* only dimensions in the same view, since a chain is drawn along one line of the part;
* and every finding is a **drawing defect**, never a contradiction of an extraction. The
  arithmetic cannot tell which of the four numbers is the wrong one.

What it catches is worth the restraint. An over-dimensioned chain -- three steps plus the
overall length, all toleranced -- is the most common drawing error there is, and it is
unbuildable: two of the four constraints will fight each other on every part made.
"""

from __future__ import annotations

from itertools import combinations

from balloonbench.schema import Characteristic
from balloonbench.verifier.base import CheckContext

__all__ = ["NAME", "run"]

NAME = "tolerance_stack"

#: How exactly a chain must close, as a fraction of the tightest tolerance in it. A chain
#: that closes only approximately is not a chain.
CLOSURE_RATIO = 0.1

#: Floor for that closure test, in millimetres, for chains where nothing carries a tolerance.
CLOSURE_FLOOR = 0.01

#: Longest chain considered. Three parts plus an overall is already four dimensions agreeing
#: to a hundredth of a millimetre, which no coincidence survives often.
MAX_PARTS = 3

#: A chain of tiny dimensions inside a large one is usually a coincidence rather than a
#: chain, so a part must be at least this fraction of the total to count.
MIN_SHARE = 0.05


def run(context: CheckContext) -> None:
    by_view: dict[str, list[Characteristic]] = {}
    for c in context.drawing.characteristics:
        if c.kind != "dimension" or c.dim_type != "linear" or c.nominal is None:
            continue
        if c.is_reference:
            # A reference dimension is explicitly the redundant one: writing it in
            # parentheses is how a draughtsman says "this repeats, do not inspect it".
            continue
        by_view.setdefault(c.view, []).append(c)

    for view, dimensions in by_view.items():
        _check_view(context, view, dimensions)


def _check_view(context: CheckContext, view: str, dimensions: list[Characteristic]) -> None:
    values = {c.id: context.mm(c.nominal) or 0.0 for c in dimensions}
    tolerances = {c.id: _span(context, c) for c in dimensions}
    reported: set[tuple[int, ...]] = set()

    for total in dimensions:
        target = values[total.id]
        if target <= 0:
            continue
        others = [c for c in dimensions if c.id != total.id and values[c.id] > 0]
        for size in range(2, MAX_PARTS + 1):
            for parts in combinations(others, size):
                if any(values[p.id] < MIN_SHARE * target for p in parts):
                    continue
                chain = sum(values[p.id] for p in parts)
                tightest = min(
                    [tolerances[p.id] for p in parts] + [tolerances[total.id]], default=0.0
                )
                closure = max(tightest * CLOSURE_RATIO, CLOSURE_FLOOR)
                if abs(chain - target) > closure:
                    continue

                key = tuple(sorted([total.id, *(p.id for p in parts)]))
                if key in reported:
                    continue
                reported.add(key)

                stacked = sum(tolerances[p.id] for p in parts)
                overall = tolerances[total.id]
                chain_ids = ", ".join(str(p.id) for p in parts)

                if overall > 0 and stacked > 0:
                    context.defect(
                        "overdimensioned_chain",
                        f"in view {view}, characteristics {chain_ids} sum to "
                        f"{chain:.3f} mm, which characteristic {total.id} also states "
                        f"({target:.3f} mm). Both the chain and the overall carry "
                        f"tolerances, so the same distance is constrained twice and the "
                        f"two will disagree on any real part",
                        total.id,
                    )
                elif stacked > overall + 1e-9:
                    context.defect(
                        "tolerance_accumulation",
                        f"in view {view}, the tolerances of characteristics {chain_ids} "
                        f"accumulate to ±{stacked / 2:.3f} mm across a distance that "
                        f"characteristic {total.id} holds to ±{overall / 2:.3f} mm; the "
                        f"chain cannot hold the overall dimension it sums to",
                        total.id,
                    )


def _span(context: CheckContext, c: Characteristic) -> float:
    upper = context.mm(c.upper_tol)
    lower = context.mm(c.lower_tol)
    if upper is None or lower is None:
        # Governed by the title block's general tolerance. Treated as zero here rather than
        # guessed at: reading a general note like "ISO 2768-mK" into a number per dimension
        # is a job for a table this check does not own.
        return 0.0
    return abs(upper - lower)
