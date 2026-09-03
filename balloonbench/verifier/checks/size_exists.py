"""Does the size this dimension states actually exist on the solid?

SPEC.md section 12.2's first check, and the one that catches the most common extraction
error there is: a misread digit. ⌀22 where the part has ⌀20 is invisible to every metric
that does not know what the part looks like, and obvious the moment you ask the geometry.

The logic is deliberately four-way rather than two-way, and the two middle cases are what
keep the false-positive rate down:

* exactly one candidate inside the tight band -> **verified**;
* several distinct candidates inside it -> **unverifiable**, because a symmetric part
  genuinely has the same distance in several places and the drawing does not say which one
  it means. Reporting that honestly is right; picking one and calling it verified is not;
* nothing inside the tight band but something inside a generous one -> **unverifiable**. A
  near miss is where a real tolerance interpretation, a fillet, or an off-by-a-hair model
  lives, and contradicting there is how a verifier earns a reputation for crying wolf;
* nothing even close -> **contradicted**, with the nearest real values attached so the
  reader can see what the part does have, and a suggested correction when one value stands
  out.

The tight band is the dimension's own tolerance plus a small epsilon, because a dimension
that says 44 ±0.05 is a claim about a 0.1 wide interval and not about a point.
"""

from __future__ import annotations

from balloonbench.verifier.base import CheckContext
from balloonbench.verifier.brep_index import Candidate

__all__ = ["NAME", "run"]

NAME = "size_exists"

#: Added to the stated tolerance before anything is called a match. A model is exact and a
#: drawing is rounded, so a nominal written to two decimals can be a hundredth away from the
#: solid without anything being wrong.
EPSILON = 0.02

#: A contradiction is only reported when nothing sits within this multiple of the tight band
#: (or this many millimetres, whichever is larger). Generous on purpose.
GENEROUS = 8.0
GENEROUS_FLOOR = 0.5

#: Dimension types this check can answer for. An angle is not a length and a thread's
#: nominal is a designation rather than a measurable diameter, so both are left alone
#: instead of being guessed at.
CHECKABLE = {"diameter", "radius", "linear"}


def run(context: CheckContext) -> None:
    for c in context.drawing.characteristics:
        if c.kind != "dimension" or c.nominal is None:
            continue
        if c.dim_type not in CHECKABLE:
            context.unverifiable(
                c, NAME, f"{c.dim_type} dimensions are not checked against geometry"
            )
            continue
        if c.is_reference:
            # A reference dimension repeats something dimensioned elsewhere; it is not an
            # independent claim about the part.
            context.unverifiable(c, NAME, "reference dimension, not an independent claim")
            continue

        target = context.mm(c.nominal)
        if target is None:
            continue
        tight = _band(context, c)
        generous = max(tight * GENEROUS, GENEROUS_FLOOR)

        if c.dim_type == "diameter":
            candidates = context.index.diameter_candidates(target, generous)
        elif c.dim_type == "radius":
            candidates = context.index.radius_candidates(target, generous)
        else:
            candidates = context.index.length_candidates(target, generous)

        close = [candidate for candidate in candidates if candidate.error(target) <= tight]

        if len(close) == 1:
            context.verify(
                c, NAME,
                f"matches {close[0].detail} (error {close[0].error(target):.3f} mm)",
                confidence=_confidence(close[0], target, tight),
            )
        elif len(close) > 1:
            values = ", ".join(f"{candidate.value:.3f}" for candidate in close[:4])
            context.unverifiable(
                c, NAME,
                f"{len(close)} distinct features match within {tight:.3f} mm ({values}); "
                f"the drawing does not say which is meant",
            )
        elif candidates:
            nearest = min(candidates, key=lambda candidate: candidate.error(target))
            context.unverifiable(
                c, NAME,
                f"nearest feature is {nearest.value:.3f} mm, "
                f"{nearest.error(target):.3f} mm away -- outside the stated tolerance but "
                f"too close to contradict",
            )
        elif c.dim_type == "diameter" and context.index.tap_drill_for(target):
            # A thread's stated diameter is the size of the thread, not of the hole in the
            # solid: the model carries the drilled hole the tap cuts into. Contradicting
            # here would fire on every tapped hole on every drawing.
            drill = context.index.tap_drill_for(target)[0]
            context.unverifiable(
                c, NAME,
                f"no ⌀{target:.3f} face, but the solid has {drill.detail}; consistent with "
                f"a threaded hole modelled at its drill diameter",
            )
        else:
            nearby = context.index.nearest_values(target, c.dim_type or "linear")
            listed = ", ".join(f"{value:.3f}" for value in nearby) or "none"
            context.contradict(
                c, NAME,
                f"no {c.dim_type} of {target:.3f} mm within {generous:.3f} mm. "
                f"Nearest on the solid: {listed}",
                confidence=_contradiction_confidence(target, nearby),
                correction=_correction(c, context, nearby),
            )


def _band(context: CheckContext, c) -> float:
    """How far a real feature may sit from the stated nominal and still be it."""
    upper = context.mm(c.upper_tol) or 0.0
    lower = context.mm(c.lower_tol) or 0.0
    return max(abs(upper), abs(lower)) + EPSILON


def _confidence(candidate: Candidate, target: float, band: float) -> float:
    """Higher when the match is dead on and the band is tight."""
    return round(min(1.0, 0.75 + 0.25 * (1.0 - candidate.error(target) / max(band, 1e-9))), 3)


def _contradiction_confidence(target: float, nearby: tuple[float, ...]) -> float:
    """Lower when the part has nothing comparable at all.

    A part with several similar sizes and none matching is strong evidence of a misread
    digit. A part with nothing remotely similar is more likely to mean the check is looking
    in the wrong place -- a dimension of a feature this index does not model -- so the
    verdict stands but says less.
    """
    if not nearby:
        return 0.6
    closest = min(abs(value - target) for value in nearby)
    return 0.95 if closest < 0.25 * max(target, 1e-9) else 0.75


def _correction(c, context: CheckContext, nearby: tuple[float, ...]) -> dict | None:
    """Propose the nearest real value, but only when one clearly stands out.

    A suggestion that is merely the closest of several equally plausible numbers is a guess
    wearing a suggestion's clothes, so two near-equal candidates produce no suggestion.
    """
    if not nearby:
        return None
    target = context.mm(c.nominal) or 0.0
    ordered = sorted(nearby, key=lambda value: abs(value - target))
    best = ordered[0]
    if len(ordered) > 1:
        runner_up = ordered[1]
        if abs(runner_up - target) < 1.6 * abs(best - target):
            return None
    scale = 25.4 if context.drawing.units == "inch" else 1.0
    return {"nominal": round(best / scale, 4)}
