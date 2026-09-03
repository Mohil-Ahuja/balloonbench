"""Injecting known errors, so the verifier can be measured rather than believed.

SPEC.md section 12.4 is blunt about it: *do not just build it -- measure it.* A verifier is
a classifier, and a classifier with no measured error rates is a claim. So this module
damages ground truth in known, counted ways, and the test suite reports what fraction of the
damage the verifier catches and how often it complains about undamaged data.

The injections are the mistakes extraction actually makes:

* **perturb_nominal** -- a misread digit. The most common error and the one ``size_exists``
  exists for.
* **swap_datum** -- a reference frame wired to the wrong letter. Invisible per
  characteristic and catastrophic in practice.
* **flip_modifier** -- an MMC dropped or invented. Where models fail most.
* **drop_characteristic** -- a callout missed entirely.
* **shuffle_datum_refs** -- the frame's references put in the wrong order, which changes
  which degrees of freedom are constrained first.
* **underconstrain_drf** -- a locating frame stripped to its primary datum, which is
  SPEC.md section 12.2's own example of a drawing defect.
* **undeclared_datum** -- a reference to a letter the drawing never establishes.
* **unit_confusion** -- every size recorded in the wrong unit system at once.

Two properties matter for the measurement to mean anything.

**Every injection is seeded and recorded.** An :class:`Injection` says what was changed and
where, so a caught error can be attributed to the right check and an uncaught one can be
looked at.

**An injection may legitimately be undetectable.** Perturbing a nominal to another size the
part happens to have produces a drawing that is *correct about a different feature*, and the
verifier is right not to flag it. Those are reported separately rather than counted as
misses, because scoring them as failures would push the verifier toward exactly the
aggressive contradicting that SPEC.md warns against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from balloonbench.schema import Drawing

__all__ = ["INJECTORS", "Injection", "inject", "injector_names"]


@dataclass(frozen=True)
class Injection:
    """One deliberate error, and what it did."""

    kind: str
    characteristic_id: int
    detail: str
    before: Any = None
    after: Any = None
    #: True when the damaged drawing is still a truthful statement about the solid, so a
    #: verifier that stays silent is right rather than wrong.
    benign: bool = False


def _perturb_nominal(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Change one size by enough to matter, the way a misread digit does."""
    candidates = [
        c
        for c in drawing.characteristics
        if c.kind == "dimension"
        and c.nominal
        and c.dim_type in {"linear", "diameter", "radius"}
        and not c.is_reference
    ]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    before = target.nominal or 0.0

    # A digit error, not a nudge: swap a digit, or shift by a decade. Both are what happens
    # when a person or a model reads 44.5 as 4.45 or 14.5.
    style = rng.integers(3)
    if style == 0:
        after = round(before * 10, 4) if before < 100 else round(before / 10, 4)
    elif style == 1:
        after = round(before + max(1.0, 0.25 * before), 4)
    else:
        after = round(before * 0.5, 4)
    if abs(after - before) < 1e-6:
        return None

    target.nominal = after
    target.raw_text = (target.raw_text or "").replace(f"{before:g}", f"{after:g}")
    return Injection(
        kind="perturb_nominal",
        characteristic_id=target.id,
        detail=f"nominal {before:g} -> {after:g}",
        before=before,
        after=after,
    )


def _swap_datum(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Point a frame's primary reference at a different declared datum."""
    labels = [d.label for d in drawing.datums]
    if len(labels) < 2:
        return None
    candidates = [c for c in drawing.characteristics if c.datum_refs]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    current = target.datum_refs[0].label
    alternatives = [label for label in labels if label != current]
    if not alternatives:
        return None
    replacement = alternatives[int(rng.integers(len(alternatives)))]

    used = {ref.label for ref in target.datum_refs}
    if replacement in used:
        # Swapping onto a label the frame already carries would repeat a reference, which
        # the schema rejects outright -- a different failure from the one being injected.
        return None

    before_refs = [ref.label for ref in target.datum_refs]
    target.datum_refs[0].label = replacement
    after_refs = [ref.label for ref in target.datum_refs]
    return Injection(
        kind="swap_datum",
        characteristic_id=target.id,
        detail=f"primary datum {current} -> {replacement}",
        before=current,
        after=replacement,
        benign=_same_constraint(drawing, target.gtol_symbol, before_refs, after_refs),
    )


def _flip_modifier(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Drop a material condition modifier, or invent one."""
    candidates = [
        c
        for c in drawing.characteristics
        if c.kind == "geometric_tolerance"
        and c.gtol_symbol in {"position", "perpendicularity", "parallelism", "angularity"}
    ]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    before = target.material_modifier
    after = None if before else "MMC"
    target.material_modifier = after
    if target.raw_text:
        target.raw_text = target.raw_text.replace("Ⓜ", "") if before else target.raw_text
    return Injection(
        kind="flip_modifier",
        characteristic_id=target.id,
        detail=f"material modifier {before} -> {after}",
        before=before,
        after=after,
        # Removing a modifier leaves a tolerance stricter than intended but not
        # geometrically false. Adding one is legal wherever the feature has a size
        # tolerance -- which, on a sheet with a general tolerance note, every feature does.
        benign=after is None or bool(drawing.title_block.general_tolerance),
    )


def _drop_characteristic(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Remove a callout, as a missed extraction does."""
    candidates = [
        c
        for c in drawing.characteristics
        if not (c.kind == "geometric_tolerance" and c.datum_refs)
    ] or list(drawing.characteristics)
    if len(candidates) < 2:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    remaining = [c for c in drawing.characteristics if c.id != target.id]
    for new_id, c in enumerate(remaining, start=1):
        c.id = new_id
    drawing.characteristics = remaining
    return Injection(
        kind="drop_characteristic",
        characteristic_id=target.id,
        detail=f"removed {target.raw_text or target.kind}",
        before=target.raw_text,
        # A drawing missing one callout is still true about the ones it keeps. No geometric
        # check can see an absence, and this is here to measure that honestly rather than to
        # be caught: it is the extraction metrics in evalkit that catch a miss.
        benign=True,
    )


def _shuffle_datum_refs(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Reverse a frame's reference order, changing which datum constrains first."""
    candidates = [c for c in drawing.characteristics if len(c.datum_refs) >= 2]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    before = [ref.label for ref in target.datum_refs]
    target.datum_refs = list(reversed(target.datum_refs))
    after = [ref.label for ref in target.datum_refs]
    return Injection(
        kind="shuffle_datum_refs",
        characteristic_id=target.id,
        detail=f"datum order {'|'.join(before)} -> {'|'.join(after)}",
        before=before,
        after=after,
        benign=_same_constraint(drawing, target.gtol_symbol, before, after),
    )


def _same_constraint(
    drawing: Drawing, symbol: str | None, before: list[str], after: list[str]
) -> bool:
    """Whether both reference frames are equally *sufficient* for this tolerance.

    The honest definition of "undetectable", computed by asking the verifier's own check
    rather than guessing from feature types. A frame that was adequate and stayed adequate
    describes a part that can still be inspected, whatever letters it names -- so no check
    working from geometry alone can tell the two apart, and it should not pretend to.
    Catching a rewired reference frame is the job of ``evalkit``'s tier-3 datum graph
    distance, which has ground truth to compare against.
    """
    from balloonbench.verifier.checks.datum_dof import is_sufficient

    if symbol is None:
        return True
    types = {d.label: d.feature_type for d in drawing.datums}
    return is_sufficient(symbol, [types.get(label, "") for label in before]) and (
        is_sufficient(symbol, [types.get(label, "") for label in after])
    )


def _underconstrain_drf(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Strip a locating frame down to its primary datum.

    SPEC.md section 12.2's own example of a drawing defect: a position tolerance referencing
    one planar datum leaves the part free to slide and spin in that plane, so the callout
    does not say where the feature is. Unlike a datum swap this *is* visible from the
    drawing alone, which is what makes it worth measuring.
    """
    from balloonbench.verifier.checks.datum_dof import is_sufficient

    types = {d.label: d.feature_type for d in drawing.datums}
    candidates = [
        c
        for c in drawing.characteristics
        if len(c.datum_refs) >= 2 and c.gtol_symbol
    ]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    before = [ref.label for ref in target.datum_refs]
    target.datum_refs = target.datum_refs[:1]
    after = [ref.label for ref in target.datum_refs]
    return Injection(
        kind="underconstrain_drf",
        characteristic_id=target.id,
        detail=f"reference frame {'|'.join(before)} -> {'|'.join(after)}",
        before=before,
        after=after,
        benign=is_sufficient(target.gtol_symbol or "", [types.get(x, "") for x in after]),
    )


def _undeclared_datum(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Reference a datum letter the drawing never establishes."""
    declared = {d.label for d in drawing.datums}
    candidates = [c for c in drawing.characteristics if c.datum_refs]
    if not candidates:
        return None
    target = candidates[int(rng.integers(len(candidates)))]
    unused = [letter for letter in "XYZWV" if letter not in declared]
    if not unused:
        return None
    replacement = unused[0]
    before = target.datum_refs[-1].label
    target.datum_refs[-1].label = replacement
    return Injection(
        kind="undeclared_datum",
        characteristic_id=target.id,
        detail=f"datum reference {before} -> {replacement}, which is never established",
        before=before,
        after=replacement,
    )


def _unit_confusion(drawing: Drawing, rng: np.random.Generator) -> Injection | None:
    """Record every size as though the sheet were in the other unit system.

    The silent, expensive failure of SPEC.md section 12.2: the output is internally
    consistent, passes every schema rule, and is wrong by a factor of 25.4 throughout.
    """
    changed = 0
    factor = 25.4 if drawing.units == "mm" else 1 / 25.4
    for c in drawing.characteristics:
        if c.kind != "dimension" or c.nominal is None:
            continue
        c.nominal = round(c.nominal * factor, 4)
        changed += 1
    if changed == 0:
        return None
    return Injection(
        kind="unit_confusion",
        characteristic_id=0,
        detail=f"every one of {changed} sizes scaled by {factor:.4g}",
        before=drawing.units,
        after=drawing.units,
    )


INJECTORS: dict[str, Callable[[Drawing, np.random.Generator], Injection | None]] = {
    "perturb_nominal": _perturb_nominal,
    "swap_datum": _swap_datum,
    "flip_modifier": _flip_modifier,
    "drop_characteristic": _drop_characteristic,
    "shuffle_datum_refs": _shuffle_datum_refs,
    "underconstrain_drf": _underconstrain_drf,
    "undeclared_datum": _undeclared_datum,
    "unit_confusion": _unit_confusion,
}


def injector_names() -> tuple[str, ...]:
    return tuple(INJECTORS)


def inject(
    drawing: Drawing,
    kind: str,
    seed: int,
) -> tuple[Drawing, Injection | None]:
    """Return a damaged copy of ``drawing`` and a record of the damage.

    The copy is deep, so a caller can inject into the same ground truth repeatedly without
    the errors compounding -- which they would otherwise, silently, and every measured rate
    after the first would be meaningless.
    """
    if kind not in INJECTORS:
        raise KeyError(f"unknown injection {kind!r}; known: {sorted(INJECTORS)}")
    damaged = drawing.model_copy(deep=True)
    injection = INJECTORS[kind](damaged, np.random.default_rng(seed))
    if injection is None:
        return drawing.model_copy(deep=True), None
    return damaged, injection
