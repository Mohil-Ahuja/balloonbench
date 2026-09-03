"""Are these numbers in the units the drawing claims?

SPEC.md section 12.2 calls this "an entire class of silent, expensive failures", and the
description is exact. A model that reads an inch drawing and reports the numbers as
millimetres produces output that is internally consistent, passes every schema rule, and is
wrong by a factor of 25.4. Nothing catches it except comparing the numbers against a part.

The test is a ratio, not a value. The largest dimension on a drawing is close to the largest
extent of the part -- not equal to it, since the biggest number is often a bolt circle or a
bore rather than the overall length, but the same order of magnitude. So:

* a ratio near 1 is right;
* a ratio near 25.4 means the values were read as millimetres on an inch drawing;
* a ratio near 1/25.4 means the reverse.

Everything else is inconclusive and is reported as nothing at all. A part whose largest
dimension is a third of its envelope tells us the drawing does not dimension its overall
size, which is unusual but not an error, and a check that complained about it would be
noise.

The verdict lands on the drawing as a whole rather than on any one characteristic, because
the failure is never a single wrong number. It is every number at once, which is precisely
what makes it detectable by a ratio and undetectable by reading any one of them.
"""

from __future__ import annotations

from balloonbench.verifier.base import CheckContext

__all__ = ["NAME", "run"]

NAME = "unit_sanity"

MM_PER_INCH = 25.4

#: How close a ratio must be to a suspected factor before it is called that factor. Wide,
#: because the largest dimension is not the envelope and the two are only expected to agree
#: to within a factor of a few.
TOLERANCE = 0.35

#: A part smaller than this in every direction gives a ratio too noisy to reason about.
MIN_ENVELOPE = 1.0


def run(context: CheckContext) -> None:
    envelope = max(context.index.envelope, default=0.0)
    if envelope < MIN_ENVELOPE:
        return

    sizes = [
        context.mm(c.nominal) or 0.0
        for c in context.drawing.characteristics
        if c.kind == "dimension"
        and c.dim_type in {"linear", "diameter", "radius"}
        and c.nominal is not None
    ]
    sizes = [size for size in sizes if size > 0]
    if not sizes:
        return

    largest = max(sizes)
    ratio = largest / envelope

    if _near(ratio, 1.0):
        return
    if _near(ratio, 1 / MM_PER_INCH):
        context.defect(
            "unit_mismatch",
            f"the largest dimension on the sheet is {largest:.2f} mm but the solid is "
            f"{envelope:.2f} mm across -- a factor of {envelope / max(largest, 1e-9):.1f}. "
            f"The values look like inches recorded as millimetres",
        )
    elif _near(ratio, MM_PER_INCH):
        context.defect(
            "unit_mismatch",
            f"the largest dimension on the sheet is {largest:.2f} mm but the solid is only "
            f"{envelope:.2f} mm across -- a factor of {ratio:.1f}. The values look like "
            f"millimetres recorded as inches",
        )
    # Any other ratio is inconclusive and deliberately produces no finding.


def _near(ratio: float, factor: float) -> bool:
    return abs(ratio - factor) <= TOLERANCE * factor
