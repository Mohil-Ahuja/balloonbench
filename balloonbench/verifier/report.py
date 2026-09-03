"""Running every check and assembling the verdict document of SPEC.md section 12.3.

One characteristic can be looked at by several checks, so the report has to decide what a
characteristic's verdict *is* when they disagree. The rule is the conservative one, and it
follows from the same principle as everything else in this module:

    contradicted  >  unverifiable  >  verified

A single contradiction stands even when three other checks were happy, because a check only
contradicts when the geometry positively disagrees, and the other checks were looking at
something else. An ``unverifiable`` likewise outranks a ``verified``: if any check could not
confirm the characteristic, the honest summary is that it needs a human, not that it passed.

Characteristics no check could look at are reported too, as ``unverifiable`` with the reason
"no applicable check". Silence would be indistinguishable from approval.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from balloonbench.schema import Drawing
from balloonbench.verifier.base import CheckContext, Defect, Verdict, Verification
from balloonbench.verifier.brep_index import BrepIndex
from balloonbench.verifier.checks import (
    datum_dof,
    mmc_consistency,
    size_exists,
    tolerance_stack,
    unit_sanity,
)

__all__ = ["CHECKS", "VerificationReport", "verify_drawing"]

#: Run in this order. Order does not change any verdict -- the ranking below settles that --
#: but it keeps the details attached to a characteristic in a readable sequence.
CHECKS = (size_exists, datum_dof, mmc_consistency, tolerance_stack, unit_sanity)

#: Worst wins.
_RANK: dict[Verdict, int] = {"verified": 0, "unverifiable": 1, "contradicted": 2}


@dataclass
class VerificationReport:
    drawing_id: str
    per_characteristic: list[Verification] = field(default_factory=list)
    drawing_defects: list[Defect] = field(default_factory=list)
    #: Every verdict from every check, before the worst-wins reduction. Kept because a
    #: reader chasing a contradiction wants to know what the other checks thought.
    all_verdicts: list[Verification] = field(default_factory=list)
    index_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        counts = {"verified": 0, "contradicted": 0, "unverifiable": 0}
        for verification in self.per_characteristic:
            counts[verification.verdict] += 1
        return counts

    def verdict_for(self, characteristic_id: int) -> Verdict | None:
        for verification in self.per_characteristic:
            if verification.id == characteristic_id:
                return verification.verdict
        return None

    @property
    def contradicted(self) -> list[int]:
        return [v.id for v in self.per_characteristic if v.verdict == "contradicted"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "drawing_id": self.drawing_id,
            "summary": self.summary,
            "per_characteristic": [v.as_dict() for v in self.per_characteristic],
            "drawing_defects": [d.as_dict() for d in self.drawing_defects],
            "index": self.index_stats,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8"
        )


def verify_drawing(drawing: Drawing, index: BrepIndex) -> VerificationReport:
    """Check one drawing against one solid.

    >>> # Exercised end to end in tests/test_verifier.py, where the false-positive rate on
    >>> # clean ground truth is the milestone's gate.
    """
    context = CheckContext(drawing=drawing, index=index)
    for check in CHECKS:
        check.run(context)

    context.defects.extend(_unused_datum_defects(drawing))

    by_id: dict[int, list[Verification]] = defaultdict(list)
    for verification in context.verifications:
        by_id[verification.id].append(verification)

    reduced: list[Verification] = []
    for c in drawing.characteristics:
        found = by_id.get(c.id)
        if not found:
            reduced.append(
                Verification(
                    id=c.id,
                    verdict="unverifiable",
                    check="none",
                    detail="no applicable check",
                    confidence=0.5,
                )
            )
            continue
        reduced.append(max(found, key=lambda v: (_RANK[v.verdict], v.confidence)))

    return VerificationReport(
        drawing_id=drawing.drawing_id,
        per_characteristic=reduced,
        drawing_defects=context.defects,
        all_verdicts=context.verifications,
        index_stats=index.stats(),
    )


def _unused_datum_defects(drawing: Drawing) -> list[Defect]:
    return [
        Defect(
            type="unused_datum",
            detail=(
                f"datum {label} is established on the drawing and referenced by no "
                f"tolerance; usually the remains of an earlier revision"
            ),
        )
        for label in datum_dof.unused_datums(drawing)
    ]
