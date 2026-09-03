"""What a verifier check is, and the three verdicts it may reach.

SPEC.md section 12.4 states the constraint that shapes every line of this module: *a
verifier that flags correct extractions is worse than no verifier.* A false contradiction
sends a quality engineer to re-read a characteristic that was right, and after a few of
those nobody looks at the output again. So the design is asymmetric on purpose. Everything
uncertain lands in ``unverifiable``; ``contradicted`` is reserved for cases where the
geometry positively disagrees.

That asymmetry is why ``unverifiable`` is not a failure bucket. It is the answer "this needs
a human", and SPEC.md section 12.3 calls it the commercially important one: in a regulated
environment, a tool that says which characteristics it could not confirm is deployable,
while one that emits a confidence score for everything is not.

A check returns two different kinds of finding and they must not be confused:

* a **verdict** on an extracted characteristic -- does the drawing agree with the solid?
* a **drawing defect** -- the drawing is internally wrong, whatever was extracted.

The second is not an extraction error, and reporting it as one would blame the model for the
draughtsman's mistake. Kept apart, it becomes the feature SPEC.md section 12.2 points at:
the tool finds problems in the customer's own drawings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from balloonbench.evalkit.matching import to_mm
from balloonbench.schema import Characteristic, Drawing
from balloonbench.verifier.brep_index import BrepIndex

__all__ = [
    "CheckContext",
    "Defect",
    "Verdict",
    "Verification",
    "size_of",
]

Verdict = Literal["verified", "contradicted", "unverifiable"]


@dataclass
class Verification:
    """One check's opinion about one characteristic."""

    id: int
    verdict: Verdict
    check: str
    detail: str
    #: How sure the check is, in ``[0, 1]``. Reported rather than thresholded, because the
    #: right threshold depends on what the reader is doing with the answer.
    confidence: float = 1.0
    suggested_correction: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "verdict": self.verdict,
            "check": self.check,
            "detail": self.detail,
            "confidence": round(self.confidence, 3),
        }
        if self.suggested_correction:
            out["suggested_correction"] = self.suggested_correction
        return out


@dataclass
class Defect:
    """Something wrong with the drawing itself, rather than with an extraction."""

    type: str
    detail: str
    characteristic_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "characteristic_id": self.characteristic_id,
            "detail": self.detail,
        }


@dataclass
class CheckContext:
    """Everything a check is given: the drawing, the solid, and the unit conversion.

    Values reach a check already converted to millimetres. An inch drawing is not a special
    case to be remembered in five places -- forgetting it in one would produce a verifier
    that contradicts every dimension on a correct imperial sheet.
    """

    drawing: Drawing
    index: BrepIndex
    verifications: list[Verification] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)

    def mm(self, value: float | None) -> float | None:
        return to_mm(value, self.drawing.units)

    def verify(self, c: Characteristic, check: str, detail: str, confidence: float = 1.0):
        self.verifications.append(
            Verification(id=c.id, verdict="verified", check=check, detail=detail,
                         confidence=confidence)
        )

    def contradict(
        self,
        c: Characteristic,
        check: str,
        detail: str,
        *,
        confidence: float = 0.9,
        correction: dict[str, Any] | None = None,
    ) -> None:
        self.verifications.append(
            Verification(
                id=c.id, verdict="contradicted", check=check, detail=detail,
                confidence=confidence, suggested_correction=correction,
            )
        )

    def unverifiable(self, c: Characteristic, check: str, detail: str) -> None:
        self.verifications.append(
            Verification(
                id=c.id, verdict="unverifiable", check=check, detail=detail, confidence=0.5
            )
        )

    def defect(self, type_: str, detail: str, characteristic_id: int | None = None) -> None:
        self.defects.append(
            Defect(type=type_, detail=detail, characteristic_id=characteristic_id)
        )


def size_of(c: Characteristic) -> float | None:
    """The number a size dimension states, or ``None`` if it states none."""
    return c.nominal if c.kind == "dimension" else None
