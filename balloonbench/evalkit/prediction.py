"""A model's answer, loaded leniently.

Ground truth goes through :class:`~balloonbench.schema.Drawing` and must satisfy every rule
in the contract. A prediction must not: the rules are exactly the things a model gets wrong,
and a harness that refused to load a flatness carrying a datum reference would be unable to
report the most interesting failure it has.

So predictions are parsed one characteristic at a time. Anything that validates becomes a
:class:`~balloonbench.schema.Characteristic` and takes its chances in the matcher; anything
that does not is kept as a :class:`Malformed` with the reason, counted as a false positive,
and reported separately. That count is a finding in its own right -- "this model produces
output that will not load into an inspection form 12% of the time" is a number a quality
manager understands immediately.

The drawing-level rules are relaxed for the same reason. Ids need not be contiguous, because
a model that skips a balloon number has made a numbering mistake and not an unloadable file;
boxes need not lie inside the image, because a hallucinated coordinate is a localisation
error the matcher should see and score, not a parse error that hides it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from balloonbench.schema import Characteristic, Datum, Drawing

__all__ = ["Malformed", "Prediction", "load_prediction", "parse_prediction"]


@dataclass(frozen=True)
class Malformed:
    """A predicted item that could not be a characteristic at all."""

    index: int
    reason: str
    raw: dict[str, Any]


@dataclass
class Prediction:
    """What a baseline produced for one drawing."""

    drawing_id: str
    characteristics: list[Characteristic] = field(default_factory=list)
    datums: list[Datum] = field(default_factory=list)
    units: str = "mm"
    malformed: list[Malformed] = field(default_factory=list)
    #: Free-form record of who produced this and how: model id, prompt hash, run date.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_predicted(self) -> int:
        """Every item the model emitted, malformed ones included.

        Malformed items count. Dropping them from the denominator would let a model improve
        its precision by emitting garbage, which is the opposite of what the number should
        reward.
        """
        return len(self.characteristics) + len(self.malformed)

    @classmethod
    def from_ground_truth(cls, drawing: Drawing) -> Prediction:
        """A perfect prediction. Used by the tests as the fixed point every metric must
        report as flawless, and as the base that perturbation tests deviate from."""
        copy = drawing.model_copy(deep=True)
        return cls(
            drawing_id=copy.drawing_id,
            characteristics=list(copy.characteristics),
            datums=list(copy.datums),
            units=copy.units,
        )


def parse_prediction(data: dict[str, Any]) -> Prediction:
    """Build a :class:`Prediction` from a loosely-structured dictionary.

    Unknown keys on a characteristic are a validation failure rather than a warning: a model
    inventing ``"tolerance_class"`` has produced something the schema cannot represent, and
    silently discarding the field would score the rest of the item as if it were fine.
    """
    prediction = Prediction(
        drawing_id=str(data.get("drawing_id", "")),
        units=str(data.get("units", "mm")),
        meta=dict(data.get("meta", {})),
    )

    for index, raw in enumerate(data.get("characteristics", []) or []):
        if not isinstance(raw, dict):
            prediction.malformed.append(
                Malformed(index=index, reason=f"not an object: {type(raw).__name__}", raw={})
            )
            continue
        try:
            prediction.characteristics.append(Characteristic.model_validate(raw))
        except ValidationError as exc:
            prediction.malformed.append(
                Malformed(index=index, reason=_first_reason(exc), raw=raw)
            )

    for raw in data.get("datums", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            prediction.datums.append(Datum.model_validate(raw))
        except ValidationError:
            # A malformed datum is not itself a scored item; its absence shows up in the
            # tier-3 graph distance, which is where a broken reference frame belongs.
            continue

    return prediction


def _first_reason(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
    return f"{location}: {error.get('msg', 'invalid')}"


def load_prediction(path: str | Path) -> Prediction:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prediction = parse_prediction(data)
    if not prediction.drawing_id:
        prediction.drawing_id = Path(path).stem
    return prediction
