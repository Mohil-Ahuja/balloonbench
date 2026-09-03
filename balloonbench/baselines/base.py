"""The uniform baseline interface, and the plumbing every baseline shares.

SPEC.md section 11 fixes the shape: ``predict(image_path, pdf_path | None, prompt_config)``
returning characteristics. Everything in this module exists so that the four baselines
differ only in the interesting way -- what they ask the model and how they combine the
answers -- and not in how they hash a request, recover JSON from prose, or turn a model's
loose object into something the harness can score.

Two pieces are worth reading closely.

**Recovering JSON.** Models wrap answers in prose, in fenced code blocks, or in both, and a
baseline that failed on a stray "Here is the JSON:" would be measuring markdown compliance
rather than extraction. :func:`extract_json` peels those off. What it deliberately does not
do is repair malformed content: a truncated object stays a parse failure, because silently
mending one would put a value in the results that no model produced.

**Normalisation before validation.** A predicted item is normalised in exactly two ways
before it is validated, and the choice of which two is a judgement about what is content
and what is bookkeeping. A missing ``id`` is filled from position, because balloon numbers
are the harness's numbering rather than something read off the drawing. A missing ``view``
becomes ``"unknown"``, which then scores as a view mismatch -- the model did not say, and
recording that as an error is more honest than either guessing "front" or discarding an
otherwise complete callout. Nothing else is touched: a wrong symbol stays wrong and an
illegal combination stays illegal, because those are the findings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from balloonbench.baselines.cache import CacheKey, ResponseCache, file_digest, text_digest
from balloonbench.baselines.providers import Provider, ProviderError
from balloonbench.evalkit.prediction import Malformed, Prediction, parse_prediction

__all__ = [
    "BaselineResult",
    "PromptConfig",
    "call_model",
    "extract_json",
    "to_prediction",
]


@dataclass(frozen=True)
class PromptConfig:
    """Everything about how a model is asked, hashed into the cache key.

    ``temperature`` defaults to zero so a single-sample run is as reproducible as an API
    allows. The self-consistency pass in ``vlm_structured`` raises it deliberately: three
    samples at temperature zero would be three copies of one answer, which measures nothing.
    """

    system: str
    template: str
    max_tokens: int = 8192
    temperature: float = 0.0
    #: Distinguishes prompt variants that share a template, so an experiment does not read
    #: another experiment's cached answers.
    variant: str = "v1"

    def render(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def hash_for(self, rendered: str) -> str:
        return text_digest(f"{self.variant}\n{self.system}\n{rendered}")


@dataclass
class BaselineResult:
    """One drawing's prediction, plus what it took to get it."""

    prediction: Prediction
    raw_responses: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return self.cache_hits + self.cache_misses


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Recover the JSON object from a model's reply, or ``None`` if there is not one.

    Tries a fenced block first, then the outermost balanced braces. Scanning for balance
    rather than taking everything between the first ``{`` and last ``}`` matters when a
    model writes a sentence containing a brace after its answer, which they do.
    """
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        stripped = candidate.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = _first_balanced_object(stripped)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            # A bare array of characteristics is a common and reasonable answer shape.
            return {"characteristics": payload}
    return None


def _first_balanced_object(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        start = text.find("[")
        if start < 0:
            return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalise(item: Any, index: int) -> Any:
    """Fill in the two fields that are bookkeeping rather than content. See the module
    docstring for why these two and nothing else."""
    if not isinstance(item, dict):
        return item
    out = dict(item)
    if "id" not in out or out["id"] in (None, 0, ""):
        out["id"] = index + 1
    if not out.get("view"):
        out["view"] = "unknown"
    return out


def to_prediction(
    payload: dict[str, Any] | None,
    *,
    drawing_id: str,
    units: str = "mm",
    meta: dict[str, Any] | None = None,
) -> Prediction:
    """Turn a model's parsed object into a scoreable prediction.

    A reply with no recoverable JSON is an empty prediction, not an exception. It counts as
    having found nothing, which is exactly what happened, and lets a run over five hundred
    drawings survive one model refusing to answer.
    """
    if payload is None:
        prediction = Prediction(drawing_id=drawing_id, units=units)
        prediction.malformed.append(
            Malformed(index=0, reason="no JSON object in the reply", raw={})
        )
        prediction.meta.update(meta or {})
        return prediction

    items = payload.get("characteristics")
    if items is None and isinstance(payload.get("results"), list):
        items = payload["results"]
    normalised = [
        _normalise(item, index) for index, item in enumerate(items or [])
    ]
    prediction = parse_prediction(
        {
            "drawing_id": drawing_id,
            "units": payload.get("units") or units,
            "characteristics": normalised,
            "datums": payload.get("datums") or [],
        }
    )
    prediction.meta.update(meta or {})
    return prediction


def call_model(
    provider: Provider,
    *,
    model: str,
    config: PromptConfig,
    rendered: str,
    image_path: str | Path | None,
    cache: ResponseCache,
    sample: int = 0,
    extra: tuple[tuple[str, str], ...] = (),
) -> tuple[str, bool]:
    """One cached call. Returns the reply and whether it came from the cache.

    The image hash is of the file's bytes. Hashing the path instead would be faster and
    wrong: a regenerated drawing at the same path is a different drawing, and the cache
    would answer for the old one.
    """
    key = CacheKey(
        model=model,
        prompt_hash=config.hash_for(rendered),
        image_hash="" if image_path is None else file_digest(image_path),
        temperature=config.temperature,
        sample=sample,
        extra=extra,
    )
    cached = cache.get(key)
    if cached is not None:
        return str(cached.get("response", {}).get("text", "")), True

    try:
        text = provider.complete(
            model=model,
            system=config.system,
            prompt=rendered,
            image_path=image_path,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - every SDK raises its own hierarchy
        raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

    cache.put(key, {"text": text})
    return text, False
