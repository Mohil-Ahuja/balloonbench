"""Running a baseline over a set of drawings, and recording what was run.

SPEC.md section 11 asks for results committed with pinned model versions, which is the part
of a leaderboard that is easy to skip and expensive to skip. A row saying "Claude, 61% F1"
is unreproducible: it does not say which snapshot, at what temperature, behind which prompt,
or how many of the calls were served from a cache rather than made. The manifest written
here says all of it, next to the predictions it describes.

Predictions are written one file per drawing rather than one file per run. A five-hundred
drawing run against a paid API takes a long time and gets interrupted; per-drawing files
mean an interrupted run resumes by skipping what exists, and a single unparseable reply
costs one drawing rather than the whole file.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from balloonbench import __version__
from balloonbench.baselines import vector_hybrid, vlm_structured, vlm_zeroshot
from balloonbench.baselines.base import BaselineResult, PromptConfig
from balloonbench.baselines.cache import ResponseCache
from balloonbench.baselines.providers import Provider
from balloonbench.schema import Drawing

__all__ = ["BASELINES", "RunManifest", "image_for", "run_baseline"]

#: The baselines available. ``detr_ocr`` joins them at M9; the registry is the single place
#: the CLI and the tests look.
BASELINES: dict[str, Callable[..., BaselineResult]] = {
    vlm_zeroshot.NAME: vlm_zeroshot.predict,
    vlm_structured.NAME: vlm_structured.predict,
    vector_hybrid.NAME: vector_hybrid.predict,
}


def image_for(json_path: Path) -> Path | None:
    """The rendered PNG belonging to a ground-truth file.

    ``drawgen`` writes the raster with its resolution in the name and a QA overlay beside
    it. The overlay has the ground-truth boxes painted on, so handing one to a model would
    be showing it the answer -- it is excluded here rather than merely ranked lower.
    """
    stem = json_path.stem
    candidates = sorted(
        p
        for p in json_path.parent.glob(f"{stem}*.png")
        if not p.stem.endswith("_overlay")
    )
    return candidates[0] if candidates else None


@dataclass
class RunManifest:
    """Everything needed to say what produced a set of predictions."""

    baseline: str
    provider: str
    model: str
    prompt_variant: str
    temperature: float
    samples: int
    n_drawings: int = 0
    n_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    balloonbench_version: str = __version__
    python: str = field(default_factory=platform.python_version)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def run_baseline(
    name: str,
    drawings: list[Path],
    out_dir: Path,
    *,
    provider: Provider,
    provider_name: str,
    model: str,
    cache: ResponseCache,
    config: PromptConfig | None = None,
    samples: int = 3,
    resume: bool = True,
    on_result: Callable[[str, BaselineResult], None] | None = None,
    **kwargs: Any,
) -> RunManifest:
    """Run one baseline over a list of ground-truth files.

    Only the sheet size, projection convention and units are taken from the ground truth and
    handed to the model. They are printed on the sheet, so a human reading the drawing has
    them; nothing that a reader would have to work out is passed.
    """
    try:
        predict = BASELINES[name]
    except KeyError:
        raise KeyError(
            f"unknown baseline {name!r}; known: {sorted(BASELINES)}"
        ) from None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = _default_config(name)
    if name == vector_hybrid.NAME:
        # It votes over nothing and tiles nothing: it reads the PDF once.
        samples = 1
        kwargs.pop("samples", None)
        kwargs.pop("tile", None)
    elif name == vlm_zeroshot.NAME:
        # The control condition is a single answer by construction. Voting is the treatment
        # that distinguishes the structured baseline, so it must not leak into the control.
        samples = 1
    else:
        kwargs.setdefault("samples", samples)

    manifest = RunManifest(
        baseline=name,
        provider=provider_name,
        model=model,
        prompt_variant=config.variant,
        temperature=config.temperature,
        samples=samples,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    for json_path in drawings:
        try:
            drawing = Drawing.from_json(json_path)
        except Exception as exc:  # noqa: BLE001 - not every JSON in a tree is a drawing
            manifest.errors.append(f"{json_path}: {exc}")
            continue

        target = out_dir / f"{drawing.drawing_id}.json"
        if resume and target.exists():
            manifest.n_drawings += 1
            continue

        image = image_for(json_path)
        if image is None:
            manifest.n_failed += 1
            manifest.errors.append(f"{json_path}: no rendered image beside it")
            continue

        result = predict(
            image,
            provider=provider,
            model=model,
            drawing_id=drawing.drawing_id,
            cache=cache,
            config=config,
            sheet=drawing.sheet.size,
            projection=drawing.projection,
            units=drawing.units,
            **kwargs,
        )

        manifest.cache_hits += result.cache_hits
        manifest.cache_misses += result.cache_misses
        manifest.errors.extend(f"{drawing.drawing_id}: {e}" for e in result.errors)
        manifest.n_drawings += 1
        if result.errors and not result.prediction.characteristics:
            manifest.n_failed += 1

        _write_prediction(target, result, drawing)
        if on_result is not None:
            on_result(drawing.drawing_id, result)

    manifest.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.write(out_dir / "manifest.json")
    return manifest


def _default_config(name: str) -> PromptConfig:
    from balloonbench.baselines.prompts import STRUCTURED, ZEROSHOT

    return ZEROSHOT if name == vlm_zeroshot.NAME else STRUCTURED


def _write_prediction(path: Path, result: BaselineResult, drawing: Drawing) -> None:
    prediction = result.prediction
    payload = {
        "drawing_id": prediction.drawing_id,
        "units": prediction.units,
        "characteristics": [
            c.model_dump(exclude_none=False) for c in prediction.characteristics
        ],
        "datums": [d.model_dump(exclude_none=False) for d in prediction.datums],
        "meta": {
            **prediction.meta,
            "malformed": [
                {"index": m.index, "reason": m.reason} for m in prediction.malformed
            ],
            "errors": result.errors,
            "cache": {"hits": result.cache_hits, "misses": result.cache_misses},
            # The source drawing's condition, carried through so a report can be sliced by
            # it without re-reading the ground truth alongside every prediction.
            "degradation_profile": drawing.provenance.degradation_profile,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
