"""Baseline 1: one image, one prompt, one answer.

The control condition for every other baseline. It is what a competent engineer would try
first -- hand the drawing to a frontier model and describe the output format -- and the
number it produces is the one that everything else has to beat to justify its complexity.

Deliberately plain. No tiling, no voting, no retry on a bad parse. A retry would improve the
number while making it a measurement of persistence rather than of the model, and the point
of a control is that it holds constant everything the treatments vary.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from balloonbench.baselines.base import (
    BaselineResult,
    PromptConfig,
    call_model,
    extract_json,
    to_prediction,
)
from balloonbench.baselines.cache import ResponseCache
from balloonbench.baselines.prompts import ZEROSHOT, schema_description
from balloonbench.baselines.providers import Provider, ProviderError

__all__ = ["NAME", "predict"]

NAME = "vlm_zeroshot"


def predict(
    image_path: str | Path,
    *,
    provider: Provider,
    model: str,
    drawing_id: str,
    cache: ResponseCache,
    config: PromptConfig | None = None,
    sheet: str = "unknown",
    projection: str = "unknown",
    units: str = "mm",
) -> BaselineResult:
    """Read one drawing.

    ``sheet`` and ``projection`` are told to the model because they are printed on the sheet
    in the title block and in the projection symbol -- a human reading the drawing has them,
    so withholding them would make this a harder task than the one being measured, not a
    fairer one. Nothing else from the ground truth is passed.
    """
    config = config or ZEROSHOT
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size

    rendered = config.render(
        sheet=sheet,
        projection=projection,
        width=width,
        height=height,
        schema=schema_description(),
    )

    result = BaselineResult(prediction=to_prediction(None, drawing_id=drawing_id))
    try:
        text, hit = call_model(
            provider,
            model=model,
            config=config,
            rendered=rendered,
            image_path=image_path,
            cache=cache,
        )
    except ProviderError as exc:
        result.errors.append(str(exc))
        result.cache_misses += 1
        return result

    result.raw_responses.append(text)
    result.cache_hits += int(hit)
    result.cache_misses += int(not hit)
    result.prediction = to_prediction(
        extract_json(text),
        drawing_id=drawing_id,
        units=units,
        meta={"baseline": NAME, "model": model, "prompt_variant": config.variant},
    )
    return result
