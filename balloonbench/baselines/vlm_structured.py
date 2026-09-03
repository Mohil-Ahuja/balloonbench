"""Baseline 2: the same models, asked properly.

SPEC.md section 11 gives this baseline three treatments over the zero-shot control -- a
constrained output schema with a worked reading order, view-by-view tiling with the crops
merged, and a self-consistency pass at k=3 -- and one question: how much of the gap is
prompting and how much is capability. Keeping the model fixed and varying only the asking
is what makes that question answerable.

**Tiles are geometric, not semantic.** The crops are an overlapping grid, not one per view,
because a baseline is not allowed to know where the views are: that is in the ground truth,
and cropping to it would leak the answer into the question. The overlap exists so a callout
straddling a seam is whole in at least one crop, and the prompt tells the model to skip
anything cut off at an edge for the same reason.

**Voting is by agreement, not by union.** Three samples are pooled and a callout is kept
only if it appears in at least two of them. A union would raise recall and wreck precision,
turning self-consistency into a synonym for "sample more", and the whole point of the k=3
pass is that agreement is evidence and a lone hallucination is not.

Merging both the tiles and the votes uses ``evalkit``'s matcher, deliberately. The question
"are these two the same callout?" already has an answer in this repository, and a second
one written here could disagree with it -- so a model could be scored as correct by the
metric and deduplicated as wrong by the baseline, or the reverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from balloonbench.baselines.prompts import STRUCTURED, TILE, schema_description
from balloonbench.baselines.providers import Provider, ProviderError
from balloonbench.evalkit.matching import MatchConfig, match
from balloonbench.evalkit.prediction import Prediction
from balloonbench.schema import Characteristic

__all__ = ["NAME", "Tile", "merge_by_vote", "predict", "tiles_for"]

NAME = "vlm_structured"

#: How much neighbouring crops overlap, as a fraction of a tile. A callout is a few hundred
#: pixels wide on a 300 DPI sheet; a tenth of a tile is comfortably more than that, so no
#: callout can be cut off in every crop that contains it.
TILE_OVERLAP = 0.12

#: Matching threshold for deciding two readings describe the same callout. Tighter than the
#: scoring default: this is not "did the model find it" but "is this the same thing twice",
#: and merging two genuinely different callouts destroys information that no later stage
#: can recover.
MERGE_CONFIG = MatchConfig(max_cost=0.35)


@dataclass(frozen=True)
class Tile:
    """A crop of the sheet, and where it sits in it."""

    box: tuple[int, int, int, int]

    @property
    def offset(self) -> tuple[int, int]:
        return self.box[0], self.box[1]

    @property
    def size(self) -> tuple[int, int]:
        return self.box[2] - self.box[0], self.box[3] - self.box[1]


def tiles_for(size: tuple[int, int], *, rows: int = 2, cols: int = 2) -> list[Tile]:
    """An overlapping grid of crops covering the sheet."""
    width, height = size
    tile_w = width / cols
    tile_h = height / rows
    pad_x = tile_w * TILE_OVERLAP
    pad_y = tile_h * TILE_OVERLAP
    out: list[Tile] = []
    for row in range(rows):
        for col in range(cols):
            x0 = max(0, int(col * tile_w - pad_x))
            y0 = max(0, int(row * tile_h - pad_y))
            x1 = min(width, int((col + 1) * tile_w + pad_x))
            y1 = min(height, int((row + 1) * tile_h + pad_y))
            out.append(Tile(box=(x0, y0, x1, y1)))
    return out


def _translate(c: Characteristic, dx: int, dy: int) -> Characteristic:
    """A copy of a callout with its boxes moved from crop into sheet coordinates."""
    moved = c.model_copy(deep=True)
    moved.bbox = [c.bbox[0] + dx, c.bbox[1] + dy, c.bbox[2] + dx, c.bbox[3] + dy]
    if c.leader_target_bbox is not None:
        moved.leader_target_bbox = [
            c.leader_target_bbox[0] + dx,
            c.leader_target_bbox[1] + dy,
            c.leader_target_bbox[2] + dx,
            c.leader_target_bbox[3] + dy,
        ]
    return moved


def _same(a: Characteristic, b: Characteristic, config: MatchConfig) -> bool:
    return match([a], [b], config).n_matched == 1


def merge_by_vote(
    runs: list[list[Characteristic]],
    *,
    min_votes: int | None = None,
    config: MatchConfig = MERGE_CONFIG,
) -> list[Characteristic]:
    """Pool several readings of the same sheet and keep what most of them agree on.

    Clusters are grown greedily and a cluster takes at most one reading per run, so three
    copies of a callout from one talkative sample cannot outvote the two samples that never
    saw it. The kept representative is the first reading in the cluster rather than an
    average of them: averaging two disagreeing tolerance values would invent a third that
    no model produced.
    """
    if not runs:
        return []
    votes = min_votes if min_votes is not None else max(1, math.ceil(len(runs) / 2))

    clusters: list[tuple[Characteristic, set[int]]] = []
    for index, run in enumerate(runs):
        for c in run:
            for representative, supporters in clusters:
                if index not in supporters and _same(representative, c, config):
                    supporters.add(index)
                    break
            else:
                clusters.append((c, {index}))

    kept = [rep for rep, supporters in clusters if len(supporters) >= votes]
    for new_id, c in enumerate(kept, start=1):
        c.id = new_id
    return kept


def _dedupe(
    items: list[Characteristic], config: MatchConfig = MERGE_CONFIG
) -> list[Characteristic]:
    """Collapse readings of the same callout that came from overlapping crops."""
    out: list[Characteristic] = []
    for c in items:
        if not any(_same(existing, c, config) for existing in out):
            out.append(c)
    return out


def predict(
    image_path: str | Path,
    *,
    provider: Provider,
    model: str,
    drawing_id: str,
    cache: ResponseCache,
    config: PromptConfig | None = None,
    tile_config: PromptConfig | None = None,
    samples: int = 3,
    tile: bool = True,
    sheet: str = "unknown",
    projection: str = "unknown",
    units: str = "mm",
) -> BaselineResult:
    """Read one drawing with the full treatment: structured prompt, tiles, and a vote."""
    config = config or STRUCTURED
    tile_config = tile_config or TILE
    image_path = Path(image_path)
    schema = schema_description()

    result = BaselineResult(prediction=to_prediction(None, drawing_id=drawing_id))

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        crops: list[tuple[Tile, Path]] = []
        if tile:
            crops = _write_crops(image, image_path, cache.root)

    runs: list[list[Characteristic]] = []
    for sample in range(samples):
        readings: list[Characteristic] = []

        text = _ask(
            result,
            provider,
            model=model,
            config=config,
            rendered=config.render(
                sheet=sheet, projection=projection, width=width, height=height,
                schema=schema,
            ),
            image_path=image_path,
            cache=cache,
            sample=sample,
        )
        if text is not None:
            readings += to_prediction(
                extract_json(text), drawing_id=drawing_id, units=units
            ).characteristics

        for crop_tile, crop_path in crops:
            crop_w, crop_h = crop_tile.size
            dx, dy = crop_tile.offset
            text = _ask(
                result,
                provider,
                model=model,
                config=tile_config,
                rendered=tile_config.render(
                    width=crop_w, height=crop_h, offset_x=dx, offset_y=dy, schema=schema
                ),
                image_path=crop_path,
                cache=cache,
                sample=sample,
                extra=(("tile", f"{dx},{dy},{crop_w},{crop_h}"),),
            )
            if text is None:
                continue
            crop_reading = to_prediction(
                extract_json(text), drawing_id=drawing_id, units=units
            ).characteristics
            readings += [_translate(c, dx, dy) for c in crop_reading]

        runs.append(_dedupe(readings))

    merged = merge_by_vote(runs)
    prediction = Prediction(
        drawing_id=drawing_id,
        characteristics=merged,
        units=units,
        meta={
            "baseline": NAME,
            "model": model,
            "prompt_variant": config.variant,
            "samples": samples,
            "tiles": len(crops),
        },
    )
    result.prediction = prediction
    return result


def _ask(
    result: BaselineResult,
    provider: Provider,
    **kwargs,
) -> str | None:
    """One call, with its bookkeeping. Returns ``None`` when the provider refused."""
    try:
        text, hit = call_model(provider, **kwargs)
    except ProviderError as exc:
        result.errors.append(str(exc))
        result.cache_misses += 1
        return None
    result.raw_responses.append(text)
    result.cache_hits += int(hit)
    result.cache_misses += int(not hit)
    return text


def _write_crops(image: Image.Image, image_path: Path, root: Path) -> list[tuple[Tile, Path]]:
    """Write the crops beside the cache and return them.

    On disk rather than in memory because the crop is what gets hashed into the cache key:
    the same crop of the same sheet must produce the same key on a later run, and a
    re-encoded in-memory buffer is not guaranteed to be byte-identical.
    """
    directory = Path(root) / "crops" / image_path.stem
    directory.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Tile, Path]] = []
    for crop_tile in tiles_for(image.size):
        x0, y0, x1, y1 = crop_tile.box
        path = directory / f"{x0}_{y0}_{x1}_{y1}.png"
        if not path.exists():
            image.crop(crop_tile.box).save(path)
        out.append((crop_tile, path))
    return out
