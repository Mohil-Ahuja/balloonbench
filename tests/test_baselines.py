"""Milestone 5 tests for the baseline harness.

PLAN.md's M5 gate is a full run on 500 synthetic drawings with results committed and model
versions pinned. That run costs money and needs API keys, so it is a thing the repository's
owner does deliberately, not something a test suite does on every commit. What *can* be
tested -- and what this file tests -- is everything the run depends on being correct before
it is paid for:

* the cache actually prevents a second call, and does not confuse two requests that differ
  in any way that could change the answer;
* a model's reply is recovered from whatever prose it arrives wrapped in;
* a reply that is wrong, truncated or absent degrades to a scored prediction rather than an
  exception that ends the run;
* tiles translate back into sheet coordinates, and the vote keeps agreement rather than
  union;
* the manifest records what was run.

Every test uses :class:`ScriptedProvider`. No test in this file can make a network call:
the real providers are never constructed, so no API key is read and nothing is billed.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from balloonbench.baselines.base import (
    PromptConfig,
    call_model,
    extract_json,
    to_prediction,
)
from balloonbench.baselines.cache import CacheKey, ResponseCache, file_digest
from balloonbench.baselines.providers import ProviderError, ScriptedProvider, provider_for_model
from balloonbench.baselines.run import BASELINES, image_for, run_baseline
from balloonbench.baselines.vlm_structured import (
    TILE_OVERLAP,
    merge_by_vote,
    tiles_for,
)
from balloonbench.drawgen.generate import generate_drawing
from balloonbench.partgen.registry import load_families
from balloonbench.schema import Characteristic

load_families()

TEST_DPI = 100.0


@pytest.fixture(scope="module")
def drawing(tmp_path_factory):
    """One generated drawing with its artifacts on disk, shared across the module."""
    out = tmp_path_factory.mktemp("baselines")
    return generate_drawing("flange", 5, out, dpi=TEST_DPI)


def _reply(bundle, count: int = 3) -> str:
    """A model reply that reproduces the first ``count`` real callouts, wrapped in prose."""
    items = [
        c.model_dump(exclude_none=False) for c in bundle.drawing.characteristics[:count]
    ]
    return (
        "Here is what I found on the drawing:\n\n```json\n"
        + json.dumps({"characteristics": items})
        + "\n```\nLet me know if you need anything else."
    )


# --- the cache ------------------------------------------------------------------------


def _config() -> PromptConfig:
    return PromptConfig(system="s", template="t {x}", variant="test-v1")


def test_a_second_identical_request_never_reaches_the_provider(drawing, tmp_path):
    """The whole justification for the cache, stated as a test: re-running the eval is free."""
    provider = ScriptedProvider(responses=["first", "second"])
    cache = ResponseCache(root=tmp_path / "cache")
    config = _config()
    args = dict(
        model="test-model",
        config=config,
        rendered="t 1",
        image_path=drawing.paths["png"],
        cache=cache,
    )

    text, hit = call_model(provider, **args)
    assert (text, hit) == ("first", False)
    text, hit = call_model(provider, **args)
    assert (text, hit) == ("first", True), "the cached reply must be returned verbatim"
    assert len(provider.calls) == 1, "the provider was called twice for one request"
    assert cache.stats == {"hits": 1, "misses": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other-model"),
        ("temperature", 0.7),
        ("sample", 1),
        ("extra", (("tile", "0,0,10,10"),)),
        ("prompt_hash", "different"),
        ("image_hash", "different"),
    ],
)
def test_anything_that_could_change_the_answer_changes_the_key(field, value):
    base = CacheKey(model="m", prompt_hash="p", image_hash="i")
    other = CacheKey(**{**base.__dict__, field: value})
    assert base.digest() != other.digest(), f"{field} does not affect the cache key"


def test_the_image_is_hashed_by_content_not_by_path(drawing, tmp_path):
    """A regenerated drawing at the same path is a different drawing. Hashing the path would
    serve the old answer for the new sheet, which is the worst kind of cache bug: silent."""
    copy = tmp_path / "copy.png"
    copy.write_bytes(drawing.paths["png"].read_bytes())
    assert file_digest(copy) == file_digest(drawing.paths["png"])

    with Image.open(copy) as image:
        image.convert("RGB").resize((image.width // 2, image.height // 2)).save(copy)
    assert file_digest(copy) != file_digest(drawing.paths["png"])


def test_a_truncated_cache_entry_is_treated_as_absent(drawing, tmp_path):
    """A run killed mid-write must cost one response, not the whole cache."""
    cache = ResponseCache(root=tmp_path / "cache")
    key = CacheKey(model="m", prompt_hash="p", image_hash="i")
    cache.put(key, {"text": "hello"})
    path = cache.path_for(key)
    path.write_text('{"key": {"model": "m"', encoding="utf-8")
    assert cache.get(key) is None


def test_a_read_only_cache_writes_nothing(tmp_path):
    cache = ResponseCache(root=tmp_path / "cache", write=False)
    key = CacheKey(model="m", prompt_hash="p", image_hash="i")
    cache.put(key, {"text": "hello"})
    assert not cache.path_for(key).exists()


# --- recovering the answer ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"characteristics": []}',
        'Sure!\n```json\n{"characteristics": []}\n```',
        '```\n{"characteristics": []}\n```',
        'Here you go: {"characteristics": []} — hope that helps :)',
    ],
)
def test_json_is_recovered_from_whatever_it_arrives_wrapped_in(text):
    assert extract_json(text) == {"characteristics": []}


def test_a_bare_array_is_accepted_as_the_characteristic_list():
    assert extract_json('[{"id": 1}]') == {"characteristics": [{"id": 1}]}


def test_a_trailing_sentence_with_a_brace_does_not_break_the_scan():
    text = '{"characteristics": []}\n\nNote: some callouts {like this} were unclear.'
    assert extract_json(text) == {"characteristics": []}


def test_truncated_json_is_a_failure_not_a_repair():
    """Mending a cut-off reply would put a value in the results that no model produced."""
    assert extract_json('{"characteristics": [{"id": 1, "kind": "dimen') is None


def test_a_missing_id_is_filled_but_a_missing_symbol_is_not():
    """Balloon numbers are the harness's bookkeeping; a geometric symbol is content."""
    payload = {
        "characteristics": [
            {"kind": "dimension", "view": "front", "bbox": [1, 1, 20, 10],
             "dim_type": "linear", "nominal": 10.0},
            {"kind": "geometric_tolerance", "view": "front", "bbox": [1, 20, 20, 30],
             "gtol_value": 0.05},
        ]
    }
    prediction = to_prediction(payload, drawing_id="d")
    assert [c.id for c in prediction.characteristics] == [1]
    assert len(prediction.malformed) == 1
    assert "gtol_symbol" in prediction.malformed[0].reason


def test_a_missing_view_becomes_unknown_rather_than_a_guess():
    payload = {
        "characteristics": [
            {"id": 1, "kind": "dimension", "view": "", "bbox": [1, 1, 20, 10],
             "dim_type": "linear", "nominal": 10.0}
        ]
    }
    prediction = to_prediction(payload, drawing_id="d")
    assert prediction.characteristics[0].view == "unknown"


def test_no_json_at_all_is_an_empty_prediction_not_an_exception():
    prediction = to_prediction(extract_json("I cannot read this drawing."), drawing_id="d")
    assert prediction.characteristics == []
    assert prediction.malformed[0].reason == "no JSON object in the reply"


# --- the zero-shot baseline -----------------------------------------------------------


def test_zeroshot_reads_a_reply_into_a_prediction(drawing, tmp_path):
    provider = ScriptedProvider(responses=[_reply(drawing, 3)])
    result = BASELINES["vlm_zeroshot"](
        drawing.paths["png"],
        provider=provider,
        model="test-model",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
        sheet=drawing.drawing.sheet.size,
        projection=drawing.drawing.projection,
    )
    assert len(result.prediction.characteristics) == 3
    assert result.prediction.meta["model"] == "test-model"
    assert result.cache_misses == 1


def test_the_prompt_carries_the_image_size_and_the_schema(drawing, tmp_path):
    provider = ScriptedProvider(responses=["{}"])
    BASELINES["vlm_zeroshot"](
        drawing.paths["png"],
        provider=provider,
        model="m",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
    )
    prompt = provider.calls[0]["prompt"]
    width, height = drawing.drawing.image_size
    assert f"{width} by {height} pixels" in prompt
    assert "gtol_symbol" in prompt and "datum_refs" in prompt


def test_a_provider_failure_is_recorded_and_does_not_raise(drawing, tmp_path):
    def refuse(_request):
        raise ProviderError("rate limited")

    result = BASELINES["vlm_zeroshot"](
        drawing.paths["png"],
        provider=ScriptedProvider(responses=refuse),
        model="m",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
    )
    assert result.errors == ["rate limited"]
    assert result.prediction.characteristics == []


# --- the structured baseline ----------------------------------------------------------


def test_tiles_cover_the_sheet_and_overlap():
    tiles = tiles_for((1000, 800))
    assert len(tiles) == 4
    assert min(t.box[0] for t in tiles) == 0
    assert max(t.box[2] for t in tiles) == 1000
    assert max(t.box[3] for t in tiles) == 800
    # Neighbouring columns share a strip, so a callout on the seam is whole somewhere.
    left, right = tiles[0], tiles[1]
    assert left.box[2] > right.box[0]
    assert (left.box[2] - right.box[0]) >= TILE_OVERLAP * 500 - 1


def _dimension(id_: int, x: float, nominal: float) -> Characteristic:
    return Characteristic(
        id=id_, kind="dimension", view="front",
        bbox=[x, 10.0, x + 40.0, 30.0], dim_type="linear", nominal=nominal,
    )


def test_the_vote_keeps_agreement_and_drops_a_lone_hallucination():
    """Three samples, one of which invents a callout the others never saw."""
    shared = [_dimension(1, 0, 10.0), _dimension(2, 200, 20.0)]
    runs = [
        list(shared),
        [c.model_copy(deep=True) for c in shared],
        [*[c.model_copy(deep=True) for c in shared], _dimension(3, 600, 99.0)],
    ]
    merged = merge_by_vote(runs)
    assert [c.nominal for c in merged] == [10.0, 20.0]
    assert [c.id for c in merged] == [1, 2]


def test_one_talkative_sample_cannot_outvote_the_others():
    """A cluster takes at most one reading per run, so three copies from one sample are one
    vote. Without that rule a model that repeats itself would beat one that agrees."""
    repeated = [_dimension(i + 1, 0, 10.0) for i in range(3)]
    merged = merge_by_vote([repeated, [_dimension(1, 600, 42.0)]], min_votes=2)
    assert merged == []


def test_a_unanimous_reading_survives_a_single_sample_run():
    merged = merge_by_vote([[_dimension(1, 0, 10.0)]])
    assert len(merged) == 1


def test_structured_translates_tile_coordinates_back_into_the_sheet(drawing, tmp_path):
    """A callout read inside a crop must come back at its place on the sheet. Getting this
    wrong would put every tiled reading in the top-left quadrant, where a few would match by
    accident -- which is worse than none of them matching."""
    seen: list[dict] = []

    def respond(request):
        seen.append(request)
        # Answer only for the crops, and always at the crop's own origin.
        if "crop from a larger" not in request["prompt"]:
            return "{}"
        return json.dumps(
            {
                "characteristics": [
                    {"id": 1, "kind": "dimension", "view": "front",
                     "bbox": [0, 0, 30, 15], "dim_type": "linear", "nominal": 10.0}
                ]
            }
        )

    result = BASELINES["vlm_structured"](
        drawing.paths["png"],
        provider=ScriptedProvider(responses=respond),
        model="m",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
        samples=1,
    )
    origins = {(round(c.bbox[0]), round(c.bbox[1])) for c in result.prediction.characteristics}
    assert len(origins) == 4, "the four crops must land at four different places"
    assert origins != {(0, 0)}
    width, height = drawing.drawing.image_size
    assert any(x > width / 3 or y > height / 3 for x, y in origins)


def test_structured_asks_once_per_tile_per_sample(drawing, tmp_path):
    provider = ScriptedProvider(responses=["{}"])
    BASELINES["vlm_structured"](
        drawing.paths["png"],
        provider=provider,
        model="m",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
        samples=2,
    )
    # Two samples of (one full sheet + four crops), and none may be served from the cache:
    # a different sample index is a different request.
    assert len(provider.calls) == 2 * 5


# --- the runner -------------------------------------------------------------------------


def test_the_runner_writes_a_prediction_and_a_manifest(drawing, tmp_path):
    provider = ScriptedProvider(responses=[_reply(drawing, 2)])
    manifest = run_baseline(
        "vlm_zeroshot",
        [drawing.paths["json"]],
        tmp_path / "preds",
        provider=provider,
        provider_name="scripted",
        model="test-model-2026-01-01",
        cache=ResponseCache(root=tmp_path / "cache"),
    )
    assert manifest.n_drawings == 1 and manifest.n_failed == 0
    assert manifest.model == "test-model-2026-01-01"
    assert manifest.prompt_variant == "zeroshot-v1"
    assert manifest.started_at and manifest.finished_at

    written = tmp_path / "preds" / f"{drawing.drawing_id}.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["drawing_id"] == drawing.drawing_id
    assert len(payload["characteristics"]) == 2
    assert json.loads((tmp_path / "preds" / "manifest.json").read_text(encoding="utf-8"))


def test_the_runner_resumes_rather_than_repaying_for_a_finished_drawing(drawing, tmp_path):
    kwargs = dict(
        provider_name="scripted",
        model="m",
        cache=ResponseCache(root=tmp_path / "cache"),
    )
    provider = ScriptedProvider(responses=[_reply(drawing, 1)])
    run_baseline(
        "vlm_zeroshot", [drawing.paths["json"]], tmp_path / "preds",
        provider=provider, **kwargs,
    )
    second = ScriptedProvider(responses=[_reply(drawing, 1)])
    manifest = run_baseline(
        "vlm_zeroshot", [drawing.paths["json"]], tmp_path / "preds",
        provider=second, **kwargs,
    )
    assert second.calls == [], "a finished drawing was asked for again"
    assert manifest.n_drawings == 1


def test_the_runner_never_shows_the_model_the_answer_overlay(drawing):
    """The overlay has the ground-truth boxes painted on it. Handing it to a model would be
    showing it the answer key, and the resulting number would be meaningless."""
    chosen = image_for(drawing.paths["json"])
    assert chosen is not None
    assert not chosen.stem.endswith("_overlay")
    assert chosen == drawing.paths["png"]


def test_the_zeroshot_control_never_votes(drawing, tmp_path):
    """Voting is the structured baseline's treatment. If it leaked into the control the
    comparison between the two would no longer isolate anything."""
    provider = ScriptedProvider(responses=[_reply(drawing, 1)])
    manifest = run_baseline(
        "vlm_zeroshot", [drawing.paths["json"]], tmp_path / "preds",
        provider=provider, provider_name="scripted", model="m",
        cache=ResponseCache(root=tmp_path / "cache"), samples=5,
    )
    assert manifest.samples == 1
    assert len(provider.calls) == 1


def test_an_unknown_baseline_is_refused(drawing, tmp_path):
    with pytest.raises(KeyError, match="unknown baseline"):
        run_baseline(
            "vlm_telepathy", [], tmp_path, provider=ScriptedProvider(),
            provider_name="scripted", model="m",
            cache=ResponseCache(root=tmp_path / "cache"),
        )


# --- provider routing -------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-5", "anthropic"),
        ("gpt-5", "openai"),
        ("gemini-3-pro", "gemini"),
    ],
)
def test_a_model_string_routes_to_its_provider(model, expected):
    assert provider_for_model(model) == expected


def test_an_unrecognised_model_is_refused_rather_than_guessed():
    """Defaulting would surface as an authentication error three layers down, by which point
    the cause is no longer obvious."""
    with pytest.raises(ProviderError, match="cannot tell which provider"):
        provider_for_model("llama-in-a-trenchcoat")
