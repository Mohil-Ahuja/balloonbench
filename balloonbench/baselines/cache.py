"""A disk cache for model responses, keyed by everything that could change the answer.

SPEC.md section 11 requires it in one sentence: *cache all API responses to disk keyed by
(model, image hash, prompt hash) so re-running the eval costs nothing.* The sentence is
short and the consequence is not. A benchmark whose numbers cost money to reproduce is a
benchmark nobody reproduces, and the leaderboard becomes a claim rather than a result.

The key is a hash of the whole request, not of a summary of it. Anything that could change
the response has to be in the key or the cache will confidently return the wrong answer:
the model string, the exact prompt text, the image bytes, the decoding temperature, and the
sample index for a self-consistency pass. Leaving temperature out would be the subtle one --
a k=3 vote at temperature 1.0 must not be served three copies of the same cached reply, or
the self-consistency measurement silently becomes a measurement of nothing.

Entries are plain JSON files under a two-level fan-out directory, one per request. Not a
single index file: runs are long, interrupted often, and executed in parallel, and a
process killed mid-write must cost one response rather than the whole cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CacheKey", "ResponseCache", "file_digest", "text_digest"]

DEFAULT_CACHE_DIR = Path("data/cache/baselines")


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, read in chunks so a large PNG is not held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """Everything about a request that could change its response."""

    model: str
    prompt_hash: str
    image_hash: str
    temperature: float = 0.0
    #: Which draw of a multi-sample pass this is. Two samples at the same temperature are
    #: different requests, and must not share an entry.
    sample: int = 0
    #: Anything else a baseline varies: a tile's coordinates, a tool-schema version.
    extra: tuple[tuple[str, str], ...] = ()

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": self.prompt_hash,
                "image": self.image_hash,
                "temperature": round(self.temperature, 6),
                "sample": self.sample,
                "extra": list(self.extra),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ResponseCache:
    """Read-through cache of raw model responses.

    ``hits`` and ``misses`` are counted because they belong in the run's metadata: a report
    that says "1000 calls, 998 cache hits" tells a reader the numbers were reproduced rather
    than regenerated, which is a different claim about the same table.
    """

    root: Path = field(default_factory=lambda: Path(DEFAULT_CACHE_DIR))
    #: When false the cache is read but never written, for a dry run that must not pollute
    #: a shared cache directory with responses from an experimental prompt.
    write: bool = True
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def path_for(self, key: CacheKey) -> Path:
        digest = key.digest()
        # Two levels of fan-out. A single directory with a hundred thousand entries is slow
        # to list on every filesystem and painful on Windows in particular.
        return self.root / digest[:2] / digest[2:4] / f"{digest}.json"

    def get(self, key: CacheKey) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated entry from a killed process. Treat it as absent rather than
            # failing the run: the cost of a re-request is one call.
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, key: CacheKey, response: dict[str, Any]) -> None:
        if not self.write:
            return
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": {
                "model": key.model,
                "prompt_hash": key.prompt_hash,
                "image_hash": key.image_hash,
                "temperature": key.temperature,
                "sample": key.sample,
                "extra": list(key.extra),
            },
            "response": response,
        }
        # Written to a temporary file in the same directory and moved into place, so a
        # reader never sees a half-written entry and an interrupted run leaves no corruption.
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(payload, file)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
