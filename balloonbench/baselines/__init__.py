"""Extraction baselines (SPEC.md section 11).

Nothing here imports a vendor SDK at module level: the optional ``baselines`` extra installs
three clients and a machine with one of them must still be able to run that one.
"""

from balloonbench.baselines.base import BaselineResult, PromptConfig, extract_json
from balloonbench.baselines.cache import CacheKey, ResponseCache
from balloonbench.baselines.providers import (
    PROVIDERS,
    ProviderError,
    ScriptedProvider,
    get_provider,
    provider_for_model,
)
from balloonbench.baselines.run import BASELINES, RunManifest, run_baseline

__all__ = [
    "BASELINES",
    "PROVIDERS",
    "BaselineResult",
    "CacheKey",
    "PromptConfig",
    "ProviderError",
    "ResponseCache",
    "RunManifest",
    "ScriptedProvider",
    "extract_json",
    "get_provider",
    "provider_for_model",
    "run_baseline",
]
