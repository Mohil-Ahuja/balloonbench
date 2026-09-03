"""One interface over the three vision APIs, plus a fake for the tests.

SPEC.md section 11 asks for the same baseline run across Claude, GPT and Gemini. Their SDKs
disagree about almost everything -- how an image is attached, what a system prompt is, how
a tool call comes back -- so the differences are absorbed here and every baseline above this
line sees one method: given a prompt and an image, return text.

Three rules this module exists to enforce.

**No SDK is imported until it is used.** The optional ``baselines`` extra installs three
clients; a machine with one of them must still be able to run that one. Importing at module
level would make ``balloonbench.baselines`` unimportable without all three, and would make
the test suite depend on packages it never calls.

**No provider is constructed without an explicit model string.** SPEC.md requires exact
model versions pinned in the results file, and a default that silently tracks a vendor's
"latest" alias would make two runs of the same command incomparable while both claimed the
same model. The alias is accepted if a caller insists, but it is recorded verbatim as given.

**The fake provider is a first-class citizen.** Every test in this milestone runs against
:class:`ScriptedProvider`, which returns canned responses and records what it was asked. A
harness that can only be tested by spending money is a harness that does not get tested.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "PROVIDERS",
    "Provider",
    "ProviderError",
    "ScriptedProvider",
    "get_provider",
    "provider_for_model",
]


class ProviderError(RuntimeError):
    """A call could not be made or came back unusable."""


class Provider(Protocol):
    """What a baseline needs from a model. Text in and an image in, text out."""

    name: str

    def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        image_path: str | Path | None,
        temperature: float,
        max_tokens: int,
    ) -> str: ...


def _image_block(image_path: str | Path) -> tuple[str, str]:
    """``(media type, base64 payload)`` for an image on disk."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix)
    if media is None:
        raise ProviderError(f"unsupported image type {suffix!r} for {path}")
    return media, base64.standard_b64encode(path.read_bytes()).decode("ascii")


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    _client: Any = field(default=None, repr=False)

    def _ensure(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ProviderError(
                    "the anthropic SDK is not installed; pip install -e '.[baselines]'"
                ) from exc
            if not os.environ.get(self.api_key_env):
                raise ProviderError(f"{self.api_key_env} is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self, *, model, system, prompt, image_path, temperature, max_tokens
    ) -> str:
        client = self._ensure()
        content: list[dict[str, Any]] = []
        if image_path is not None:
            media, data = _image_block(image_path)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": data},
                }
            )
        content.append({"type": "text", "text": prompt})
        message = client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


@dataclass
class OpenAIProvider:
    name: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    _client: Any = field(default=None, repr=False)

    def _ensure(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ProviderError(
                    "the openai SDK is not installed; pip install -e '.[baselines]'"
                ) from exc
            if not os.environ.get(self.api_key_env):
                raise ProviderError(f"{self.api_key_env} is not set")
            self._client = openai.OpenAI()
        return self._client

    def complete(
        self, *, model, system, prompt, image_path, temperature, max_tokens
    ) -> str:
        client = self._ensure()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            media, data = _image_block(image_path)
            content.insert(
                0,
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}},
            )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content or ""


@dataclass
class GeminiProvider:
    name: str = "gemini"
    api_key_env: str = "GOOGLE_API_KEY"
    _client: Any = field(default=None, repr=False)

    def _ensure(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ProviderError(
                    "the google-genai SDK is not installed; pip install -e '.[baselines]'"
                ) from exc
            if not os.environ.get(self.api_key_env):
                raise ProviderError(f"{self.api_key_env} is not set")
            self._client = genai.Client()
        return self._client

    def complete(
        self, *, model, system, prompt, image_path, temperature, max_tokens
    ) -> str:
        client = self._ensure()
        from google.genai import types

        parts: list[Any] = []
        if image_path is not None:
            media, _ = _image_block(image_path)
            parts.append(
                types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=media)
            )
        parts.append(types.Part.from_text(text=prompt))
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""


@dataclass
class ScriptedProvider:
    """A provider that returns prepared answers. The tests' model.

    ``responses`` may be a list, consumed in order and then repeated from the last entry, or
    a callable taking the request keywords. Every call is appended to ``calls``, which is
    what lets a test assert the thing that actually matters about the cache: that the second
    identical request never reached a provider at all.
    """

    responses: Any = field(default_factory=list)
    name: str = "scripted"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self, *, model, system, prompt, image_path, temperature, max_tokens
    ) -> str:
        request = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "image_path": None if image_path is None else str(image_path),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.calls.append(request)
        if callable(self.responses):
            return self.responses(request)
        if not self.responses:
            return ""
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


#: Constructors, not instances: a provider builds its client lazily on first use, and one
#: that is never called must never look for an API key.
PROVIDERS: dict[str, Any] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise ProviderError(
            f"unknown provider {name!r}; known: {sorted(PROVIDERS)}"
        ) from None


def provider_for_model(model: str) -> str:
    """Guess the provider from a model string, so a caller need only name the model.

    A guess, and it says so by raising when it cannot tell rather than defaulting. Silently
    routing an unrecognised model to one vendor would produce an authentication error three
    layers down, at which point the cause is no longer obvious.
    """
    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if lowered.startswith("gemini"):
        return "gemini"
    raise ProviderError(
        f"cannot tell which provider serves {model!r}; pass --provider explicitly"
    )
