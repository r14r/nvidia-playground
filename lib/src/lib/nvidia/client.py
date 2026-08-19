from __future__ import annotations

import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI

from .models import Model, ModelCatalog, model_id, sanitize_secrets


class NVIDIA:
    """NVIDIA NIM client configured from nvidia-cli models.json."""

    def __init__(self, models_file: str | Path = "models.json"):
        self.models_file = Path(models_file)
        self.catalog = ModelCatalog(self.models_file)

    def reload(self) -> None:
        self.catalog.reload()

    def models(self, *, chat_only: bool = True, safe: bool = True) -> list[Model]:
        items = self.catalog.list_models(chat_only)
        return [sanitize_secrets(m) for m in items] if safe else items

    def model(self, identifier: str | None = None, *, safe: bool = True) -> Model:
        model = self.catalog.resolve(identifier, chat_only=True)
        return sanitize_secrets(model) if safe else model

    def default_model(self) -> str:
        return model_id(self.catalog.default())

    def resolve_model(self, identifier: str | None = None) -> str:
        return model_id(self.catalog.resolve(identifier))

    def set_default_model(self, identifier: str) -> str:
        """Set and persist the catalog default model."""
        return model_id(self.catalog.set_default(identifier))

    def can_run(self, identifier: str | None = None) -> bool:
        try:
            self.catalog.endpoint(self.catalog.resolve(identifier))
            return True
        except Exception:
            return False

    def _client_for(self, identifier: str | None = None):
        model = self.catalog.resolve(identifier)
        base_url, api_key = self.catalog.endpoint(model)
        return model, OpenAI(base_url=base_url, api_key=api_key)

    def _async_client_for(self, identifier: str | None = None):
        model = self.catalog.resolve(identifier)
        base_url, api_key = self.catalog.endpoint(model)
        return model, AsyncOpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _messages(
        prompt: str,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if messages is not None:
            return messages
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        result.append({"role": "user", "content": prompt})
        return result

    @staticmethod
    def _request_options(
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build provider-safe OpenAI-compatible completion options."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **extra,
        }
        if top_p > 0:
            options["top_p"] = min(top_p, 1.0)
        return options

    @staticmethod
    def _content_text(content: Any) -> str:
        """Normalize OpenAI-compatible text content into one string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _completion_text(completion: Any) -> str:
        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError(
                "Model returned no completion choices. "
                "The provider response contained no usable result."
            )
        choice = next(iter(choices), None)
        if choice is None:
            raise RuntimeError("Model returned no completion choices.")
        message = getattr(choice, "message", None)
        if message is None:
            raise RuntimeError("Model returned a completion choice without a message.")
        return NVIDIA._content_text(getattr(message, "content", None))

    def query(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None, messages: list[dict[str, Any]] | None = None, temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 16384, **kwargs: Any) -> str:
        info, client = self._client_for(model)
        completion = client.chat.completions.create(model=model_id(info), messages=self._messages(prompt, system_prompt, messages), **self._request_options(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=False, extra=kwargs))
        return self._completion_text(completion)

    async def async_query(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None, messages: list[dict[str, Any]] | None = None, temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 16384, **kwargs: Any) -> str:
        info, client = self._async_client_for(model)
        completion = await client.chat.completions.create(model=model_id(info), messages=self._messages(prompt, system_prompt, messages), **self._request_options(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=False, extra=kwargs))
        return self._completion_text(completion)

    def stream(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None, messages: list[dict[str, Any]] | None = None, temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 16384, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_events(prompt, model=model, system_prompt=system_prompt, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, include_raw=False, **kwargs):
            if event["content"]:
                yield event["content"]

    def stream_events(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None, messages: list[dict[str, Any]] | None = None, temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 16384, include_raw: bool = False, **kwargs: Any) -> Iterator[dict[str, Any]]:
        info, client = self._client_for(model)
        started = time.perf_counter(); previous = started
        completion = client.chat.completions.create(model=model_id(info), messages=self._messages(prompt, system_prompt, messages), **self._request_options(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=True, extra=kwargs))
        for number, chunk in enumerate(completion, start=1):
            now = time.perf_counter(); choices = getattr(chunk, "choices", None); choice = next(iter(choices), None) if choices else None
            if choice is None:
                previous = now; continue
            delta = getattr(choice, "delta", None)
            content = self._content_text(getattr(delta, "content", None)) if delta is not None else ""
            event = {"chunk": number, "content": content, "chars": len(content), "elapsed_ms": round((now-started)*1000,1), "gap_ms": round((now-previous)*1000,1), "finish_reason": getattr(choice,"finish_reason",None)}
            if include_raw:
                dump = getattr(chunk, "model_dump", None); event["raw"] = dump() if callable(dump) else None
            previous = now; yield event

    async def async_stream_events(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None, messages: list[dict[str, Any]] | None = None, temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 16384, include_raw: bool = False, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        info, client = self._async_client_for(model)
        started = time.perf_counter(); previous = started
        completion = await client.chat.completions.create(model=model_id(info), messages=self._messages(prompt, system_prompt, messages), **self._request_options(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=True, extra=kwargs))
        number = 0
        async for chunk in completion:
            number += 1; now = time.perf_counter(); choices = getattr(chunk, "choices", None); choice = next(iter(choices), None) if choices else None
            if choice is None:
                previous = now; continue
            delta = getattr(choice, "delta", None)
            content = self._content_text(getattr(delta, "content", None)) if delta is not None else ""
            event = {"chunk": number, "content": content, "chars": len(content), "elapsed_ms": round((now-started)*1000,1), "gap_ms": round((now-previous)*1000,1), "finish_reason": getattr(choice,"finish_reason",None)}
            if include_raw:
                dump = getattr(chunk, "model_dump", None); event["raw"] = dump() if callable(dump) else None
            previous = now; yield event
