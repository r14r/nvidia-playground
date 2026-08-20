from __future__ import annotations

import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

PROVIDERS = ("Ollama", "NVIDIA", "models.json")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODELS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

NVIDIA_MODELS_URL = (
    "https://build.nvidia.com/models"
    "?filters=nimType%3Anim_type_preview&pageSize=200"
)

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 nvidia-playground/0.8.0",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}


def _request(url: str, *, timeout: float = 15.0) -> bytes:
    request = Request(url, headers=_DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Any:
    headers = dict(_DEFAULT_HEADERS)
    headers["Accept"] = "application/json"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_ollama_models() -> list[dict[str, Any]]:
    payload = _request_json(OLLAMA_MODELS_URL, timeout=10.0)
    rows: list[dict[str, Any]] = []
    for item in payload.get("models", []):
        details = item.get("details") or {}
        rows.append(
            {
                "Provider": "Ollama",
                "Model": item.get("model") or item.get("name") or "",
                "Family": details.get("family") or "",
                "Parameter Size": details.get("parameter_size") or "",
                "Quantization": details.get("quantization_level") or "",
                "Size": item.get("size") or 0,
                "Modified": item.get("modified_at") or "",
                "Source": OLLAMA_MODELS_URL,
            }
        )
    return sorted(rows, key=lambda row: str(row["Model"]).lower())


def list_models_json(models_file: Path) -> list[dict[str, Any]]:
    if not models_file.is_file():
        raise FileNotFoundError(f"models.json not found: {models_file}")

    payload = json.loads(models_file.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for item in payload.get("models", []):
        capabilities = item.get("capabilities") or {}
        model_id = item.get("id") or item.get("model") or ""
        rows.append(
            {
                "Provider": "models.json",
                "Model": model_id,
                "Type": item.get("type") or "",
                "Base URL": item.get("base_url") or "",
                "Default": bool(item.get("default")),
                "Chat": capabilities.get("chat"),
                "Streaming": capabilities.get("streaming"),
                "Configured": bool(item.get("base_url") and item.get("api_key")),
                "Source": str(models_file),
            }
        )

    return sorted(rows, key=lambda row: str(row["Model"]).lower())


class _NvidiaLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.add(html.unescape(href))


def _nvidia_model_id_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None

    owner, model = parts[:2]
    blocked = {
        "models",
        "docs",
        "blog",
        "playground",
        "collections",
        "account",
        "search",
        "pricing",
        "support",
        "settings",
    }
    if owner.lower() in blocked:
        return None

    token = re.compile(r"^[A-Za-z0-9._-]+$")
    if not token.match(owner) or not token.match(model):
        return None
    return f"{owner}/{model}".lower()


def _extract_nvidia_model_ids(document: str) -> list[str]:
    candidates: set[str] = set()

    patterns = (
        r'"modelId"\s*:\s*"([^"]+/[^"]+)"',
        r'"model_id"\s*:\s*"([^"]+/[^"]+)"',
        r'"modelName"\s*:\s*"([^"]+/[^"]+)"',
        r'"id"\s*:\s*"([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"',
    )
    for pattern in patterns:
        for match in re.findall(pattern, document, flags=re.IGNORECASE):
            value = html.unescape(match).strip().lower()
            if re.match(r"^[a-z0-9._-]+/[a-z0-9._-]+$", value):
                candidates.add(value)

    parser = _NvidiaLinkParser()
    parser.feed(document)
    for href in parser.links:
        absolute = urljoin("https://build.nvidia.com", href)
        parsed = urlparse(absolute)
        if parsed.netloc != "build.nvidia.com":
            continue
        model_id = _nvidia_model_id_from_path(parsed.path)
        if model_id:
            candidates.add(model_id)

    return sorted(candidates)


def list_nvidia_models() -> list[dict[str, Any]]:
    document = _request(NVIDIA_MODELS_URL, timeout=20.0).decode(
        "utf-8",
        errors="replace",
    )
    model_ids = _extract_nvidia_model_ids(document)
    if not model_ids:
        raise RuntimeError(
            "NVIDIA model catalog was reachable, but no model identifiers "
            "could be extracted from the response."
        )

    return [
        {
            "Provider": "NVIDIA",
            "Model": model_id,
            "URL": f"https://build.nvidia.com/{model_id}",
            "Source": NVIDIA_MODELS_URL,
        }
        for model_id in model_ids
    ]


def list_provider_models(
    provider: str,
    *,
    models_file: Path,
) -> list[dict[str, Any]]:
    if provider == "Ollama":
        return list_ollama_models()
    if provider == "NVIDIA":
        return list_nvidia_models()
    if provider == "models.json":
        return list_models_json(models_file)
    raise ValueError(f"Unsupported provider: {provider}")


def runtime_model_ids(
    provider: str,
    *,
    nvidia: Any | None = None,
) -> list[str]:
    if provider == "Ollama":
        return [
            str(row["Model"])
            for row in list_ollama_models()
            if row.get("Model")
        ]

    if provider in {"NVIDIA", "models.json"}:
        if nvidia is None:
            raise RuntimeError("NVIDIA model configuration is not available.")
        return [
            str(model.get("id") or model.get("model"))
            for model in nvidia.models(chat_only=True, safe=True)
        ]

    raise ValueError(f"Unsupported provider: {provider}")


def provider_can_run(
    provider: str,
    model: str,
    *,
    nvidia: Any | None = None,
) -> bool:
    if provider == "Ollama":
        return bool(model)
    if provider in {"NVIDIA", "models.json"}:
        return bool(nvidia and nvidia.can_run(model))
    return False


def provider_model_info(
    provider: str,
    model: str,
    *,
    nvidia: Any | None = None,
) -> dict[str, Any]:
    if provider == "Ollama":
        for row in list_ollama_models():
            if row.get("Model") == model:
                return dict(row)
        return {"Provider": "Ollama", "Model": model}

    if provider in {"NVIDIA", "models.json"}:
        if nvidia is None:
            return {"Provider": provider, "Model": model}
        info = dict(nvidia.model(model, safe=True))
        info["provider"] = provider
        return info

    return {"Provider": provider, "Model": model}


def provider_supports_streaming(
    provider: str,
    model: str,
    *,
    nvidia: Any | None = None,
) -> bool:
    if provider == "Ollama":
        return True

    if provider in {"NVIDIA", "models.json"} and nvidia is not None:
        info = nvidia.model(model, safe=True)
        capabilities = info.get("capabilities") or {}
        return capabilities.get("streaming", True) is not False

    return False


def _ollama_messages(
    prompt: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def ollama_query(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    payload = _request_json(
        OLLAMA_CHAT_URL,
        payload={
            "model": model,
            "messages": _ollama_messages(prompt, system_prompt),
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        },
    )
    return str((payload.get("message") or {}).get("content") or "")


def ollama_stream_events(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 2048,
    include_raw: bool = False,
) -> Iterator[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": _ollama_messages(prompt, system_prompt),
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    }
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **_DEFAULT_HEADERS,
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    started = time.perf_counter()
    previous = started
    chunk_number = 0

    with urlopen(request, timeout=120.0) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue

            item = json.loads(raw_line.decode("utf-8"))
            now = time.perf_counter()
            chunk_number += 1
            content = str((item.get("message") or {}).get("content") or "")

            event: dict[str, Any] = {
                "chunk": chunk_number,
                "content": content,
                "chars": len(content),
                "elapsed_ms": (now - started) * 1000.0,
                "gap_ms": (now - previous) * 1000.0,
                "finish_reason": "stop" if item.get("done") else None,
            }
            if include_raw:
                event["raw"] = item

            previous = now
            yield event


def provider_query(
    provider: str,
    prompt: str,
    *,
    model: str,
    nvidia: Any | None,
    system_prompt: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    if provider == "Ollama":
        return ollama_query(
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider in {"NVIDIA", "models.json"}:
        if nvidia is None:
            raise RuntimeError("NVIDIA model configuration is not available.")
        return nvidia.query(
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def provider_stream_events(
    provider: str,
    prompt: str,
    *,
    model: str,
    nvidia: Any | None,
    system_prompt: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    include_raw: bool,
) -> Iterator[dict[str, Any]]:
    if provider == "Ollama":
        yield from ollama_stream_events(
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            include_raw=include_raw,
        )
        return

    if provider in {"NVIDIA", "models.json"}:
        if nvidia is None:
            raise RuntimeError("NVIDIA model configuration is not available.")
        yield from nvidia.stream_events(
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            include_raw=include_raw,
        )
        return

    raise ValueError(f"Unsupported provider: {provider}")


def console_provider_name(provider: str) -> str:
    return "NVIDIA" if provider in {"NVIDIA", "models.json"} else provider
