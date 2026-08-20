from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


_MODEL_PAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 nvidia-playground/0.8.2",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

_BLOCK_TAGS = {
    "article",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "li",
    "main",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self._in_title = False
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        attributes = {
            key.lower(): value
            for key, value in attrs
            if value is not None
        }
        if tag in {"script", "style", "svg"}:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (
                attributes.get("name")
                or attributes.get("property")
                or ""
            ).lower()
            content = attributes.get("content") or ""
            if key and content:
                self.meta[key] = html.unescape(content).strip()
        if tag in _BLOCK_TAGS and self._ignored_depth == 0:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS and self._ignored_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = html.unescape(data).strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        self.text_parts.append(value + " ")

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def text(self) -> str:
        raw = "".join(self.text_parts)
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in raw.splitlines()
        ]
        return "\n".join(line for line in lines if line)


def nvidia_modelcard_url(model: str) -> str:
    value = model.strip().strip("/")
    if "/" in value:
        owner, model_name = value.split("/", 1)
    else:
        owner, model_name = "nvidia", value
    return (
        "https://build.nvidia.com/"
        f"{quote(owner, safe='._-')}/"
        f"{quote(model_name, safe='._-')}/modelcard"
    )


def ollama_library_url(model: str) -> str:
    return (
        "https://ollama.com/library/"
        f"{quote(model.strip(), safe='._-')}"
    )


def model_details_url(provider: str, model: str) -> str | None:
    if provider == "NVIDIA":
        return nvidia_modelcard_url(model)
    if provider == "Ollama":
        return ollama_library_url(model)
    return None


def _fetch_page(url: str, timeout_seconds: float = 20.0) -> _PageParser:
    request = Request(url, headers=_MODEL_PAGE_HEADERS)
    with urlopen(request, timeout=timeout_seconds) as response:
        document = response.read().decode("utf-8", errors="replace")
    parser = _PageParser()
    parser.feed(document)
    return parser


def _field(text: str, labels: tuple[str, ...]) -> str:
    lines = text.splitlines()
    lower_labels = tuple(label.lower() for label in labels)
    for index, line in enumerate(lines):
        normalized = line.lower()
        for label, lower_label in zip(labels, lower_labels):
            if normalized.startswith(lower_label):
                remainder = line[len(label):].lstrip(" :–-\t")
                if remainder:
                    return remainder[:600]
                if index + 1 < len(lines):
                    return lines[index + 1][:600]
    return ""


def _excerpt(text: str, markers: tuple[str, ...]) -> str:
    lines = text.splitlines()
    start = 0
    for marker in markers:
        for index, line in enumerate(lines):
            if marker.lower() in line.lower():
                start = index + 1
                break
        if start:
            break
    selected = []
    for line in lines[start:]:
        if line in {"Try API", "Deploy", "API Reference"}:
            continue
        selected.append(line)
        if sum(len(item) for item in selected) >= 2400:
            break
    return "\n\n".join(selected).strip()[:2600]


def load_remote_model_details(
    provider: str,
    model: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    url = model_details_url(provider, model)
    if not url:
        return {}

    parser = _fetch_page(url, timeout_seconds=timeout_seconds)
    text = parser.text()
    description = (
        parser.meta.get("description")
        or parser.meta.get("og:description")
        or parser.meta.get("twitter:description")
        or ""
    )

    details: dict[str, Any] = {
        "Provider": provider,
        "Model": model,
        "Source URL": url,
        "Page Title": parser.title(),
        "Description": description,
    }

    if provider == "NVIDIA":
        field_map = {
            "Model Developer": ("Model Developer",),
            "Model Dates": ("Model Dates",),
            "Data Freshness": ("Data Freshness",),
            "Total Parameters": ("Total Parameters",),
            "Architecture": ("Architecture", "Architecture Type"),
            "Context Length": ("Context Length",),
            "Minimum GPU Requirement": ("Minimum GPU Requirement",),
            "Supported Languages": ("Supported Languages",),
            "Recommended Sampling": ("Recommended Sampling",),
            "Best For": ("Best For",),
            "Reasoning Mode": ("Reasoning Mode",),
            "License": ("License", "License/Terms of Use"),
            "Release Date": ("Release Date",),
            "Use Case": ("Use Case",),
            "Input": ("Input", "Input Types"),
            "Output": ("Output", "Output Types"),
        }
        for key, labels in field_map.items():
            value = _field(text, labels)
            if value:
                details[key] = value
        details["Model Card Excerpt"] = _excerpt(
            text,
            ("Model Overview", "Description"),
        )

    elif provider == "Ollama":
        details["Library Excerpt"] = _excerpt(
            text,
            ("Readme", "Details"),
        )

    return {
        key: value
        for key, value in details.items()
        if value not in {None, ""}
    }
