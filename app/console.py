from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def header(mode: str, title: str) -> None:
    print(
        f"{_timestamp()} [{mode}] ===== {title} =====",
        flush=True,
    )


def step(
    mode: str,
    number: int,
    message: str,
    **details: Any,
) -> None:
    """Print execution metadata only; never response/chunk content."""
    rendered = " ".join(
        f"{key}={value}"
        for key, value in details.items()
        if value is not None
    )
    suffix = f" | {rendered}" if rendered else ""

    print(
        f"{_timestamp()} [{mode}] [{number:02d}] {message}{suffix}",
        flush=True,
    )


def error(
    mode: str,
    message: str,
    *,
    exc: Exception | str,
    response: str = "",
    chunks: list[dict[str, Any]] | None = None,
    raw_chunks: list[Any] | None = None,
) -> None:
    """Print response/chunk content only when a request failed."""
    print(
        f"{_timestamp()} [{mode}] [ERROR] {message}: {exc}",
        flush=True,
    )

    if response:
        print(
            f"{_timestamp()} [{mode}] [ERROR] Partial response follows:",
            flush=True,
        )
        print(response, flush=True)

    if chunks:
        print(
            f"{_timestamp()} [{mode}] [ERROR] Chunk log follows:",
            flush=True,
        )
        print(
            json.dumps(
                chunks,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )

    if raw_chunks:
        print(
            f"{_timestamp()} [{mode}] [ERROR] Raw chunks follow:",
            flush=True,
        )
        print(
            json.dumps(
                raw_chunks,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )
