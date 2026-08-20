from __future__ import annotations

import json
from datetime import datetime


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"{_timestamp()} {message}", flush=True)


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def run_prompt(
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
) -> None:
    _log(f"Run Prompt on Model: {model}")
    _log(f"System Prompt: {_quoted(system_prompt)}")
    _log(f"User Prompt: {_quoted(user_prompt)}")


def connect_to_provider(provider: str) -> None:
    _log(f" Connect to {provider}")


def connect_to_nvidia() -> None:
    connect_to_provider("NVIDIA")


def execute_prompt() -> None:
    _log(" Run Prompt")


def waiting_for_response() -> None:
    _log(" Waiting for Response")


def response(text: str) -> None:
    _log(" Response:")
    if text:
        print(text, flush=True)


def error(
    model: str,
    exc: Exception | str,
    *,
    partial_response: str = "",
) -> None:
    _log(f" ERROR from Prompt on Model: {model}")
    _log(f" Error: {exc}")
    if partial_response:
        _log(" Partial Response:")
        print(partial_response, flush=True)
