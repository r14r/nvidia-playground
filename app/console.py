from __future__ import annotations

from datetime import datetime


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _print_block(label: str, value: str) -> None:
    print(f"   {label}: ---", flush=True)
    if value:
        print(value, flush=True)
    else:
        print("<empty>", flush=True)
    print("   ---", flush=True)


def selected_model(model: str) -> None:
    print(
        f"{_timestamp()} Selected Model: {model}",
        flush=True,
    )


def run_prompt(
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
) -> None:
    print(
        f"{_timestamp()} Run Prompt on Model: {model}",
        flush=True,
    )
    _print_block("System Prompt", system_prompt)
    _print_block("User Prompt", user_prompt)


def result(
    model: str,
    text: str,
) -> None:
    print(
        f"{_timestamp()} Result from Prompt on Model: {model}",
        flush=True,
    )
    _print_block("Result", text)


def error(
    model: str,
    exc: Exception | str,
    *,
    partial_response: str = "",
) -> None:
    print(
        f"{_timestamp()} ERROR from Prompt on Model: {model}",
        flush=True,
    )
    print(f"   Error: {exc}", flush=True)

    if partial_response:
        _print_block("Partial Result", partial_response)
