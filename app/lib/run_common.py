from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import streamlit as st


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def render_prompt_tabs() -> None:
    system_prompt_tab, user_prompt_tab = st.tabs(
        ["System Prompt", "User Prompt"]
    )
    with system_prompt_tab:
        st.text_area(
            "System Prompt",
            key="system_prompt",
            height=180,
            label_visibility="collapsed",
        )
    with user_prompt_tab:
        st.text_area(
            "User Prompt",
            key="prompt",
            height=220,
            label_visibility="collapsed",
        )


def render_runtime_settings(
    *,
    supports_streaming: bool = True,
    include_run_parallel: bool = False,
) -> None:
    if not supports_streaming:
        st.session_state["streaming"] = False

    if float(st.session_state.get("top_p", 0.7)) <= 0:
        st.session_state["top_p"] = 0.05

    st.markdown("**Settings**")

    temperature_tab, max_tokens_tab, top_p_tab, timeout_tab = st.tabs(
        ["Temperature", "Max Tokens", "Top P", "Timeout"]
    )

    with temperature_tab:
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="temperature",
            label_visibility="collapsed",
        )

    with max_tokens_tab:
        st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=32768,
            step=128,
            key="max_tokens",
            label_visibility="collapsed",
            help=(
                "Maximum number of generated tokens. "
                "The default is 2048."
            ),
        )

    with top_p_tab:
        st.slider(
            "Top P",
            min_value=0.05,
            max_value=1.0,
            step=0.05,
            key="top_p",
            label_visibility="collapsed",
            help=(
                "NVIDIA NIM requires Top P to be greater than 0 "
                "and at most 1."
            ),
        )

    with timeout_tab:
        st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=3600,
            step=30,
            key="timeout_seconds",
            label_visibility="collapsed",
            help=(
                "Maximum time to wait for provider network operations. "
                "The default is 300 seconds."
            ),
        )

    st.markdown("**Optionen**")

    st.toggle(
        "Streaming",
        key="streaming",
        disabled=not supports_streaming,
        help=(
            "Receive and display the model response incrementally "
            "as chunks arrive."
        ),
    )
    st.toggle(
        "Raw chunk data",
        key="show_raw_chunks",
        help=(
            "Keep and display the original low-level streaming chunk "
            "payloads for debugging and protocol inspection."
        ),
    )

    if include_run_parallel:
        st.toggle(
            "Run Parallel",
            key="run_parallel",
            help=(
                "Run selected models concurrently in separate worker "
                "threads. No asyncio execution is used."
            ),
        )


def response_filename(prefix: str = "nvidia-response") -> str:
    return (
        f"{prefix}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
