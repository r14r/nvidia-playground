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


def render_runtime_settings(*, supports_streaming: bool = True) -> None:
    if not supports_streaming:
        st.session_state["streaming"] = False

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        key="temperature",
    )
    # NVIDIA NIM endpoints may reject top_p=0. Normalize stale
    # session values before rendering the widget.
    if float(st.session_state.get("top_p", 0.7)) <= 0:
        st.session_state["top_p"] = 0.05

    st.slider(
        "Top P",
        min_value=0.05,
        max_value=1.0,
        step=0.05,
        key="top_p",
        help="NVIDIA NIM requires Top P to be greater than 0 and at most 1.",
    )
    st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=32768,
        step=128,
        key="max_tokens",
    )
    st.toggle(
        "Streaming",
        key="streaming",
        disabled=not supports_streaming,
        help=(
            "Receive and display the model response incrementally as chunks "
            "arrive. Disable this to wait for the complete response before "
            "showing it."
        ),
    )
    st.toggle(
        "Raw chunk data",
        key="show_raw_chunks",
        help=(
            "Keep and display the original low-level streaming chunk payloads "
            "for debugging and protocol inspection. This does not enable "
            "streaming by itself."
        ),
    )


def response_filename(prefix: str = "nvidia-response") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
