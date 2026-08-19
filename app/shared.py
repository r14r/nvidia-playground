from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from lib.nvidia import NVIDIA, NVIDIAError

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"

APP_VERSION = (
    VERSION_FILE.read_text(encoding="utf-8").strip()
    if VERSION_FILE.is_file()
    else "unknown"
)

load_dotenv(PROJECT_ROOT / ".env")

_models_value = os.getenv("NVIDIA_MODELS_FILE", "models.json").strip() or "models.json"
MODELS_FILE = Path(_models_value)
if not MODELS_FILE.is_absolute():
    MODELS_FILE = PROJECT_ROOT / MODELS_FILE


@st.cache_resource(show_spinner=False)
def _cached_nvidia(path: str, mtime_ns: int) -> NVIDIA:
    # mtime_ns is deliberately part of the cache key. A newly generated
    # models.json is picked up automatically on the next Streamlit rerun.
    return NVIDIA(path)


def get_nvidia() -> NVIDIA:
    if not MODELS_FILE.is_file():
        raise NVIDIAError(
            f"Model catalog not found: {MODELS_FILE}. Run `just models` first."
        )

    return _cached_nvidia(
        str(MODELS_FILE),
        MODELS_FILE.stat().st_mtime_ns,
    )


def reload_nvidia() -> None:
    st.cache_resource.clear()


def model_ids(nvidia: NVIDIA) -> list[str]:
    return [
        str(model.get("id") or model.get("model"))
        for model in nvidia.models(chat_only=True, safe=True)
    ]


def ensure_settings(nvidia: NVIDIA) -> None:
    ids = model_ids(nvidia)
    if not ids:
        raise NVIDIAError("models.json contains no chat-capable models.")

    selected = st.session_state.get("selected_model")
    if selected not in ids:
        default = nvidia.default_model()
        st.session_state["selected_model"] = default if default in ids else ids[0]

    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.7)
    # Reasoning-capable NVIDIA models can spend a large part of the
    # generation budget on reasoning before emitting final answer content.
    # NVIDIA's hosted examples commonly use 16384 tokens for these models.
    current_max_tokens = st.session_state.get("max_tokens")
    if current_max_tokens is None or current_max_tokens == 2048:
        st.session_state["max_tokens"] = 16384
    st.session_state.setdefault("streaming", True)
    st.session_state.setdefault("show_raw_chunks", False)
    st.session_state.setdefault(
        "system_prompt",
        "You are a helpful technical assistant.",
    )
    st.session_state.setdefault(
        "prompt",
        "Explain HTTP streaming in Python and show a small practical example.",
    )

    info = nvidia.model(st.session_state["selected_model"], safe=True)
    capabilities = info.get("capabilities") or {}
    if capabilities.get("streaming", True) is False:
        st.session_state["streaming"] = False


def selected_model_info(nvidia: NVIDIA) -> dict[str, Any]:
    ensure_settings(nvidia)
    return nvidia.model(st.session_state["selected_model"], safe=True)


def require_nvidia() -> NVIDIA:
    try:
        nvidia = get_nvidia()
        ensure_settings(nvidia)
        return nvidia
    except NVIDIAError as exc:
        st.error(str(exc))
        st.info(
            "Generate the catalog with "
            "`nvidia-cli models --list --details --with-api-key "
            "--json --save models.json`."
        )
        st.stop()
        raise
