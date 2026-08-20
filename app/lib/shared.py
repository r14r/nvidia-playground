from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

APP_LIB_DIR = Path(__file__).resolve().parent
APP_DIR = APP_LIB_DIR.parent
PROJECT_ROOT = APP_DIR.parent
NVIDIA_LIB_SRC = PROJECT_ROOT / "lib" / "src"

if NVIDIA_LIB_SRC.is_dir() and str(NVIDIA_LIB_SRC) not in sys.path:
    sys.path.insert(0, str(NVIDIA_LIB_SRC))

import streamlit as st
from dotenv import load_dotenv

from lib.nvidia import NVIDIA, NVIDIAError

VERSION_FILE = PROJECT_ROOT / "VERSION"
APP_VERSION = (
    VERSION_FILE.read_text(encoding="utf-8").strip()
    if VERSION_FILE.is_file()
    else "unknown"
)

load_dotenv(PROJECT_ROOT / ".env")

_models_value = (
    os.getenv("NVIDIA_MODELS_FILE", "models.json").strip()
    or "models.json"
)
MODELS_FILE = Path(_models_value)
if not MODELS_FILE.is_absolute():
    MODELS_FILE = PROJECT_ROOT / MODELS_FILE


@st.cache_resource(show_spinner=False)
def _cached_nvidia(path: str, mtime_ns: int) -> NVIDIA:
    return NVIDIA(path)


def get_nvidia() -> NVIDIA:
    if not MODELS_FILE.is_file():
        raise NVIDIAError(
            f"Model catalog not found: {MODELS_FILE}. "
            "Run `just models` first."
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


DEFAULT_SYSTEM_PROMPT = "You are a helpful technical assistant."
DEFAULT_USER_PROMPT = (
    "**Describe advanced Streamlit features in three sentences per feature. "
    "No further information or additional details.**"
)
OLD_DEFAULT_USER_PROMPTS = {
    "Explain HTTP streaming in Python and show a small practical example."
}


def ensure_base_settings() -> None:
    st.session_state.setdefault("run_provider", "models.json")
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.7)
    st.session_state.setdefault("timeout_seconds", 300)
    st.session_state.setdefault("run_parallel", True)

    current_max_tokens = st.session_state.get("max_tokens")
    if not st.session_state.get("_max_tokens_v074_migrated", False):
        if current_max_tokens is None or int(current_max_tokens) == 16384:
            st.session_state["max_tokens"] = 2048
        st.session_state["_max_tokens_v074_migrated"] = True
    st.session_state.setdefault("max_tokens", 2048)

    st.session_state.setdefault("streaming", True)
    st.session_state.setdefault("show_raw_chunks", False)
    st.session_state.setdefault(
        "system_prompt",
        DEFAULT_SYSTEM_PROMPT,
    )

    current_prompt = st.session_state.get("prompt")
    if current_prompt is None or current_prompt in OLD_DEFAULT_USER_PROMPTS:
        st.session_state["prompt"] = DEFAULT_USER_PROMPT


def ensure_settings(nvidia: NVIDIA) -> None:
    ensure_base_settings()

    ids = model_ids(nvidia)
    if not ids:
        raise NVIDIAError(
            "models.json contains no chat-capable models."
        )

    selected = st.session_state.get("selected_model")
    if selected not in ids:
        default = nvidia.default_model()
        st.session_state["selected_model"] = (
            default if default in ids else ids[0]
        )

    info = nvidia.model(
        st.session_state["selected_model"],
        safe=True,
    )
    capabilities = info.get("capabilities") or {}
    if capabilities.get("streaming", True) is False:
        st.session_state["streaming"] = False


def selected_model_info(nvidia: NVIDIA) -> dict[str, Any]:
    ensure_settings(nvidia)
    return nvidia.model(
        st.session_state["selected_model"],
        safe=True,
    )


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
