from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.providers import (
    NVIDIA_MODELS_URL,
    OLLAMA_MODELS_URL,
    PROVIDERS,
    list_provider_models,
)
from app.lib.shared import MODELS_FILE

st.title("Provider")
st.caption("Connect to a model provider and list the models currently available.")

provider = st.selectbox(
    "Provider",
    options=PROVIDERS,
    index=0,
    key="provider_catalog_source",
)

source_label = {
    "Ollama": OLLAMA_MODELS_URL,
    "NVIDIA": NVIDIA_MODELS_URL,
    "models.json": str(MODELS_FILE),
}[provider]
st.caption(f"Source: `{source_label}`")

list_models = st.button(
    "List Models",
    type="primary",
)

if list_models:
    try:
        with st.spinner(f"Loading models from {provider}…"):
            rows = list_provider_models(
                provider,
                models_file=MODELS_FILE,
            )
        st.session_state["provider_models_rows"] = rows
        st.session_state["provider_models_source"] = provider
        st.session_state["provider_models_error"] = ""
    except Exception as exc:
        st.session_state["provider_models_rows"] = []
        st.session_state["provider_models_source"] = provider
        st.session_state["provider_models_error"] = str(exc)

if st.session_state.get("provider_models_source") == provider:
    error = st.session_state.get("provider_models_error", "")
    rows = st.session_state.get("provider_models_rows", [])

    if error:
        st.error(error)
    elif rows:
        st.metric("Models", len(rows))
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )
    elif list_models:
        st.info("No models returned by this provider.")
