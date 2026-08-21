from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.lib.model_details import (
    load_remote_model_details,
    model_details_url,
)
from app.lib.nvidia_catalog import (
    NVIDIA_MODELS_API_URL,
    list_nvidia_api_models,
)
from app.lib.providers import (
    NVIDIA_MODELS_URL,
    OLLAMA_MODELS_URL,
    PROVIDERS,
    list_provider_models,
)
from app.lib.shared import MODELS_FILE


@st.cache_data(ttl=3600, show_spinner=False)
def _remote_details(provider: str, model: str) -> dict[str, Any]:
    return load_remote_model_details(provider, model)


def _details_table(details: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Property": key, "Value": value}
            for key, value in details.items()
            if key not in {"Model Card Excerpt", "Library Excerpt"}
        ]
    )


def _load_models(provider: str) -> list[dict[str, Any]]:
    if provider != "NVIDIA":
        return list_provider_models(
            provider,
            models_file=MODELS_FILE,
        )

    try:
        return list_nvidia_api_models()
    except Exception as api_exc:
        try:
            return list_provider_models(
                provider,
                models_file=MODELS_FILE,
            )
        except Exception as catalog_exc:
            raise RuntimeError(
                "Could not load the NVIDIA model list. "
                f"Official /v1/models endpoint failed: {api_exc}. "
                f"build.nvidia.com HTML fallback failed: {catalog_exc}"
            ) from catalog_exc


st.title("Provider")
st.caption(
    "Connect to a model provider, list available models, and inspect "
    "details for a selected model."
)

provider = st.selectbox(
    "Provider",
    options=PROVIDERS,
    index=0,
    key="provider_catalog_source",
)

source_label = {
    "Ollama": OLLAMA_MODELS_URL,
    "NVIDIA": NVIDIA_MODELS_API_URL,
    "models.json": str(MODELS_FILE),
}[provider]
st.caption(f"Model list source: `{source_label}`")

if provider == "NVIDIA":
    st.caption(
        "Model cards and browsing remain on "
        f"`{NVIDIA_MODELS_URL}`. The model list itself is loaded from "
        "NVIDIA's official `/v1/models` JSON endpoint because the Build "
        "catalog page is dynamically rendered."
    )

catalogs = st.session_state.setdefault("provider_catalogs", {})
errors = st.session_state.setdefault("provider_catalog_errors", {})

if st.button("List Models", type="primary"):
    try:
        with st.spinner(f"Loading models from {provider}…"):
            catalogs[provider] = _load_models(provider)
        errors[provider] = ""
    except Exception as exc:
        catalogs[provider] = []
        errors[provider] = str(exc)

error = errors.get(provider, "")
rows = catalogs.get(provider, [])

if error:
    st.error(error)

model_options = [
    str(row.get("Model"))
    for row in rows
    if row.get("Model")
]

selected_model = st.selectbox(
    "Model",
    options=model_options,
    index=0 if model_options else None,
    disabled=not model_options,
    placeholder="List models first",
    key=f"provider_selected_model_{provider}",
)

if rows:
    st.metric("Models", len(rows))
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )
elif not error:
    st.info("Use **List Models** to load models from the selected provider.")

if selected_model:
    st.divider()
    st.subheader("Model details")

    local_details = next(
        (
            dict(row)
            for row in rows
            if str(row.get("Model")) == selected_model
        ),
        {"Provider": provider, "Model": selected_model},
    )
    local_details.pop("api_key", None)

    st.markdown("**Provider data**")
    st.dataframe(
        _details_table(local_details),
        width="stretch",
        hide_index=True,
    )

    detail_url = model_details_url(provider, selected_model)
    if detail_url:
        st.markdown("**External model information**")
        st.link_button("Open model page", detail_url)
        try:
            with st.spinner("Loading model details…"):
                remote = _remote_details(provider, selected_model)
            if remote:
                st.dataframe(
                    _details_table(remote),
                    width="stretch",
                    hide_index=True,
                )
                excerpt = (
                    remote.get("Model Card Excerpt")
                    or remote.get("Library Excerpt")
                )
                if excerpt:
                    st.markdown("**Description / Model Card**")
                    st.text(excerpt)
        except Exception as exc:
            st.warning(
                "The external model page could not be loaded: "
                f"{exc}"
            )
    elif provider == "models.json":
        st.caption(
            "models.json uses the local catalog entry as its detail source."
        )
