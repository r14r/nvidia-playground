import pandas as pd
import streamlit as st

from shared import MODELS_FILE, reload_nvidia, require_nvidia

st.title("Models · Catalog")
st.caption("Overview and default-model configuration for models.json.")

nvidia = require_nvidia()
catalog = nvidia.catalog.safe_catalog()
models = nvidia.models(chat_only=False, safe=True)
chat_models = nvidia.models(chat_only=True, safe=True)

model_ids = [
    str(model.get("id") or model.get("model"))
    for model in models
]
default_model = nvidia.default_model()

control1, control2 = st.columns([2, 1])

with control1:
    selected_default = st.selectbox(
        "Default model",
        options=model_ids,
        index=(
            model_ids.index(default_model)
            if default_model in model_ids
            else 0
        ),
        key="catalog_default_model",
    )

with control2:
    st.write("")
    st.write("")
    if st.button(
        "Save default model",
        type="primary",
        width="stretch",
    ):
        try:
            saved = nvidia.set_default_model(selected_default)
            st.session_state["selected_model"] = saved
            reload_nvidia()
            st.success(f"Default model saved: {saved}")
            st.rerun()
        except Exception as exc:
            st.exception(exc)

if st.button("Reload models.json"):
    reload_nvidia()
    st.rerun()

# Refresh safe snapshots after potential operations.
nvidia = require_nvidia()
catalog = nvidia.catalog.safe_catalog()
models = nvidia.models(chat_only=False, safe=True)
chat_models = nvidia.models(chat_only=True, safe=True)

default_models = [
    model for model in models if model.get("default") is True
]

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Models", len(models))
metric2.metric("Chat Models", len(chat_models))
metric3.metric("Default Models", len(default_models))
metric4.metric("Scopes", len(catalog.get("selected_scopes", [])))

st.caption(f"Catalog file: `{MODELS_FILE}`")
st.caption(
    "Saving changes models.json so exactly one model has `default: true`."
)

rows = []
for model in models:
    capabilities = model.get("capabilities") or {}
    rows.append(
        {
            "number": model.get("number"),
            "default": bool(model.get("default")),
            "id": model.get("id") or model.get("model"),
            "provider": model.get("provider"),
            "source": model.get("source"),
            "type": model.get("type"),
            "credential": model.get("credential_status"),
            "streaming": capabilities.get("streaming"),
            "structured_output": capabilities.get("structured_output"),
            "tools": capabilities.get("tools"),
            "embedding": capabilities.get("embedding"),
            "rerank": capabilities.get("rerank"),
        }
    )

if rows:
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No models found.")

with st.expander("Config folders"):
    st.json(catalog.get("config_folders", {}))

with st.expander("Selected scopes"):
    st.json(catalog.get("selected_scopes", []))

with st.expander("Raw safe catalog"):
    st.json(catalog)
