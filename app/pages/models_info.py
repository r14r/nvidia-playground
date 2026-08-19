import streamlit as st

from shared import require_nvidia, selected_model_info

st.title("Models · Info")
st.caption("Human-readable information for the currently selected model.")

nvidia = require_nvidia()
info = selected_model_info(nvidia)
selected = st.session_state["selected_model"]

st.subheader(selected)

col1, col2, col3 = st.columns(3)
col1.metric("Provider", info.get("provider", "—"))
col2.metric("Source", info.get("source", "—"))
col3.metric("Type", info.get("type", "—"))

col4, col5, col6 = st.columns(3)
col4.metric("Default", "Yes" if info.get("default") else "No")
col5.metric("Credential", info.get("credential_status", "—") or "—")
col6.metric("Runnable", "Yes" if nvidia.can_run(selected) else "No")

if info.get("base_url"):
    st.subheader("Endpoint")
    st.code(info["base_url"], language=None)

capabilities = info.get("capabilities") or {}
if capabilities:
    st.subheader("Capabilities")
    capability_rows = [
        {
            "Capability": name,
            "Supported": bool(enabled),
        }
        for name, enabled in capabilities.items()
    ]
    st.dataframe(
        capability_rows,
        width="stretch",
        hide_index=True,
    )

if info.get("model_source"):
    st.subheader("Model Source")
    st.json(info["model_source"])

if info.get("sample_code"):
    st.subheader("NVIDIA Sample Code")
    st.code(info["sample_code"], language="python")
