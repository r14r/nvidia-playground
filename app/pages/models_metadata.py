import streamlit as st

from app.lib.shared import require_nvidia, selected_model_info

st.title("Models · Metadata")
st.caption("Raw safe metadata for the currently selected model.")

nvidia = require_nvidia()
info = selected_model_info(nvidia)

st.subheader(st.session_state["selected_model"])
st.info("Sensitive credential values are masked by nvidia-lib.")
st.json(info)
