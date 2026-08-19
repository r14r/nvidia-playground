from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from shared import APP_VERSION

st.set_page_config(
    page_title="NVIDIA NIM Playground",
    page_icon="⚡",
    layout="wide",
)

pages = {
    "": [
        st.Page(
            "pages/run.py",
            title="Run",
            icon=":material/play_arrow:",
            default=True,
        ),
    ],
    "Models": [
        st.Page(
            "pages/models_info.py",
            title="Info",
            icon=":material/info:",
        ),
        st.Page(
            "pages/models_metadata.py",
            title="Metadata",
            icon=":material/data_object:",
        ),
        st.Page(
            "pages/models_catalog.py",
            title="Catalog",
            icon=":material/view_list:",
        ),
    ],
}

page = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

st.sidebar.caption(f"NVIDIA NIM Playground · v{APP_VERSION}")

page.run()
