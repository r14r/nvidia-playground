from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.lib.shared import APP_VERSION

st.set_page_config(
    page_title="NVIDIA NIM Playground",
    page_icon="⚡",
    layout="wide",
)

pages = {
    "Run": [
        st.Page(
            "pages/run_single.py",
            title="Single model",
            icon=":material/play_arrow:",
            default=True,
        ),
        st.Page(
            "pages/run_multiple.py",
            title="Multiple models",
            icon=":material/stack:",
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
