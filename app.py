"""
app.py — Streamlit entrypoint (runs on Streamlit Community Cloud).

This app is BACKEND-FREE: the trained XGBoost model is exported to JSON and runs
directly in the browser (see export_model_js.py). This Python file is just a thin
Streamlit wrapper that injects the precomputed static data into the self-contained
Three.js UI and renders it. No Flask, no API, no server-side model.

Regenerate the static assets after retraining (needs the ML stack — see
requirements-dev.txt):
    python export_model_js.py       # static/model.json       (browser model)
    python build_circuits_geo.py    # static/circuits_geo.json (needs internet)
    python build_app_data.py        # static/app_data.json     (circuits + insights)

Run locally:
    streamlit run app.py
"""
import pathlib

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="F1 Pit-Stop Predictor", page_icon="🏎️", layout="wide")

BASE   = pathlib.Path(__file__).parent
TPL    = BASE / "templates"
STATIC = BASE / "static"


@st.cache_data
def load_assets():
    return (
        (STATIC / "model.json").read_text(encoding="utf-8"),
        (STATIC / "circuits_geo.json").read_text(encoding="utf-8"),
        (STATIC / "app_data.json").read_text(encoding="utf-8"),
    )


MODEL_JSON, GEO_JSON, DATA_JSON = load_assets()


def inject(html: str, with_model: bool) -> str:
    """Replace the <!--DATA_INJECT--> placeholder with the inlined data."""
    parts = [f"window.APPDATA={DATA_JSON};", f"window.GEO={GEO_JSON};"]
    if with_model:
        parts.insert(0, f"window.MODEL={MODEL_JSON};")
    return html.replace("<!--DATA_INJECT-->", "<script>" + "".join(parts) + "</script>")


# Strip Streamlit chrome so the app uses the full width/height.
st.markdown(
    """<style>
      #MainMenu, footer, header {visibility:hidden;}
      .block-container {padding:0 !important; max-width:100% !important;}
      .stApp {background:#0a0e14;}
      div[data-testid="stTabs"] button p {font-size:15px;}
    </style>""",
    unsafe_allow_html=True,
)

tab_predict, tab_insights = st.tabs(["🏎️  Predictor", "📊  Feature insights"])

with tab_predict:
    html = (TPL / "index.html").read_text(encoding="utf-8")
    components.html(inject(html, with_model=True), height=860, scrolling=False)

with tab_insights:
    html = (TPL / "features.html").read_text(encoding="utf-8")
    components.html(inject(html, with_model=False), height=900, scrolling=True)
