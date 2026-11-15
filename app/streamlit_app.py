"""Trend Explorer – entry point.

Run from the repo root:   streamlit run app/streamlit_app.py
"""
import streamlit as st

from lib import data
from lib import text as T

st.set_page_config(page_title=T.APP_TITLE, page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# Small shared style: cards, tiles, badges (used through st.markdown(..., unsafe_allow_html=True))
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.te-sub {color:#4B5563; font-size:14px; margin-top:-0.6rem;}
.te-tile {background:#F7F9F8; border-radius:10px; padding:10px 12px; height:100%;}
.te-tile .l {font-size:12px; color:#4B5563;}
.te-tile .v {font-size:21px; font-weight:600; line-height:1.3;}
.te-badge {display:inline-block; font-size:12px; font-weight:600; padding:4px 10px; border-radius:999px; margin:0 6px 6px 0;}
.te-bar {display:flex; height:18px; border-radius:6px; overflow:hidden; gap:2px; margin:6px 0;}
.te-key {display:inline-flex; align-items:center; gap:6px; margin-right:16px; font-size:13px; color:#374151;}
.te-sw {display:inline-block; width:14px; height:10px; border-radius:2px;}
.te-small {font-size:12px; color:#4B5563; line-height:1.5;}
.te-expl {font-size:11px; font-weight:700; color:#6B21A8; background:#F3E8FF; padding:2px 8px; border-radius:999px;}
.te-hero {background:#0F2E2B; border-radius:14px; padding:22px; color:#FFFFFF;}
.te-hero .k {font-size:12px; font-weight:700; letter-spacing:.06em; color:#99F6E4;}
.te-hero .m {font-size:21px; line-height:1.45; margin:8px 0 14px;}
.te-hero .t {background:#134E48; border-radius:10px; padding:12px;}
.te-hero .t b {font-size:26px;} .te-hero .t div {font-size:13px; color:#CCFBF1;}
</style>
""", unsafe_allow_html=True)

if not data.has("trend_cells.nc"):
    st.error(f"Result files not found in `{data.DATA_DIR}`. Put the processed files in "
             "data/processed/ or set the TRENDLAB_DATA_DIR environment variable.")
    st.stop()

pages = [
    st.Page("views/map_explorer.py", title="Map explorer", icon=":material/map:", default=True),
    st.Page("views/discovery.py", title="Discovery", icon=":material/travel_explore:"),
    st.Page("views/lab.py", title="Discovery Lab", icon=":material/science:"),
    st.Page("views/regions.py", title="Regions", icon=":material/stacked_line_chart:"),
    st.Page("views/exploratory.py", title="Exploratory", icon=":material/explore:"),
    st.Page("views/about_methods.py", title="About & methods", icon=":material/info:"),
]
st.navigation(pages, position="top").run()
