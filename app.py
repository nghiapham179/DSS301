"""
app.py — Drone DSS Streamlit Entry Point  |  DSS301 Course
============================================================
Run:  python -m streamlit run app.py

NOTE: folder chứa các trang được đặt tên 'views/' (KHÔNG phải 'pages/')
      để tránh Streamlit auto-discovery — nếu dùng 'pages/' Streamlit sẽ
      tự thêm chúng vào sidebar mặc định, conflict với custom nav.

Multi-page structure:
  app.py            ← entry point: page_config + sidebar nav + routing
  core.py           ← shared logic (data, models, helpers)
  ui.py             ← reusable UI components
  views/
    dashboard.py    ← System Overview
    parameters.py   ← Điều chỉnh thông số (slider + form tabs)
    templates.py    ← Phiếu mẫu (8 ready-made scenarios)
    analysis.py     ← Phân tích theo drone
    model_info.py   ← Accuracy, F1, confusion matrix, feature importance
"""

import streamlit as st

# Page config MUST come before any other Streamlit call
st.set_page_config(
    page_title="Drone DSS | DSS301",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import load_css, render_sidebar_header, render_sidebar_nav
load_css()

# Import view modules (note: views/, not pages/)
from app_views import dashboard, parameters, analysis, model_info


# ─────────────────────────────────────────────────────────────
# SIDEBAR — header + nav
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar_header()

    st.markdown(
        "<p style='font-size:.62rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:.08em;color:rgba(255,255,255,.28);margin:0 0 10px 4px;'>"
        "Navigation</p>",
        unsafe_allow_html=True,
    )

    page = render_sidebar_nav([
        ("🏠", "Dashboard"),
        ("🎛️", "Điều chỉnh thông số"),
        ("📊", "Phân tích drone"),
        ("🤖", "Model Info"),
    ], key="main_nav")


# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────

ROUTES = {
    "Dashboard":            dashboard.render,
    "Điều chỉnh thông số":  parameters.render,
    "Phân tích drone":      analysis.render,
    "Model Info":           model_info.render,
}

render_fn = ROUTES.get(page, dashboard.render)
render_fn()