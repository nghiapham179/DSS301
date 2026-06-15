"""
ui.py — Drone DSS Design System  |  DSS301 Course
===================================================
Import all render_* functions into app.py.
Call load_css() once, immediately after st.set_page_config().

Design tokens:
  --accent        #4f63d2   slate-indigo
  --bg-page       #f0f2f7   page background
  --bg-card       #ffffff   card / panel
  --text-primary  #1c1e2e
  --text-secondary #64697a
  --text-muted    #9da3b5
  --sidebar-bg    #1a1d2e
"""

import streamlit as st
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────
# CSS SYSTEM
# ─────────────────────────────────────────────────────────────

def load_css():
    """Global stylesheet. Call once after st.set_page_config()."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

        /* ── TOKENS ── */
        :root {
            --bg-page:         #f0f2f7;
            --bg-card:         #ffffff;
            --bg-card-alt:     #f7f8fc;
            --accent:          #4f63d2;
            --accent-light:    #eef0fb;
            --accent-muted:    #8a97e0;
            --c-success:       #16a34a;
            --c-success-bg:    #dcfce7;
            --c-warning:       #d97706;
            --c-warning-bg:    #fef3c7;
            --c-danger:        #dc2626;
            --c-danger-bg:     #fee2e2;
            --c-info:          #0284c7;
            --c-info-bg:       #e0f2fe;
            --text-primary:    #1c1e2e;
            --text-secondary:  #64697a;
            --text-muted:      #9da3b5;
            --border:          rgba(0, 0, 0, 0.07);
            --border-hover:    rgba(79, 99, 210, 0.25);
            --shadow-sm:       0 1px 4px rgba(0,0,0,.06), 0 0 0 1px rgba(0,0,0,.04);
            --shadow-md:       0 4px 16px rgba(0,0,0,.09), 0 0 0 1px rgba(0,0,0,.04);
            --sidebar-bg:      #1a1d2e;
            --sidebar-text:    rgba(255, 255, 255, .72);
        }

        /* ── BASE ── */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
            background-color: var(--bg-page) !important;
            color: var(--text-primary) !important;
        }

        /* ── TYPOGRAPHY ── */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }
        [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
            line-height: 1.65 !important;
        }

        /* ── CARD ── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 8px !important;
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px) !important;
            box-shadow: var(--shadow-md) !important;
            border-color: var(--border-hover) !important;
        }

        /* ── METRIC ── */
        div[data-testid="stMetricLabel"] { margin-bottom: 4px !important; }
        div[data-testid="stMetricLabel"] * {
            color: var(--text-secondary) !important;
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.07em !important;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-primary) !important;
            font-size: 1.9rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stMetricDelta"] { margin-top: 5px !important; }
        div[data-testid="stMetricDelta"] * { font-size: 0.78rem !important; font-weight: 600 !important; }

        /* ── NOTE ── */
        .drone-note {
            display: block;
            color: var(--text-muted) !important;
            font-size: 0.75rem !important;
            margin-top: 10px !important;
            padding-top: 9px !important;
            border-top: 1px solid var(--border) !important;
            line-height: 1.55 !important;
        }

        /* ── SIDEBAR ── */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid rgba(255,255,255,.05) !important;
        }
        section[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
        section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }

        /* ── SIDEBAR NAV ── */
        div[role="radiogroup"] > label {
            background: transparent !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            margin-bottom: 3px !important;
            border: 1px solid transparent !important;
            transition: background .15s ease, border-color .15s ease !important;
            cursor: pointer !important;
        }
        div[role="radiogroup"] > label span[data-baseweb="radio"] { display: none !important; }
        div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
            color: var(--sidebar-text) !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            margin: 0 !important;
            transition: color .15s ease !important;
        }
        div[role="radiogroup"] > label:hover {
            background: rgba(255,255,255,.06) !important;
            border-color: rgba(255,255,255,.08) !important;
        }
        div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p { color: #fff !important; }
        div[role="radiogroup"] > label[data-checked="true"] {
            background: rgba(79,99,210,.22) !important;
            border-color: rgba(79,99,210,.42) !important;
        }
        div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
            color: #fff !important;
            font-weight: 600 !important;
        }

        /* ── UPGRADE CARD ── */
        .pro-upgrade-card {
            background: rgba(79,99,210,.13);
            border: 1px solid rgba(79,99,210,.26);
            border-radius: 14px;
            padding: 18px 14px;
            text-align: center;
            margin-top: 32px;
        }
        .pro-upgrade-card h4 { color: #fff !important; font-size: 0.9rem !important; margin: 0 0 4px !important; }
        .pro-upgrade-card p  { color: rgba(255,255,255,.55) !important; font-size: 0.73rem !important; margin: 0 0 12px !important; line-height: 1.45 !important; }
        section[data-testid="stSidebar"] a.pro-upgrade-btn {
            background-color: var(--accent) !important;
            color: #fff !important;
            padding: 8px 18px;
            border-radius: 9px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
            width: 100%;
            box-sizing: border-box;
            transition: opacity .15s ease;
        }
        section[data-testid="stSidebar"] a.pro-upgrade-btn:hover { opacity: .85; }

        /* ── FORM INPUTS ── */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            background: var(--bg-card-alt) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-size: 0.87rem !important;
            transition: border-color .15s ease, box-shadow .15s ease !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--accent-muted) !important;
            box-shadow: 0 0 0 3px var(--accent-light) !important;
            outline: none !important;
        }
        [data-testid="stFormSubmitButton"] button {
            background-color: var(--accent) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 10px 24px !important;
            transition: opacity .15s ease !important;
        }
        [data-testid="stFormSubmitButton"] button:hover { opacity: .88 !important; }

        /* ── SLIDER ── */
        [data-testid="stSlider"] [role="slider"] {
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        /* ── SELECT ── */
        [data-testid="stSelectbox"] > div > div { border-radius: 10px !important; border-color: var(--border) !important; }

        /* ── STATUS BADGE ── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        /* ── DIVIDER ── */
        hr[data-testid="stDivider"] { border-color: var(--border) !important; margin: 16px 0 !important; }

        /* ── DATAFRAME ── */
        [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border) !important; }

        /* ── EXPANDER ── */
        [data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid var(--border) !important; overflow: hidden !important; }
        [data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 0.88rem !important; color: var(--text-primary) !important; }

        /* ── PROGRESS BAR (shared) ── */
        .prog-track { background: #f1f3f9; border-radius: 6px; height: 6px; overflow: hidden; margin: 8px 0 3px; }
        .prog-fill  { height: 100%; border-radius: 6px; transition: width .6s cubic-bezier(.4,0,.2,1); }
        .prog-label { font-size: 0.68rem; color: var(--text-muted); margin: 0; }

        /* ── CLASS ROW (Model Info) ── */
        .class-row { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--border); }
        .class-row:last-child { border-bottom: none; }
        .class-name { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); min-width: 160px; }
        .class-f1   { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; min-width: 44px; text-align: right; }
        .class-bar-track { flex: 1; background: #f1f3f9; border-radius: 4px; height: 6px; overflow: hidden; }
        .class-bar-fill  { height: 100%; border-radius: 4px; }

        /* ── SYSTEM ── */
        /* Hide Streamlit chrome — header, footer, deploy button, toolbar */
        header[data-testid="stHeader"]      { display: none !important; }
        footer                              { visibility: hidden !important; }
        [data-testid="stToolbar"]           { display: none !important; }
        [data-testid="stDecoration"]        { display: none !important; }
        [data-testid="stStatusWidget"]      { display: none !important; }
        .stDeployButton                     { display: none !important; }
        #MainMenu                           { display: none !important; }
        .viewerBadge_container__1QSob       { display: none !important; }
        .styles_viewerBadge__1yB5_          { display: none !important; }

        /* Reduce top padding so content sits closer to top */
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.5rem !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem !important;
        }

        /* Smoother divider */
        hr { border-top: 1px solid var(--border) !important; }

        /* Better captions under charts */
        [data-testid="stCaptionContainer"] {
            font-size: 0.78rem !important;
            color: var(--text-secondary) !important;
            line-height: 1.55 !important;
            padding: 0 4px !important;
            margin-top: 4px !important;
        }
        .js-plotly-plot .plotly .modebar { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.09) -> str:
    """
    Convert a 6-char hex color string to an rgba() string Plotly accepts.
    Plotly's fillcolor does NOT accept 8-char hex (#rrggbbaa) — use rgba() instead.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_sparkline(data, color: str = "#4f63d2") -> go.Figure:
    """Minimal sparkline: spline + subtle fill. No axes, no tooltips."""
    pts  = list(data)
    fill = _hex_to_rgba(color, alpha=0.09)   # rgba() — Plotly compatible
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(pts))), y=pts,
        mode="lines",
        line=dict(color=color, width=2.5, shape="spline"),
        fill="tozeroy", fillcolor=fill,
        hoverinfo="skip",
    ))
    pad = (max(pts) - min(pts)) * 0.1 or 0.5
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=4), height=50,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, len(pts) - 1]),
        yaxis=dict(visible=False, range=[min(pts) - pad, max(pts) + pad]),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# RENDER — TOP NAV & TITLES
# ─────────────────────────────────────────────────────────────

def render_top_nav():
    """Top bar: title, search input, notification bell, user avatar."""
    c_title, c_search, c_profile = st.columns([5, 3, 1], vertical_alignment="center")

    with c_title:
        st.markdown(
            """
            <div style="padding:4px 0;">
                <h2 style="margin:0;font-size:1.35rem;font-weight:700;
                           letter-spacing:-0.025em;color:var(--text-primary);">
                    Dashboard
                </h2>
                <p style="margin:2px 0 0;font-size:0.78rem;
                          color:var(--text-secondary);font-weight:400;">
                    Drone Decision Support System &nbsp;·&nbsp; DSS301
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_search:
        st.text_input("Search", placeholder="🔍  Search drones, records…",
                      label_visibility="collapsed")

    with c_profile:
        st.markdown(
            """
            <div style="display:flex;justify-content:flex-end;align-items:center;gap:12px;">
                <span style="font-size:1.05rem;cursor:pointer;color:var(--text-secondary);"
                      title="Notifications">🔔</span>
                <div style="width:34px;height:34px;border-radius:50%;
                            background:var(--accent-light);display:flex;
                            justify-content:center;align-items:center;
                            font-weight:700;font-size:0.72rem;color:var(--accent);
                            border:1.5px solid var(--accent-muted);"
                     title="Admin">AD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)


def render_page_title(title: str, subtitle: str = ""):
    """Page section title with optional muted subtitle."""
    sub = (
        f"<p style='font-size:0.8rem;color:var(--text-secondary);margin:2px 0 0;'>{subtitle}</p>"
        if subtitle else ""
    )
    st.markdown(
        f"""<div style="margin-bottom:16px;">
            <h3 style="font-weight:700;font-size:1.12rem;letter-spacing:-0.02em;
                       margin:0;color:var(--text-primary);">{title}</h3>{sub}
        </div>""",
        unsafe_allow_html=True,
    )


def render_section_label(text: str):
    """Eyebrow label — small, uppercase, letter-spaced."""
    st.markdown(
        f"""<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.09em;color:var(--text-muted);margin:20px 0 8px;">
            {text}</p>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# RENDER — METRIC CARDS
# ─────────────────────────────────────────────────────────────

def render_metric_card(label: str, value: str, note: str = "", delta: str = None):
    """Basic metric card, no sparkline."""
    with st.container(border=True):
        st.metric(label=label, value=value, delta=delta)
        if note:
            st.markdown(f"<span class='drone-note'>{note}</span>", unsafe_allow_html=True)


def render_metric_with_chart(label: str, value: str, data_points,
                             color: str = "#4f63d2", delta: str = None):
    """Metric card with sparkline trend chart."""
    with st.container(border=True):
        st.metric(label=label, value=value, delta=delta)
        pts = list(data_points) if data_points is not None else []
        if len(pts) > 1:
            st.plotly_chart(
                create_sparkline(pts, color),
                use_container_width=True,
                config={"displayModeBar": False},
            )


def render_risk_score(score: float, risk_level: str, note: str = ""):
    """
    Risk score card with colour-coded progress bar + badge.
    Levels: High / Medium / Low  (README Section 11)
    """
    risk_level = str(risk_level).strip().title()
    _P = {
        "High":   dict(bar="#dc2626", bg="#fee2e2", fg="#b91c1c", delta="inverse", dot="🔴"),
        "Medium": dict(bar="#d97706", bg="#fef3c7", fg="#b45309", delta="off",     dot="🟡"),
        "Low":    dict(bar="#16a34a", bg="#dcfce7", fg="#15803d", delta="normal",  dot="🟢"),
    }
    p   = _P.get(risk_level, dict(bar="#4f63d2", bg="#eef0fb", fg="#4f63d2", delta="normal", dot="🔵"))
    pct = max(0, min(100, int(float(score) / 10 * 100)))

    with st.container(border=True):
        c_num, c_badge = st.columns([3, 1])
        with c_num:
            st.metric(label="Overall Risk Score",
                      value=f"{score:.1f} / 10",
                      delta=f"{p['dot']}  {risk_level} Risk",
                      delta_color=p["delta"])
        with c_badge:
            st.markdown(
                f"""<div style="display:flex;align-items:flex-end;height:100%;padding-bottom:6px;">
                    <span class="status-badge" style="background:{p['bg']};color:{p['fg']};">
                        {risk_level}</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:{p['bar']};"></div></div>
            <p class="prog-label">{pct}% of maximum risk threshold</p>""",
            unsafe_allow_html=True,
        )
        if note:
            st.markdown(f"<span class='drone-note'>{note}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RENDER — BANNERS & BADGES
# ─────────────────────────────────────────────────────────────

def render_result_badge(label: str, value: str, level: str = "info"):
    """Streamlit alert with icon, used for prediction results."""
    _ICON = {"danger": "⛔", "warning": "⚠️", "success": "✅", "info": "ℹ️"}
    _FN   = {"danger": st.error, "warning": st.warning, "success": st.success, "info": st.info}
    body  = f"**{label}:** {value}" if label else value
    _FN.get(level, st.info)(body, icon=_ICON.get(level, "ℹ️"))


def render_banner(text: str, level: str = "info"):
    """Short notification banner (wrapper for render_result_badge)."""
    render_result_badge("", text, level)


# ─────────────────────────────────────────────────────────────
# RENDER — MODEL INFO COMPONENTS
# ─────────────────────────────────────────────────────────────

def render_model_accuracy_card(model_name: str, accuracy: float,
                               cv_mean: float, cv_std: float,
                               trained_at: str = ""):
    """
    Accuracy card with colour-coded progress bar.
    Green ≥ 90%, Yellow ≥ 75%, Red < 75%.
    """
    pct       = int(accuracy * 100)
    bar_color = "#16a34a" if accuracy >= 0.90 else "#d97706" if accuracy >= 0.75 else "#dc2626"
    display   = model_name.replace("_", " ").title()

    with st.container(border=True):
        st.metric(
            label=display,
            value=f"{accuracy:.1%}",
            delta=f"CV: {cv_mean:.1%} ± {cv_std:.1%}",
            delta_color="normal",
        )
        st.markdown(
            f"""<div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:{bar_color};"></div></div>
            <p class="prog-label">Test accuracy  ·  {trained_at}</p>""",
            unsafe_allow_html=True,
        )


def render_class_f1_table(classes: list, f1_scores: dict,
                          precision: dict = None, recall: dict = None):
    """Per-class F1 table — compact rows with progress bar + score."""
    st.caption("PER-CLASS F1 SCORE")

    for cls in classes:
        f1  = float(f1_scores.get(cls, 0.0))
        pct = int(f1 * 100)

        if f1 >= 0.90:
            bar_col, txt_col, icon = "#16a34a", "#15803d", "🟢"
        elif f1 >= 0.70:
            bar_col, txt_col, icon = "#d97706", "#b45309", "🟡"
        else:
            bar_col, txt_col, icon = "#dc2626", "#b91c1c", "🔴"

        pr  = precision.get(cls, 0) if precision else None
        rec = recall.get(cls, 0)    if recall    else None
        extra = ""
        if pr is not None and rec is not None:
            extra = f"P {pr:.2f}  ·  R {rec:.2f}"

        # Truncate long class names
        short = cls if len(cls) <= 32 else cls[:30] + "…"

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;
                        padding:5px 0;border-bottom:1px solid #eef0f5;
                        font-family:Inter,system-ui,sans-serif;">
                <span style="font-size:0.8rem;font-weight:600;color:#1c1e2e;
                             min-width:170px;max-width:170px;
                             white-space:nowrap;overflow:hidden;
                             text-overflow:ellipsis;" title="{cls}">
                    {icon} {short}
                </span>
                <div style="flex:1;background:#f1f3f9;border-radius:3px;
                            height:6px;overflow:hidden;min-width:80px;">
                    <div style="width:{pct}%;background:{bar_col};
                                height:100%;border-radius:3px;"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;
                             font-size:0.85rem;font-weight:600;
                             color:{txt_col};min-width:38px;text-align:right;">
                    {f1:.2f}
                </span>
                <span style="font-size:0.7rem;color:#9da3b5;
                             font-family:'JetBrains Mono',monospace;
                             min-width:96px;text-align:right;">
                    {extra}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confusion_matrix_heatmap(cm: list, classes: list) -> go.Figure:
    """
    Plotly confusion matrix heatmap, normalised by row (%).
    Long labels are wrapped/abbreviated to avoid overlap.
    """
    import numpy as np
    cm_arr  = np.array(cm)
    row_sum = cm_arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm_arr / row_sum

    # Shorten class labels for axis ticks (full name shown in hover)
    def _short(name, n=18):
        return name if len(name) <= n else name[:n - 1] + "…"

    short_x = [_short(c, 16) for c in classes]
    short_y = [_short(c, 22) for c in classes]

    text = [[f"<b>{cm_arr[i][j]}</b><br>{cm_norm[i][j]:.0%}"
             for j in range(len(classes))]
            for i in range(len(classes))]

    # Hover shows full class name
    hover = [[f"Actual: {classes[i]}<br>Predicted: {classes[j]}"
              f"<br>Count: {cm_arr[i][j]}<br>Row %: {cm_norm[i][j]:.1%}"
              for j in range(len(classes))]
             for i in range(len(classes))]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=short_x, y=short_y,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono"),
        colorscale=[[0, "#f0f2f7"], [0.5, "#8a97e0"], [1, "#4f63d2"]],
        showscale=False,
        hoverinfo="text",
        hovertext=hover,
    ))

    # Dynamic height based on number of classes
    n = len(classes)
    height = max(240, 60 + n * 55)

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Predicted", font=dict(size=11, color="#64697a")),
            tickfont=dict(size=10, color="#1c1e2e"),
            tickangle=0 if max(len(c) for c in short_x) <= 12 else -25,
            side="bottom",
        ),
        yaxis=dict(
            title=dict(text="Actual", font=dict(size=11, color="#64697a")),
            tickfont=dict(size=10, color="#1c1e2e"),
            autorange="reversed",
        ),
        margin=dict(t=10, b=80, l=120, r=20),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui", color="#64697a", size=10),
    )
    return fig


def render_feature_importance_chart(fi_df) -> go.Figure:
    """
    Horizontal bar chart for averaged feature importance.
    fi_df must have columns: feature, importance.
    Returns Figure.
    """
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale=[[0, "#eef0fb"], [1, "#4f63d2"]],
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{v:.3f}" for v in fi_df["importance"]],
        textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono", color="#64697a"),
    ))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#1c1e2e")),
        margin=dict(t=8, b=8, l=8, r=70),
        height=max(260, len(fi_df) * 32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui", color="#64697a"),
        bargap=0.3,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# RENDER — SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar_header():
    """App logo + title in dark sidebar."""
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;
                    margin-bottom:26px;padding:2px 4px;">
            <div style="width:32px;height:32px;border-radius:8px;
                        background:rgba(79,99,210,.28);display:flex;
                        align-items:center;justify-content:center;
                        font-size:1.15rem;line-height:1;">🚁</div>
            <div>
                <p style="margin:0;font-size:0.98rem;font-weight:700;
                          color:#fff;letter-spacing:-0.02em;">Drone DSS</p>
                <p style="margin:0;font-size:0.62rem;color:rgba(255,255,255,.35);
                          font-weight:500;letter-spacing:.05em;text-transform:uppercase;">
                    DSS301 &nbsp;·&nbsp; Warranty &amp; Ops</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav(items: list, key: str = "main_nav") -> str:
    """
    Render sidebar navigation as styled boxes instead of plain radio buttons.

    items: list of (icon, label) tuples
    Returns: the label of the currently selected item

    Uses st.session_state to persist selection across reruns.
    """
    # Init state
    if key not in st.session_state:
        st.session_state[key] = items[0][1]

    current = st.session_state[key]

    # CSS for nav boxes
    st.markdown(
        """
        <style>
            .nav-box-btn {
                width: 100%;
                background: rgba(255,255,255,0.04) !important;
                color: rgba(255,255,255,0.72) !important;
                border: 1px solid rgba(255,255,255,0.06) !important;
                border-radius: 10px !important;
                padding: 11px 14px !important;
                font-size: 0.88rem !important;
                font-weight: 500 !important;
                text-align: left !important;
                margin-bottom: 6px !important;
                transition: all .15s ease !important;
                box-shadow: none !important;
            }
            .nav-box-btn:hover {
                background: rgba(255,255,255,0.08) !important;
                color: #ffffff !important;
                border-color: rgba(255,255,255,0.12) !important;
            }
            .nav-box-btn.active {
                background: linear-gradient(135deg, rgba(79,99,210,0.35), rgba(79,99,210,0.18)) !important;
                color: #ffffff !important;
                border-color: rgba(79,99,210,0.55) !important;
                box-shadow: 0 2px 12px rgba(79,99,210,0.18) !important;
                font-weight: 600 !important;
            }
            section[data-testid="stSidebar"] button[kind="secondary"] {
                background: rgba(255,255,255,0.04) !important;
                color: rgba(255,255,255,0.72) !important;
                border: 1px solid rgba(255,255,255,0.06) !important;
                border-radius: 10px !important;
                padding: 11px 14px !important;
                font-size: 0.88rem !important;
                font-weight: 500 !important;
                text-align: left !important;
                margin-bottom: 6px !important;
                box-shadow: none !important;
                justify-content: flex-start !important;
            }
            section[data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: rgba(255,255,255,0.08) !important;
                color: #ffffff !important;
                border-color: rgba(255,255,255,0.12) !important;
            }
            section[data-testid="stSidebar"] button[kind="primary"] {
                background: linear-gradient(135deg, rgba(79,99,210,0.40), rgba(79,99,210,0.20)) !important;
                color: #ffffff !important;
                border: 1px solid rgba(79,99,210,0.55) !important;
                border-radius: 10px !important;
                padding: 11px 14px !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                text-align: left !important;
                margin-bottom: 6px !important;
                box-shadow: 0 2px 12px rgba(79,99,210,0.20) !important;
                justify-content: flex-start !important;
            }
            section[data-testid="stSidebar"] button p {
                text-align: left !important;
                margin: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render each item as a Streamlit button
    for i, (icon, label) in enumerate(items):
        is_active = (label == current)
        btn_type  = "primary" if is_active else "secondary"

        if st.button(
                f"{icon}   {label}",
                key=f"{key}_btn_{i}",
                use_container_width=True,
                type=btn_type,
        ):
            st.session_state[key] = label
            st.rerun()

    return current


def render_sidebar_upgrade_card():
    """Stub — kept for backward compatibility, no longer renders anything."""
    return None