"""
ui.py — Drone DSS Design System  |  DSS301 Course
===================================================
Centralized component library + design tokens.
Call load_css() once after st.set_page_config().

Design tokens (defined as CSS variables in :root):
  --accent          #4f63d2   slate-indigo
  --bg-page         #eef1f8   page background
  --bg-card         #ffffff   card / panel
  --text-primary    #1c1e2e
  --text-secondary  #64697a
  --text-muted      #9da3b5
  --sidebar-bg      #1a1d2e
"""

import streamlit as st
import plotly.graph_objects as go


# ═════════════════════════════════════════════════════════════════════
#  CSS SYSTEM
#  Tất cả CSS được gom vào 1 chỗ + scoped class.
#  Animation respect prefers-reduced-motion.
# ═════════════════════════════════════════════════════════════════════

def load_css():
    """Global stylesheet. Call once after st.set_page_config()."""
    st.markdown(
        """
        <style>
        /* Font display=swap → text hiện ngay, không bị FOIT khi font load */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* ── DESIGN TOKENS ─────────────────────────────────────── */
        :root {
            --bg-page:         #eef1f8;
            --bg-card:         #ffffff;
            --bg-card-alt:     #f7f8fc;
            --accent:          #4f63d2;
            --accent-light:    #eef0fb;
            --accent-muted:    #8a97e0;
            --lime:            #c8e85a;
            --lime-ink:        #2b3a00;
            --lime-soft:       rgba(200, 232, 90, 0.16);
            --hero-bg:         #1b1e30;
            --hero-card:       #262a44;
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
            --radius-sm:       8px;
            --radius-md:       12px;
            --radius-lg:       16px;
            --transition:      .2s cubic-bezier(.4,0,.2,1);
        }

        /* ── BASE ──────────────────────────────────────────────── */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
            background-color: var(--bg-page) !important;
            color: var(--text-primary) !important;
        }

        /* ── TYPOGRAPHY ────────────────────────────────────────── */
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
            line-height: 1.5 !important;
        }
        /* Body 14.5px CHỈ cho đoạn văn trơn — p có class/style tự mang cỡ
           riêng (KPI 32px, section label 17px...), không được đè chúng. */
        [data-testid="stMarkdownContainer"] p:not([class]):not([style]) {
            font-size: 14.5px !important;
        }
        [data-testid="stCaptionContainer"] p { font-size: 12.5px !important; }

        /* ── CARD (st.container border=True) ───────────────────── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 8px !important;
            transition: transform var(--transition),
                        box-shadow var(--transition),
                        border-color var(--transition) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-md) !important;
            border-color: var(--border-hover) !important;
        }

        /* ── METRIC ────────────────────────────────────────────── */
        div[data-testid="stMetricLabel"] { margin-bottom: 4px !important; }
        div[data-testid="stMetricLabel"] * {
            color: var(--text-secondary) !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.07em !important;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-primary) !important;
            font-size: 26px !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stMetricDelta"] { margin-top: 5px !important; }
        div[data-testid="stMetricDelta"] * {
            font-size: 12px !important;
            font-weight: 600 !important;
        }

        .drone-note {
            display: block;
            color: var(--text-muted) !important;
            font-size: 12px !important;
            margin-top: 10px !important;
            padding-top: 9px !important;
            border-top: 1px solid var(--border) !important;
            line-height: 1.55 !important;
        }

        /* ── SIDEBAR ───────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid rgba(255,255,255,.05) !important;
        }
        section[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
        section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
            display: none !important;
        }
        [data-testid="stSidebarNav"] { display: none !important; }

        /* ── FORM INPUTS ───────────────────────────────────────── */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            background: var(--bg-card-alt) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            font-size: 14px !important;
            transition: border-color var(--transition),
                        box-shadow var(--transition) !important;
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
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 10px 24px !important;
            transition: opacity var(--transition) !important;
        }
        [data-testid="stFormSubmitButton"] button:hover { opacity: .88 !important; }

        /* ── SLIDER ────────────────────────────────────────────── */
        [data-testid="stSlider"] [role="slider"] {
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        [data-testid="stSelectbox"] > div > div {
            border-radius: var(--radius-md) !important;
            border-color: var(--border) !important;
        }

        /* ── STATUS PILL ───────────────────────────────────────── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .dss-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }

        /* ── DIVIDER ───────────────────────────────────────────── */
        hr[data-testid="stDivider"] {
            border-color: var(--border) !important;
            margin: 16px 0 !important;
        }

        /* ── DATAFRAME ─────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
        }

        /* ── EXPANDER ──────────────────────────────────────────── */
        [data-testid="stExpander"] {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            font-size: 14px !important;
            color: var(--text-primary) !important;
        }

        /* ── PROGRESS BAR (shared) ─────────────────────────────── */
        .prog-track {
            background: #f1f3f9;
            border-radius: 6px;
            height: 6px;
            overflow: hidden;
            margin: 8px 0 3px;
        }
        .prog-fill {
            height: 100%;
            border-radius: 6px;
            transition: width .6s cubic-bezier(.4,0,.2,1);
        }
        .prog-label { font-size: 12px; color: var(--text-muted); margin: 0; }

        /* ── HIDE STREAMLIT CHROME ─────────────────────────────── */
        /* KHÔNG dùng display:none cho stHeader / stToolbar: nút mở lại
           sidebar (stExpandSidebarButton) nằm BÊN TRONG chúng — ẩn cả cụm
           thì thu gọn sidebar xong sẽ không mở lại được.
           Header vốn position:absolute (không đẩy layout) và nền trùng màu
           nền trang nên đã vô hình; chỉ cần tắt pointer-events để nó không
           chặn click vào nội dung bên dưới. */
        header[data-testid="stHeader"] { pointer-events: none !important; }
        [data-testid="stToolbar"]      { pointer-events: none !important; }

        [data-testid="stHeaderActionElements"] { display: none !important; }
        footer                              { visibility: hidden !important; }
        [data-testid="stDecoration"]        { display: none !important; }
        [data-testid="stStatusWidget"]      { display: none !important; }
        .stDeployButton                     { display: none !important; }
        #MainMenu                           { display: none !important; }

        /* Nút mở lại sidebar — bật lại pointer-events (cha đang tắt) và tạo
           kiểu khớp ngôn ngữ thiết kế: khối tối bo tròn, gợi đúng hình ảnh
           thanh sidebar sắp trượt ra. */
        [data-testid="stExpandSidebarButton"] {
            pointer-events: auto !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 38px !important;
            height: 38px !important;
            margin: 8px 0 0 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: var(--sidebar-bg) !important;
            color: #fff !important;
            border: 1px solid rgba(255, 255, 255, .10) !important;
            border-radius: var(--radius-md) !important;
            box-shadow: 0 4px 14px rgba(27, 30, 48, .28),
                        0 1px 3px rgba(0, 0, 0, .16) !important;
            transition: background var(--transition),
                        transform var(--transition),
                        box-shadow var(--transition) !important;
        }
        [data-testid="stExpandSidebarButton"]:hover {
            background: var(--accent) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 22px rgba(79, 99, 210, .36),
                        0 2px 6px rgba(0, 0, 0, .14) !important;
        }
        [data-testid="stExpandSidebarButton"]:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 8px rgba(27, 30, 48, .30) !important;
        }
        [data-testid="stExpandSidebarButton"]:focus-visible {
            outline: 2px solid var(--accent) !important;
            outline-offset: 2px !important;
        }
        /* Icon (material font) — nhích nhẹ sang phải khi hover, gợi ý mở ra */
        [data-testid="stExpandSidebarButton"] span {
            font-size: 20px !important;
            line-height: 1 !important;
            transition: transform var(--transition) !important;
        }
        [data-testid="stExpandSidebarButton"]:hover span {
            transform: translateX(2px) !important;
        }

        [data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; }
        section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem !important; }
        hr { border-top: 1px solid var(--border) !important; }

        [data-testid="stCaptionContainer"] {
            font-size: 12.5px !important;
            color: var(--text-secondary) !important;
            line-height: 1.55 !important;
            padding: 0 4px !important;
            margin-top: 4px !important;
        }
        .js-plotly-plot .plotly .modebar { display: none !important; }

        /* ── DESIGN SYSTEM: HERO KẾT LUẬN ──────────────────────── */
        .conc-hero {
            position: relative; overflow: hidden;
            background: var(--hero-bg);
            border-radius: var(--radius-lg);
            padding: 26px 28px;
            box-shadow: var(--shadow-md);
            margin-bottom: 4px;
        }
        .conc-hero .ch-glow {
            position: absolute; top: -70px; right: -50px;
            width: 260px; height: 260px; pointer-events: none;
            background: radial-gradient(circle, rgba(200,232,90,.14) 0%, transparent 66%);
        }
        /* !important + độ đặc hiệu cao: thắng rule toàn cục
           [data-testid="stMarkdownContainer"] p (nhuộm đen mọi <p>) —
           nếu không, chữ trắng trong hero nền tối sẽ tàng hình. */
        [data-testid="stMarkdownContainer"] .conc-hero .ch-eyebrow {
            position: relative; margin: 0 0 10px;
            font-size: 12px; font-weight: 700; letter-spacing: .13em;
            text-transform: uppercase;
            color: rgba(255,255,255,.45) !important;
        }
        [data-testid="stMarkdownContainer"] .conc-hero .ch-headline {
            position: relative; margin: 0;
            font-size: 21px; line-height: 1.5 !important; font-weight: 700;
            color: #ffffff !important; max-width: 78ch;
        }
        [data-testid="stMarkdownContainer"] .conc-hero .ch-headline .hl {
            color: var(--lime) !important; font-weight: 700;
        }
        .conc-hero .ch-stats {
            position: relative; display: flex; flex-wrap: wrap; gap: 34px;
            margin-top: 20px; padding-top: 18px;
            border-top: 1px solid rgba(255,255,255,.10);
        }
        .conc-hero .ch-stat-v {
            font-family: 'JetBrains Mono', monospace;
            font-size: 30px; font-weight: 700; color: #fff; line-height: 1.1;
        }
        .conc-hero .ch-stat-l {
            font-size: 12.5px; color: rgba(255,255,255,.55);
            margin-top: 4px; letter-spacing: .02em;
        }

        /* ── DESIGN SYSTEM: CALLOUT ────────────────────────────── */
        .dss-callout {
            display: flex; gap: 13px; align-items: flex-start;
            border-radius: var(--radius-md);
            padding: 15px 18px; margin: 4px 0 6px;
            background: var(--accent-light);
            border: 1px solid rgba(79,99,210,.16);
        }
        .dss-callout.tone-success { background: var(--c-success-bg); border-color: rgba(22,163,74,.18); }
        .dss-callout.tone-warning { background: var(--c-warning-bg); border-color: rgba(217,119,6,.18); }
        .dss-callout.tone-danger  { background: var(--c-danger-bg);  border-color: rgba(220,38,38,.18); }
        .dss-callout .co-icon { font-size: 17px; line-height: 1.4; }
        .dss-callout .co-title {
            margin: 0 0 3px; font-size: 13.5px; font-weight: 700; color: var(--text-primary);
        }
        .dss-callout .co-text {
            margin: 0; font-size: 13px; line-height: 1.5; color: var(--text-secondary);
        }

        /* ── DESIGN SYSTEM: PANEL + THANH SO SÁNH ──────────────── */
        .dss-panel {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius-md); padding: 20px 22px;
            box-shadow: var(--shadow-sm); height: 100%;
        }
        .dss-panel-title {
            margin: 0 0 16px; font-size: 17px; font-weight: 700; color: var(--text-primary);
        }
        .cb-row { margin-bottom: 14px; }
        .cb-head {
            display: flex; justify-content: space-between; align-items: baseline;
            font-size: 13.5px; color: var(--text-secondary); margin-bottom: 6px;
        }
        .cb-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; }
        .cb-val.tone-success { color: var(--c-success); }
        .cb-val.tone-danger  { color: var(--c-danger); }
        .cb-val.tone-primary { color: var(--accent); }
        .cb-track {
            height: 8px; border-radius: 999px; background: var(--bg-card-alt); overflow: hidden;
        }
        .cb-fill { height: 100%; border-radius: 999px; }
        .cb-fill.tone-success { background: var(--c-success); }
        .cb-fill.tone-danger  { background: var(--c-danger); }
        .cb-fill.tone-primary { background: var(--accent); }
        .cb-note {
            margin: 12px 0 0; font-size: 12px; line-height: 1.5; color: var(--text-muted);
        }

        /* ── DESIGN SYSTEM: THẺ QUYẾT ĐỊNH ─────────────────────── */
        .dc-card { display: flex; flex-direction: column; }
        .dc-role {
            margin: 0 0 6px; font-size: 12px; font-weight: 700;
            letter-spacing: .13em; text-transform: uppercase; color: var(--accent);
        }
        .dc-question {
            margin: 0 0 14px; font-size: 14.5px; font-weight: 600;
            line-height: 1.5; color: var(--text-primary);
        }
        .dc-foot {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: auto; padding-top: 12px; border-top: 1px solid var(--border);
        }
        .dc-acc {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px; font-weight: 700; color: var(--text-primary);
        }
        .dc-acc-l { font-size: 12px; color: var(--text-muted); margin-left: 5px; }
        .dc-target {
            font-family: 'JetBrains Mono', monospace; font-size: 12px;
            background: var(--bg-card-alt); color: var(--text-secondary);
            padding: 3px 8px; border-radius: 6px;
        }
        .dc-extra {
            margin: 10px 0 0; font-size: 12px; line-height: 1.5; color: var(--text-muted);
        }

        /* ── DESIGN SYSTEM: MODEL DETAIL (tab 2 & 3) ───────────── */
        .mi-head {
            display: flex; justify-content: space-between; align-items: center;
            gap: 18px; flex-wrap: wrap;
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius-md); padding: 18px 22px;
            box-shadow: var(--shadow-sm); margin-bottom: 4px;
        }
        .mi-head .mh-title { margin: 0; font-size: 17px; font-weight: 700;
                             color: var(--text-primary); }
        .mi-head .mh-role  {
            display: inline-block; margin-top: 5px;
            font-size: 12px; font-weight: 700; letter-spacing: .13em;
            text-transform: uppercase; color: var(--accent);
            background: var(--accent-light); border-radius: 999px; padding: 3px 10px;
        }
        .mi-head .mh-acc   {
            font-family: 'JetBrains Mono', monospace; text-align: right;
            font-size: 32px; font-weight: 700; color: var(--accent); line-height: 1.05;
        }
        .mi-head .mh-acc-l { font-size: 12.5px; color: var(--text-muted);
                             text-align: right; margin-top: 3px; }

        .cv-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius-md); padding: 16px 18px;
            box-shadow: var(--shadow-sm); text-align: center;
        }
        .cv-card.ok { border-color: rgba(22,163,74,.4); }
        .cv-card .cv-v {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px; font-weight: 700; color: var(--text-primary);
        }
        .cv-card.ok .cv-v { color: var(--c-success); }
        .cv-card .cv-l { font-size: 12.5px; color: var(--text-muted); margin-top: 4px; }

        /* Pill selector (st.segmented_control) — pill active nền primary.
           DOM thật của Streamlit 1.58: [data-testid="stButtonGroup"] chứa
           button[kind="segmented_control"|"segmented_controlActive"]. */
        [data-testid="stButtonGroup"] button { border-radius: 999px !important; }
        [data-testid="stButtonGroup"] button p { font-size: 14px !important; font-weight: 600 !important; }
        button[kind="segmented_controlActive"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        button[kind="segmented_controlActive"] p { color: #fff !important; }

        /* ─────────────────────────────────────────────────────────
           ANIMATIONS — tôn trọng prefers-reduced-motion
           ───────────────────────────────────────────────────────── */
        @keyframes droneFadeUp      { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
        @keyframes droneBarRise     { from { transform:scaleY(0); }                  to { transform:scaleY(1); } }
        @keyframes dronePulse       { 0%,100% { opacity:1; transform:scale(1); }     50% { opacity:.4; transform:scale(.6); } }
        @keyframes droneGaugeReveal { from { stroke-dashoffset:0; }                  to { stroke-dashoffset:-101; } }

        /* Tắt animation nếu user đã set "reduce motion" trong OS */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* ── HERO PANEL (dark) ─────────────────────────────────── */
        .dss-hero {
            background: var(--hero-bg);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 18px;
            animation: droneFadeUp .6s cubic-bezier(.2,.7,.3,1) both;
        }
        .dss-hero-grid {
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 16px;
        }
        @media (max-width: 900px) {
            .dss-hero-grid { grid-template-columns: 1fr; }
        }
        .dss-hero-sub {
            background: var(--hero-card);
            border-radius: 14px;
            padding: 16px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .dss-hero-eyebrow {
            color: rgba(255,255,255,.55);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin: 0 0 6px 0 !important;
        }
        .dss-hero-big {
            color: #fff;
            font-size: 32px;
            font-weight: 700;
            margin: 0;
            line-height: 1.1;
            font-family: 'JetBrains Mono', monospace;
        }
        .dss-hero-unit {
            font-size: 13px;
            color: rgba(255,255,255,.5);
            font-weight: 500;
            font-family: 'Inter', sans-serif;
        }
        .dss-bars {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 64px;
            margin-top: 16px;
        }
        .dss-bar {
            flex: 1;
            border-radius: 2px 2px 0 0;
            transform-origin: bottom;
            transform: scaleY(0);
            animation: droneBarRise .8s cubic-bezier(.16,1,.3,1) forwards;
            transition: filter .2s, transform .2s;
        }
        .dss-bar:hover {
            filter: brightness(1.35);
            cursor: pointer;
        }
        .dss-axis {
            display: flex;
            justify-content: flex-start;
            gap: 3px;
            color: rgba(255,255,255,.3);
            font-size: 10px;
            margin-top: 6px;
            font-family: 'JetBrains Mono', monospace;
        }
        .dss-gseg {
            fill: none;
            stroke-width: 14;
            stroke-linecap: round;
            transition: stroke-dasharray 1s ease, stroke-dashoffset 1s ease;
        }
        .dss-gcover {
            fill: none;
            stroke: rgba(255,255,255,0.06);
            stroke-width: 14;
            stroke-linecap: round;
        }
        .dss-legend-row {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13.5px;
            color: rgba(255,255,255,.7);
            width: 130px;
            margin-bottom: 2px;
        }
        .dss-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .dss-pulse { animation: dronePulse 1.4s ease-in-out infinite; }

        /* ── KPI TILES ─────────────────────────────────────────── */
        .dss-kpis {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 18px;
        }
        @media (max-width: 900px) {
            .dss-kpis { grid-template-columns: repeat(2, 1fr); }
        }
        .dss-kpi {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 15px 16px;
            box-shadow: var(--shadow-sm);
            animation: droneFadeUp .6s cubic-bezier(.2,.7,.3,1) both;
            transition: transform .25s, box-shadow .25s, border-color .25s;
        }
        .dss-kpi:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
            border-color: var(--border-hover);
        }
        .dss-kpi-label {
            display: flex;
            align-items: center;
            gap: 7px;
            color: var(--text-secondary);
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: .04em;
            text-transform: uppercase;
            margin: 0;
        }
        .dss-kpi-icon { color: var(--accent); }
        .dss-kpi-value {
            font-size: 32px;
            font-weight: 700;
            margin: 7px 0 0;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -.01em;
        }
        .dss-kpi-delta { font-size: 12px; font-weight: 600; margin: 3px 0 0; }

        /* ── LIVE BUTTON ───────────────────────────────────────── */
        .dss-live {
            background: var(--lime);
            color: var(--lime-ink);
            border: none;
            border-radius: 11px;
            padding: 8px 15px;
            font-size: 12.5px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 7px;
        }

        /* ── RECORDS TABLE ─────────────────────────────────────── */
        .dss-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
        .dss-table th {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .05em;
            text-transform: uppercase;
            text-align: left;
            padding: 8px 6px;
        }
        .dss-table td {
            padding: 11px 6px;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
        }
        .dss-table td b {
            color: var(--text-primary);
            font-weight: 700;
        }
        .dss-table tbody tr      { transition: background .15s; }
        .dss-table tbody tr:hover { background: var(--bg-card-alt); }

        /* ── SIDEBAR NAV BUTTONS (gom từ render_sidebar_nav) ───── */
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background: rgba(255,255,255,0.04) !important;
            color: rgba(255,255,255,0.72) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: var(--radius-md) !important;
            padding: 11px 14px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            text-align: left !important;
            margin-bottom: 6px !important;
            box-shadow: none !important;
            justify-content: flex-start !important;
            transition: all .15s ease !important;
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
            border-radius: var(--radius-md) !important;
            padding: 11px 14px !important;
            font-size: 14px !important;
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

        /* ── TYPE SCALE bổ sung (design handoff) ───────────────── */
        .stTabs [data-baseweb="tab"] p {
            font-size: 14px !important; font-weight: 600 !important;
        }
        .stButton button p, .stDownloadButton button p {
            font-size: 14px !important; font-weight: 600 !important;
        }
        [data-testid="stTable"] table { font-size: 13.5px !important; }

        /* ── TYPE SCALE enforcement ─────────────────────────────
           Streamlit (emotion) có rule `.st-emotion-xxx p {font-size:inherit}`
           đặc hiệu 0-1-1, THẮNG mọi class đơn 0-1-0 của ta → cỡ chữ class
           bị vô hiệu. Khối này ép lại bằng !important cho toàn bộ thang. */
        .dss-kpi-value   { font-size: 32px !important; }
        .dss-kpi-label   { font-size: 12.5px !important; }
        .dss-kpi-delta   { font-size: 12px !important; }
        .dss-hero-eyebrow{ font-size: 12px !important; }
        .dss-hero-big    { font-size: 32px !important; }
        .dss-hero-unit   { font-size: 13px !important; }
        .prog-label      { font-size: 12px !important; }
        .status-badge, .dss-pill { font-size: 12px !important; }
        .dss-live        { font-size: 12.5px !important; }
        .dss-legend-row  { font-size: 13.5px !important; }
        .dss-table       { font-size: 13.5px !important; }
        .dss-table th    { font-size: 12px !important; }
        .dss-panel-title { font-size: 17px !important; }
        .cb-head  { font-size: 13.5px !important; }
        .cb-val   { font-size: 15px !important; }
        .cb-note  { font-size: 12px !important; }
        .co-title { font-size: 13.5px !important; }
        .co-text  { font-size: 13px !important; }
        .dc-role  { font-size: 12px !important; }
        .dc-question { font-size: 14.5px !important; }
        .dc-acc   { font-size: 26px !important; }
        .dc-acc-l, .dc-target, .dc-extra { font-size: 12px !important; }
        .mh-title { font-size: 17px !important; }
        .mh-role  { font-size: 12px !important; }
        .mh-acc   { font-size: 32px !important; }
        .mh-acc-l { font-size: 12.5px !important; }
        .cv-v { font-size: 26px !important; }
        .cv-l { font-size: 12.5px !important; }
        .ch-eyebrow { font-size: 12px !important; }
        .ch-headline{ font-size: 21px !important; }
        .ch-stat-v  { font-size: 30px !important; }
        .ch-stat-l  { font-size: 12.5px !important; }
        /* Hai rule dưới cần đặc hiệu ≥ (0,3,1) để thắng rule body
           p:not([class]):not([style]) ở trên (đặt sau nên thắng khi hòa). */
        [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p:not([class]) {
            font-size: 12.5px !important; font-weight: 600 !important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] p:not([class]) {
            font-size: 14px !important; font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ═════════════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, alpha: float = 0.09) -> str:
    """
    Convert 6-char hex color to rgba() string.
    Plotly's fillcolor không nhận 8-char hex (#rrggbbaa) — phải dùng rgba().
    Validates input và fallback an toàn.
    """
    if not isinstance(hex_color, str):
        return f"rgba(79,99,210,{alpha})"  # fallback to accent
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(79,99,210,{alpha})"
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({r},{g},{b},{a})"
    except (ValueError, TypeError):
        return f"rgba(79,99,210,{alpha})"


def create_sparkline(data, color: str = "#4f63d2") -> go.Figure:
    """Minimal sparkline: spline + subtle fill. No axes, no tooltips."""
    pts = list(data)
    if not pts:
        return go.Figure()  # empty fig fallback

    fill = _hex_to_rgba(color, alpha=0.09)
    fig = go.Figure()
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


# ═════════════════════════════════════════════════════════════════════
#  RENDER — TOP NAV & TITLES
# ═════════════════════════════════════════════════════════════════════

def render_top_nav(title: str = "Dashboard",
                   subtitle: str = "Drone Decision Support System  ·  DSS301"):
    """
    Top bar: page title + live indicator + drone icon + user avatar.

    Args:
        title:    Page title shown in top-left (default "Dashboard").
        subtitle: Caption below title.
    """
    c_title, c_profile = st.columns([7, 3], vertical_alignment="center")

    with c_title:
        st.markdown(
            f"""
                <div style="padding:4px 0;">
                    <h2 style="margin:0;font-size:17px;font-weight:700;
                               letter-spacing:-0.025em;color:var(--text-primary);">
                        {title}
                    </h2>
                    <p style="margin:2px 0 0;font-size:12.5px;
                              color:var(--text-secondary);font-weight:400;">
                        {subtitle}
                    </p>
                </div>
                """,
            unsafe_allow_html=True,
        )

    with c_profile:
        st.markdown(
            """
            <div style="display:flex;justify-content:flex-end;align-items:center;gap:14px;">
                <span class="dss-live" title="Giám sát trực tiếp">
                    <span class="dss-dot dss-pulse" style="background:var(--lime-ink);"></span>
                    Giám sát trực tiếp
                </span>
                <span style="font-size:20px;cursor:pointer;color:var(--text-secondary);"
                      title="Quản lý Fleet">🚁</span>
                <div style="width:34px;height:34px;border-radius:50%;
                            background:var(--accent-light);display:flex;
                            justify-content:center;align-items:center;
                            font-weight:700;font-size:12px;color:var(--accent);
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
        f"<p style='font-size:14.5px;color:var(--text-secondary);margin:4px 0 0;line-height:1.5;max-width:680px;'>{subtitle}</p>"
        if subtitle else ""
    )
    st.markdown(
        f"""<div style="margin-bottom:16px;">
                <h3 style="font-weight:800;font-size:28px;letter-spacing:-0.02em;
                           margin:0;color:var(--text-primary);">{title}</h3>{sub}
            </div>""",
        unsafe_allow_html=True,
    )


def render_section_label(text: str):
    """Eyebrow label — small, uppercase, letter-spaced."""
    st.markdown(
        f"""<p style="font-size:17px;font-weight:700;
                      color:var(--text-primary);margin:22px 0 10px;">
                {text}</p>""",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
#  RENDER — METRIC CARDS
# ═════════════════════════════════════════════════════════════════════

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
                width="stretch",
                config={"displayModeBar": False},
            )


def render_risk_score(score: float, risk_level: str, note: str = ""):
    """
    Risk score card with colour-coded progress bar + badge.
    Levels: High / Medium / Low
    """
    risk_level = str(risk_level).strip().title()
    _P = {
        "High": dict(bar="#dc2626", bg="#fee2e2", fg="#b91c1c", delta="inverse", dot="🔴"),
        "Medium": dict(bar="#d97706", bg="#fef3c7", fg="#b45309", delta="off", dot="🟡"),
        "Low": dict(bar="#16a34a", bg="#dcfce7", fg="#15803d", delta="normal", dot="🟢"),
    }
    p = _P.get(risk_level,
               dict(bar="#4f63d2", bg="#eef0fb", fg="#4f63d2",
                    delta="normal", dot="🔵"))
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


# ═════════════════════════════════════════════════════════════════════
#  RENDER — STATUS PILLS, HERO, KPI TILES, TABLE
# ═════════════════════════════════════════════════════════════════════

_PILL_COLORS = {
    "success": ("#dcfce7", "#15803d", "#16a34a"),
    "warning": ("#fef3c7", "#b45309", "#d97706"),
    "danger": ("#fee2e2", "#b91c1c", "#dc2626"),
    "info": ("#e0f2fe", "#0369a1", "#0284c7"),
}


def status_pill_html(label: str, level: str = "info") -> str:
    """Return a status-pill HTML snippet (coloured background + dot)."""
    bg, fg, dot = _PILL_COLORS.get(level, _PILL_COLORS["info"])
    return (f"<span class='dss-pill' style='background:{bg};color:{fg};'>"
            f"<span class='dss-dot' style='background:{dot};'></span>{label}</span>")


def render_hero_panel(total_records: int, avg_score: float,
                      low_pct: float, med_pct: float, high_pct: float,
                      bar_heights, counts=None, delta_str: str = "+3.2% tuần này"):
    """
    Dark hero panel — risk-score distribution bars + semicircle risk gauge.

    Args:
        total_records: tổng số records (hiển thị số lớn)
        avg_score:     điểm rủi ro trung bình (0-10)
        low_pct, med_pct, high_pct: phần trăm phân bổ (cộng ≈ 100)
        bar_heights:   list 11 phần tử (0-100) — chiều cao % của từng cột
        counts:        list 11 phần tử — số bản ghi gốc, để hiện tooltip (optional)
        delta_str:     text hiển thị trong badge bên phải
    """
    # Build các thanh bar — escape values cẩn thận, không gọi inline <style>
    bars_parts = []
    for i, h in enumerate(bar_heights):
        h = max(2.0, min(100.0, float(h)))
        color = "var(--lime)" if h >= 70 else "rgba(255,255,255,.18)"
        tip = f" title='Điểm {i}: {counts[i]:,} bản ghi'" if counts else ""
        bars_parts.append(
            f"<div class='dss-bar' style='height:{h:.0f}%;background:{color};"
            f"animation-delay:{i * 0.03:.2f}s'{tip}></div>"
        )
    bars = "".join(bars_parts)

    # Axis labels — chỉ hiện ở số chẵn
    axis_parts = [
        f"<span style='flex:1;text-align:center;'>{i if i % 2 == 0 else ''}</span>"
        for i in range(len(bar_heights))
    ]
    axis_labels = "".join(axis_parts)

    off_med = -low_pct
    off_high = -(low_pct + med_pct)

    html_str = f"""
        <div class="dss-hero">
          <div class="dss-hero-grid">

            <!-- LEFT: Distribution bars -->
            <div class="dss-hero-sub">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <p class="dss-hero-eyebrow">Phân phối điểm rủi ro</p>
                  <p style="font-size:12.5px;color:rgba(255,255,255,.55);margin:0 0 10px 0;line-height:1.5;max-width:85%;">
                    Số lượng bản ghi theo thang điểm rủi ro (0-10). Di chuột vào cột để xem chi tiết.
                  </p>
                  <p class="dss-hero-big">{total_records:,} <span class="dss-hero-unit">bản ghi</span></p>
                </div>
                <span class="dss-pill" style="background:var(--lime-soft);color:var(--lime);">↗ {delta_str}</span>
              </div>
              <div class="dss-bars">{bars}</div>
              <div class="dss-axis">{axis_labels}</div>
            </div>

            <!-- RIGHT: Gauge -->
            <div class="dss-hero-sub">
              <div>
                <p class="dss-hero-eyebrow">Tổng quan rủi ro</p>
                <p style="font-size:12.5px;color:rgba(255,255,255,.55);margin:0 0 8px 0;line-height:1.5;">
                  Điểm trung bình và tỷ lệ phân bổ trạng thái bay của toàn hệ thống.
                </p>
              </div>
              <div style="display:flex;align-items:center;gap:12px;margin-top:auto;">
                <svg viewBox="0 0 220 128" width="130" style="overflow:visible;flex-shrink:0;" role="img"
                     aria-label="Risk score gauge {avg_score:.1f} of 10">
                  <path class="dss-gcover" d="M28,110 A82,82 0 0 1 192,110" pathLength="100" stroke-dasharray="100 100"/>
                  <path class="dss-gseg" d="M28,110 A82,82 0 0 1 192,110" stroke="#3ad17e" pathLength="100"
                        stroke-dasharray="{low_pct:.1f} 100" stroke-dashoffset="0"/>
                  <path class="dss-gseg" d="M28,110 A82,82 0 0 1 192,110" stroke="#f0b429" pathLength="100"
                        stroke-dasharray="{med_pct:.1f} 100" stroke-dashoffset="{off_med:.1f}"/>
                  <path class="dss-gseg" d="M28,110 A82,82 0 0 1 192,110" stroke="#f0584f" pathLength="100"
                        stroke-dasharray="{high_pct:.1f} 100" stroke-dashoffset="{off_high:.1f}"/>
                  <text x="110" y="98" text-anchor="middle" fill="#fff" font-size="32" font-weight="700"
                        font-family="JetBrains Mono, monospace">{avg_score:.1f}</text>
                  <text x="110" y="118" text-anchor="middle" fill="rgba(255,255,255,.5)" font-size="11"
                        font-weight="600">/ 10 điểm TB</text>
                </svg>
                <div style="display:flex;flex-direction:column;gap:10px;width:100%;">
                  <div class="dss-legend-row"><span class="dss-dot" style="background:#3ad17e;"></span>
                    An toàn <b style="color:#fff;margin-left:auto;">{low_pct:.0f}%</b></div>
                  <div class="dss-legend-row"><span class="dss-dot" style="background:#f0b429;"></span>
                    Cảnh báo <b style="color:#fff;margin-left:auto;">{med_pct:.0f}%</b></div>
                  <div class="dss-legend-row"><span class="dss-dot" style="background:#f0584f;"></span>
                    Nguy hiểm <b style="color:#fff;margin-left:auto;">{high_pct:.0f}%</b></div>
                </div>
              </div>
            </div>

          </div>
        </div>
        """

    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)


def render_kpi_tiles(tiles: list):
    """
    Bright KPI-tile row (4-up grid, responsive 2-up trên mobile).
    Each tile is a dict:
      { 'icon': '🔋', 'label': 'PIN TB', 'value': '67%',
        'delta': '+4% tuần này', 'delta_color': '#16a34a' }
    """
    parts = []
    for i, t in enumerate(tiles):
        delta_col = t.get("delta_color", "var(--text-secondary)")
        parts.append(
            f"<div class='dss-kpi' style='animation-delay:{0.05 + i * 0.07:.2f}s;'>"
            f"<p class='dss-kpi-label'>"
            f"<span class='dss-kpi-icon'>{t.get('icon', '')}</span>"
            f"{t['label']}</p>"
            f"<p class='dss-kpi-value'>{t['value']}</p>"
            f"<p class='dss-kpi-delta' style='color:{delta_col};'>{t.get('delta', '')}</p>"
            f"</div>"
        )
    st.markdown(f"<div class='dss-kpis'>{''.join(parts)}</div>", unsafe_allow_html=True)


def render_drone_records_table(rows: list, title: str = "Bản ghi drone"):
    """
    Records table with coloured status pills.
    rows: list of dict { 'id', 'battery', 'wind', 'status_label', 'status_level' }
    """
    body_parts = []
    for r in rows:
        pill = status_pill_html(r["status_label"], r.get("status_level", "info"))
        body_parts.append(
            f"<tr><td><b>🚁 {r['id']}</b></td>"
            f"<td>{r['battery']}</td>"
            f"<td>{r['wind']}</td>"
            f"<td>{pill}</td></tr>"
        )
    body = "".join(body_parts)
    st.markdown(
        f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:var(--radius-lg);box-shadow:var(--shadow-sm);
                        padding:16px;margin-bottom:18px;">
              <p style="font-size:17px;font-weight:700;margin:0 0 12px;color:var(--text-primary);">
                 {title}
                 <span style="background:var(--bg-card-alt);color:var(--text-secondary);
                              font-size:12px;font-weight:700;padding:2px 8px;
                              border-radius:7px;margin-left:6px;">{len(rows)}</span>
              </p>
              <table class="dss-table">
                <thead><tr><th>Drone</th><th>Pin TB</th><th>Gió TB</th><th>Rủi ro</th></tr></thead>
                <tbody>{body}</tbody>
              </table>
            </div>
            """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
#  RENDER — BANNERS & BADGES
# ═════════════════════════════════════════════════════════════════════

_ALERT_FN = {"danger": st.error, "warning": st.warning,
             "success": st.success, "info": st.info}
_ALERT_ICON = {"danger": "⛔", "warning": "⚠️", "success": "✅", "info": "ℹ️"}


def render_result_badge(label: str, value: str, level: str = "info"):
    """Streamlit alert with icon, used for prediction results."""
    body = f"**{label}:** {value}" if label else value
    _ALERT_FN.get(level, st.info)(body, icon=_ALERT_ICON.get(level, "ℹ️"))


def render_banner(text: str, level: str = "info"):
    """Short notification banner (wrapper for render_result_badge)."""
    render_result_badge("", text, level)


# ═════════════════════════════════════════════════════════════════════
#  RENDER — DESIGN SYSTEM COMPONENTS (theo bản thiết kế Drone DSS)
# ═════════════════════════════════════════════════════════════════════

def render_conclusion_hero(headline_html: str, stats: list):
    """
    Hero "Kết luận trong 1 câu" — nền tối, vệt sáng lime, kèm dãy số lớn.

    headline_html : câu kết luận (cho phép <b class='hl'>…</b> để tô lime)
    stats         : list of (value, label) — các số lớn hiển thị bên dưới
    """
    stat_html = "".join(
        f"<div class='ch-stat'><div class='ch-stat-v'>{v}</div>"
        f"<div class='ch-stat-l'>{l}</div></div>"
        for v, l in stats
    )
    st.markdown(
        f"""
        <div class="conc-hero">
          <div class="ch-glow"></div>
          <p class="ch-eyebrow">Kết luận trong 1 câu</p>
          <p class="ch-headline">{headline_html}</p>
          <div class="ch-stats">{stat_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(title: str, body: str, icon: str = "💡", tone: str = "primary"):
    """Callout giải thích — nền nhạt theo tone (primary/success/warning/danger)."""
    st.markdown(
        f"""
        <div class="dss-callout tone-{tone}">
          <span class="co-icon">{icon}</span>
          <div class="co-body">
            <p class="co-title">{title}</p>
            <p class="co-text">{body}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_bars(title: str, rows: list, note: str = ""):
    """
    Thẻ insight có thanh ngang so sánh.
    rows: list of (label, value_text, pct_width 0-100, tone)
    """
    bars = "".join(
        f"""<div class="cb-row">
              <div class="cb-head"><span>{lbl}</span><b class="cb-val tone-{tone}">{val}</b></div>
              <div class="cb-track"><div class="cb-fill tone-{tone}" style="width:{pct}%"></div></div>
            </div>"""
        for lbl, val, pct, tone in rows
    )
    note_html = f"<p class='cb-note'>{note}</p>" if note else ""
    st.markdown(
        f"""<div class="dss-panel">
              <p class="dss-panel-title">{title}</p>
              {bars}{note_html}
            </div>""",
        unsafe_allow_html=True,
    )


def render_decision_card(role: str, question: str, target: str,
                         accuracy: str, extra: str = ""):
    """Thẻ 'mô hình phục vụ quyết định nào cho ai'."""
    extra_html = f"<p class='dc-extra'>{extra}</p>" if extra else ""
    st.markdown(
        f"""<div class="dss-panel dc-card">
              <p class="dc-role">{role}</p>
              <p class="dc-question">{question}</p>
              <div class="dc-foot">
                <div><span class="dc-acc">{accuracy}</span>
                     <span class="dc-acc-l">accuracy</span></div>
                <code class="dc-target">{target}</code>
              </div>
              {extra_html}
            </div>""",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════
#  RENDER — MODEL INFO COMPONENTS
# ═════════════════════════════════════════════════════════════════════

def render_model_accuracy_card(model_name: str, accuracy: float,
                               cv_mean: float, cv_std: float,
                               trained_at: str = ""):
    """
    Accuracy card with colour-coded progress bar.
    Green >= 90%, Yellow >= 75%, Red < 75%.
    """
    pct = int(accuracy * 100)
    bar_color = "#16a34a" if accuracy >= 0.90 else "#d97706" if accuracy >= 0.75 else "#dc2626"
    display = model_name.replace("_", " ").title()

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
        f1 = float(f1_scores.get(cls, 0.0))
        pct = int(f1 * 100)

        if f1 >= 0.90:
            bar_col, txt_col, icon = "#16a34a", "#15803d", "🟢"
        elif f1 >= 0.70:
            bar_col, txt_col, icon = "#d97706", "#b45309", "🟡"
        else:
            bar_col, txt_col, icon = "#dc2626", "#b91c1c", "🔴"

        pr = precision.get(cls, 0) if precision else None
        rec = recall.get(cls, 0) if recall else None
        extra = f"P {pr:.2f}  ·  R {rec:.2f}" if (pr is not None and rec is not None) else ""

        short = cls if len(cls) <= 32 else cls[:30] + "…"

        st.markdown(
            f"""
                <div style="display:flex;align-items:center;gap:10px;
                            padding:5px 0;border-bottom:1px solid #eef0f5;
                            font-family:Inter,system-ui,sans-serif;">
                    <span style="font-size:13.5px;font-weight:600;color:#1c1e2e;
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
                                 font-size:15px;font-weight:600;
                                 color:{txt_col};min-width:38px;text-align:right;">
                        {f1:.2f}
                    </span>
                    <span style="font-size:12px;color:#9da3b5;
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
    Long labels are abbreviated; full name shown in hover.
    """
    import numpy as np
    cm_arr = np.array(cm)
    row_sum = cm_arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm_arr / row_sum

    def _short(name, n=18):
        return name if len(name) <= n else name[:n - 1] + "…"

    short_x = [_short(c, 16) for c in classes]
    short_y = [_short(c, 22) for c in classes]

    text = [[f"<b>{cm_arr[i][j]}</b><br>{cm_norm[i][j]:.0%}"
             for j in range(len(classes))]
            for i in range(len(classes))]

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


# ═════════════════════════════════════════════════════════════════════
#  RENDER — SIDEBAR
# ═════════════════════════════════════════════════════════════════════

def render_sidebar_header():
    """App logo + title in dark sidebar."""
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;
                    margin-bottom:26px;padding:2px 4px;">
            <div style="width:32px;height:32px;border-radius:8px;
                        background:rgba(79,99,210,.28);display:flex;
                        align-items:center;justify-content:center;
                        font-size:18px;line-height:1;">🚁</div>
            <div>
                <p style="margin:0;font-size:14px;font-weight:700;
                          color:#fff;letter-spacing:-0.02em;">Drone DSS</p>
                <p style="margin:0;font-size:11px;color:rgba(255,255,255,.35);
                          font-weight:500;letter-spacing:.05em;text-transform:uppercase;">
                    DSS301 &nbsp;·&nbsp; Warranty &amp; Ops</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav(items: list, key: str = "main_nav") -> str:
    """
    Render sidebar navigation as styled buttons.
    Uses Streamlit native buttons + CSS from load_css() (no inline <style>).

    Args:
        items: list of (icon, label) tuples
        key:   session_state key for persistence

    Returns:
        Label of the currently selected item
    """
    # Init session state
    if key not in st.session_state:
        st.session_state[key] = items[0][1]

    current = st.session_state[key]

    for i, (icon, label) in enumerate(items):
        is_active = (label == current)
        btn_type = "primary" if is_active else "secondary"

        if st.button(
                f"{icon}   {label}",
                key=f"{key}_btn_{i}",
                width="stretch",
                type=btn_type,
        ):
            st.session_state[key] = label
            st.rerun()

    return current


def render_sidebar_upgrade_card():
    """Stub — kept for backward compatibility, no longer renders anything."""
    return None