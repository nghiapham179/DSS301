import streamlit as st


def load_css():
    st.markdown("""
    <style>
        /* =========================
           GLOBAL
        ========================== */

        :root {
            --primary: #2563eb;
            --primary-soft: #dbeafe;
            --secondary: #14b8a6;
            --dark: #0f172a;
            --muted: #64748b;
            --card: rgba(255, 255, 255, 0.92);
            --border: #e2e8f0;
            --shadow: rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(37, 99, 235, 0.15), transparent 30%),
                radial-gradient(circle at 92% 6%, rgba(20, 184, 166, 0.14), transparent 32%),
                linear-gradient(180deg, #f1f8ff 0%, #f8fafc 45%, #ffffff 100%);
            color: var(--dark);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
            animation: pageFade 0.55s ease-out;
        }

        footer {
            visibility: hidden;
        }

        /* =========================
           SIDEBAR
        ========================== */

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(18px);
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            color: var(--dark);
        }

        div[role="radiogroup"] label {
            border-radius: 14px;
            padding: 8px 10px;
            transition: 0.2s ease;
        }

        div[role="radiogroup"] label:hover {
            background: #eff6ff;
        }

        /* =========================
           ANIMATION
        ========================== */

        @keyframes pageFade {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes floatIcon {
            0% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-6px);
            }
            100% {
                transform: translateY(0px);
            }
        }

        @keyframes pulseGlow {
            0% {
                box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.22);
            }
            70% {
                box-shadow: 0 0 0 16px rgba(37, 99, 235, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(37, 99, 235, 0);
            }
        }

        /* =========================
           TOP NAV
        ========================== */

        .top-nav {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 28px;
            padding: 20px 26px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow:
                0 20px 50px var(--shadow),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            animation: slideUp 0.5s ease-out;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .brand-icon {
            width: 54px;
            height: 54px;
            border-radius: 18px;
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            animation: floatIcon 3s ease-in-out infinite, pulseGlow 2.8s infinite;
            box-shadow: 0 14px 34px rgba(37, 99, 235, 0.25);
        }

        .brand {
            font-size: 27px;
            font-weight: 850;
            letter-spacing: -0.6px;
            color: var(--dark);
            line-height: 1.1;
        }

        .brand-small {
            font-size: 13px;
            color: var(--muted);
            margin-top: 5px;
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .nav-badge {
            background: linear-gradient(135deg, #eff6ff, #ecfeff);
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 9px 15px;
            font-size: 13px;
            font-weight: 700;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.14);
        }

        /* =========================
           TITLES
        ========================== */

        .main-title {
            font-size: 34px;
            font-weight: 850;
            letter-spacing: -0.7px;
            color: var(--dark);
            margin-bottom: 8px;
            animation: slideUp 0.5s ease-out;
        }

        .sub-title {
            font-size: 15px;
            color: var(--muted);
            margin-bottom: 26px;
            animation: slideUp 0.62s ease-out;
        }

        .section-title {
            font-size: 21px;
            font-weight: 800;
            color: var(--dark);
            letter-spacing: -0.3px;
            margin-bottom: 5px;
            animation: slideUp 0.45s ease-out;
        }

        .section-caption {
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 18px;
            animation: slideUp 0.55s ease-out;
        }

        /* =========================
           METRIC CARD
        ========================== */

        .metric-card {
            position: relative;
            overflow: hidden;
            min-height: 128px;
            padding: 22px 20px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 24px;
            text-align: center;
            box-shadow:
                0 16px 36px var(--shadow),
                inset 0 1px 0 rgba(255, 255, 255, 0.75);
            transition: all 0.25s ease;
            animation: slideUp 0.65s ease-out;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow:
                0 24px 50px rgba(15, 23, 42, 0.12),
                0 0 0 5px rgba(37, 99, 235, 0.04);
            border-color: #bfdbfe;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, #2563eb, #14b8a6);
        }

        .metric-card::after {
            content: "";
            position: absolute;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: rgba(37, 99, 235, 0.06);
            right: -45px;
            bottom: -45px;
        }

        .metric-label {
            position: relative;
            z-index: 1;
            font-size: 13px;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 8px;
        }

        .metric-value {
            position: relative;
            z-index: 1;
            font-size: 32px;
            font-weight: 900;
            color: var(--dark);
            letter-spacing: -0.6px;
        }

        .metric-note {
            position: relative;
            z-index: 1;
            font-size: 12px;
            color: #94a3b8;
            margin-top: 7px;
        }

        /* =========================
           SCORE BOX
        ========================== */

        .score-box {
            min-height: 250px;
            display: flex;
            align-items: center;
            padding: 28px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92));
            border: 1px solid var(--border);
            border-radius: 28px;
            box-shadow:
                0 18px 42px var(--shadow),
                inset 0 1px 0 rgba(255, 255, 255, 0.75);
            animation: slideUp 0.6s ease-out;
            transition: 0.25s ease;
        }

        .score-box:hover {
            transform: translateY(-4px);
            box-shadow: 0 26px 56px rgba(15, 23, 42, 0.12);
        }

        .score-main {
            font-size: 52px;
            font-weight: 950;
            color: var(--dark);
            letter-spacing: -1.2px;
        }

        .score-status {
            display: inline-block;
            margin-top: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: #fff7ed;
            font-size: 18px;
            font-weight: 850;
        }

        .score-note {
            max-width: 300px;
            margin-top: 14px;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.55;
        }

        /* =========================
           STREAMLIT COMPONENT POLISH
        ========================== */

        div[data-testid="stPlotlyChart"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 18px;
            box-shadow:
                0 16px 36px var(--shadow),
                inset 0 1px 0 rgba(255, 255, 255, 0.75);
            animation: slideUp 0.7s ease-out;
            transition: all 0.25s ease;
        }

        div[data-testid="stPlotlyChart"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.12);
            border-color: #bfdbfe;
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 18px;
            box-shadow: 0 14px 32px var(--shadow);
            transition: 0.25s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: #bfdbfe;
        }

        div[data-testid="stMetricValue"] {
            font-size: 26px;
            font-weight: 850;
            color: var(--dark);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
        }

        div[data-testid="stAlert"] {
            border-radius: 20px;
            animation: slideUp 0.45s ease-out;
        }

        .stSlider {
            padding: 8px 2px 14px 2px;
        }

        .stSelectbox {
            animation: slideUp 0.55s ease-out;
        }

        /* Input container effect */
        div[data-testid="stVerticalBlock"] > div:has(.stSlider) {
            animation: slideUp 0.65s ease-out;
        }

        /* =========================
           RESPONSIVE
        ========================== */

        @media screen and (max-width: 900px) {
            .top-nav {
                flex-direction: column;
                align-items: flex-start;
                gap: 16px;
            }

            .main-title {
                font-size: 27px;
            }

            .score-main {
                font-size: 40px;
            }

            .brand {
                font-size: 23px;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def render_top_nav():
    st.markdown("""
    <div class="top-nav">
        <div class="brand-wrap">
            <div class="brand-icon">🚁</div>
            <div>
                <div class="brand">Drone DSS</div>
                <div class="brand-small">Operation Risk • Maintenance Action • Final Recommendation</div>
            </div>
        </div>

        <div class="nav-actions">
            <div class="status-dot"></div>
            <div class="nav-badge">DSS301 Project</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_main_title(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sub-title">{subtitle}</div>',
        unsafe_allow_html=True
    )


def open_card():
    # Không bọc chart bằng HTML để tránh lỗi trắng màn hình Streamlit.
    st.write("")


def close_card():
    # Không bọc chart bằng HTML để tránh lỗi trắng màn hình Streamlit.
    st.write("")


def render_metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_title(title, caption=""):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if caption:
        st.markdown(
            f'<div class="section-caption">{caption}</div>',
            unsafe_allow_html=True
        )


def render_score_box(score, status, note, color):
    st.markdown(
        f"""
        <div class="score-box">
            <div>
                <div class="score-main">{score}</div>
                <div class="score-status" style="color:{color};">{status}</div>
                <div class="score-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )