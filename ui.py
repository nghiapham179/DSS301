import streamlit as st

def load_css():
    st.markdown("""
    <style>
        /* =========================
           1. CÀI ĐẶT CHUNG & FONT CHỮ
        ========================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700;800&display=swap');

        :root {
            --bg-base: #09090b;       
            --card-bg: #18181b;       
            --border: #27272a;        
            --text-main: #f8fafc;     
            --text-muted: #a1a1aa;    
            --accent-glow: rgba(56, 189, 248, 0.15); 
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-base) !important;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(255, 255, 255, 0.02), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.03), transparent 25%) !important;
        }

        header[data-testid="stHeader"] { background: transparent !important; }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
            animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* =========================
           2. THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR) - ÉP BUỘC WIDTH 100%
        ========================== */
        
        section[data-testid="stSidebar"] {
            background-color: #09090b !important;
            border-right: 1px solid #27272a !important;
        }

        .stRadio > label { display: none !important; }

        /* Khung chứa các nút */
        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 12px !important;
            width: 100% !important;
            align-items: stretch !important; /* ÉP CÁC NÚT CON PHẢI KÉO DÃN HẾT CỠ */
        }

        /* Nút bấm */
        div[role="radiogroup"] label {
            width: 100% !important; /* ÉP RỘNG 100% LẦN 2 */
            max-width: 100% !important;
            box-sizing: border-box !important;
            background-color: #18181b !important;
            border: 1px solid #27272a !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
        }

        div[role="radiogroup"] label:hover {
            background-color: #27272a !important;
            border-color: #3f3f46 !important;
            transform: translateX(4px);
        }

        div[role="radiogroup"] label span[data-baseweb="radio"] div:first-child { display: none !important; }

        div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            color: #a1a1aa !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            margin: 0 !important;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background-color: rgba(14, 165, 233, 0.1) !important;
            border-color: #0ea5e9 !important;
            box-shadow: inset 4px 0 0 #0ea5e9, 0 0 15px rgba(14, 165, 233, 0.1) !important;
            transform: translateX(4px);
        }

        div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }

       /* =========================
           3. SỬA LỖI CẮT CHỮ TRONG THẺ METRIC - CẤP ĐỘ CAO NHẤT
        ========================== */
        
        div[data-testid="stMetric"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            transition: all 0.2s ease !important;
            height: 100% !important;
            min-height: 145px !important; /* <--- ÉP CHIỀU CAO ĐỀU NHAU */
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        div[data-testid="stMetric"]:hover {
            border-color: #3f3f46 !important;
            background: #1f1f22 !important;
            transform: translateY(-2px);
        }

        /* Tác động trực tiếp vào mọi thành phần chứa chữ bên trong Value */
        div[data-testid="stMetricValue"], 
        div[data-testid="stMetricValue"] > div,
        div[data-testid="stMetricValue"] * {
            white-space: normal !important;       /* Bắt buộc xuống dòng */
            word-wrap: break-word !important;     
            word-break: break-word !important;
            overflow: visible !important;         
            text-overflow: clip !important;       /* Xóa dấu ba chấm */
            font-size: 22px !important;           /* Chữ nhỏ lại để nằm vừa 2 dòng */
            line-height: 1.3 !important;          
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-main) !important;
        }

        div[data-testid="stMetricValue"] {
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }

        div[data-testid="stMetricLabel"] * {
            color: var(--text-muted) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            white-space: normal !important; 
        }
        /* =========================
           4. GIAO DIỆN COMPONENT KHÁC
        ========================== */
        
        .top-nav {
            background: rgba(24, 24, 27, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 100px;
            padding: 12px 24px;
            margin-bottom: 36px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand-wrap { display: flex; align-items: center; gap: 16px; }
        
        .brand-icon {
            width: 44px; height: 44px; border-radius: 50%;
            background: #0ea5e9; color: #fff;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }

        .brand { font-size: 18px; font-weight: 700; color: var(--text-main); }
        .brand-small { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
        .nav-actions { display: flex; align-items: center; gap: 12px; }
        
        .nav-badge {
            background: #27272a; color: #e4e4e7;
            border: 1px solid #3f3f46; border-radius: 999px;
            padding: 4px 12px; font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-dot {
            width: 8px; height: 8px; background: #10b981;
            border-radius: 50%; box-shadow: 0 0 8px #10b981;
        }

        .main-title {
            font-size: 32px; font-weight: 800; color: var(--text-main);
            letter-spacing: -1px; margin-bottom: 8px;
        }
        .sub-title { font-size: 15px; color: var(--text-muted); margin-bottom: 36px; }
        .section-title { font-size: 18px; font-weight: 600; color: var(--text-main); margin-bottom: 6px; }
        .section-caption { font-size: 13px; color: var(--text-muted); margin-bottom: 20px; }

        .metric-card {
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: 16px; padding: 24px; display: flex; flex-direction: column;
            justify-content: center; transition: all 0.2s ease; position: relative; overflow: hidden;
        }
        .metric-card:hover { border-color: #3f3f46; background: #1f1f22; transform: translateY(-2px); }
        .metric-label { font-size: 13px; font-weight: 500; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; }
        .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 38px; font-weight: 800; color: var(--text-main); line-height: 1; }
        .metric-note { font-size: 12px; color: #52525b; margin-top: 10px; font-family: 'JetBrains Mono', monospace; }

        .score-box {
            background: var(--card-bg); border: 1px solid var(--border); border-radius: 20px;
            padding: 32px; display: flex; align-items: center; position: relative; overflow: hidden;
        }
        .score-box::before {
            content: ""; position: absolute; top: -50px; left: -50px; width: 150px; height: 150px;
            background: var(--accent-glow); filter: blur(50px); border-radius: 50%;
        }
        .score-main { font-family: 'JetBrains Mono', monospace; font-size: 64px; font-weight: 800; color: var(--text-main); line-height: 1; position: relative; z-index: 1; }
        .score-status { display: inline-block; margin-top: 16px; padding: 6px 14px; border-radius: 6px; font-size: 14px; font-weight: 700; text-transform: uppercase; background: #09090b; position: relative; z-index: 1; }
        .score-note { margin-top: 16px; color: var(--text-muted); font-size: 14px; line-height: 1.5; position: relative; z-index: 1; }

        div[data-testid="stPlotlyChart"] {
            background: var(--card-bg) !important; border: 1px solid var(--border) !important;
            border-radius: 16px !important; padding: 12px !important; transition: all 0.2s ease;
        }
        div[data-testid="stPlotlyChart"]:hover { border-color: #3f3f46 !important; }
        
        .stSlider div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
            background-color: #0ea5e9 !important; color: #fff !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_top_nav():
    st.markdown("""
    <div class="top-nav">
        <div class="brand-wrap">
            <div class="brand-icon">🚁</div>
            <div>
                <div class="brand">Drone DSS </div>
                <div class="brand-small">Hệ Thống Ra Quyết Định Chạy Bằng AI</div>
            </div>
        </div>
        <div class="nav-actions">
            <div class="status-dot"></div>
            <div class="nav-badge">Hệ Thống Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_main_title(title, subtitle):
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)

def open_card():
    st.write("")

def close_card():
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
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

def render_score_box(score, status, note, color):
    st.markdown(
        f"""
        <div class="score-box">
            <div>
                <div class="score-main">{score}</div>
                <div class="score-status" style="color:{color}; border: 1px solid {color};">{status}</div>
                <div class="score-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )