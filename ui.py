import streamlit as st
import plotly.graph_objects as go
import numpy as np


# =====================================================
# DRONE DSS — PREMIUM LIGHT UI COMPONENTS
# Style: Modern Purple/White Dashboard with Sparklines
# =====================================================

def load_css():
    """Tải cấu trúc CSS cốt lõi, định nghĩa font chữ, màu sắc và ép màu text."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

            :root {
                --bg-main: #f4f5f9; 
                --sidebar-gradient: linear-gradient(145deg, #6b5dd3 0%, #5649b8 100%);
                --card-bg: #ffffff;
                --text-main: #11142d; /* Chữ chính màu xám đen đậm */
                --text-muted: #808191; /* Chữ phụ màu xám nhạt */
                --accent-purple: #6b5dd3;
            }

            /* 1. ÉP MÀU NỀN VÀ MÀU CHỮ TỔNG THỂ */
            html, body, [data-testid="stAppViewContainer"], [class*="css"] {
                font-family: 'Nunito Sans', sans-serif !important;
                background-color: var(--bg-main) !important;
                color: var(--text-main) !important;
            }

            /* 2. ÉP MÀU CHO CÁC TIÊU ĐỀ (Tránh bị màu trắng tàng hình) */
            h1, h2, h3, h4, h5, h6, 
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4,
            [data-testid="stMarkdownContainer"] p {
                color: var(--text-main) !important;
            }

            /* 3. LÀM RÕ CÁC HỘP (BOX/CARD) CỦA THÔNG SỐ */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: var(--card-bg) !important;
                border: 1px solid rgba(0, 0, 0, 0.08) !important; /* Thêm viền nhạt để hiện rõ Box */
                border-radius: 20px !important;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05) !important; /* Bóng đổ rõ hơn */
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease !important;
                padding: 10px !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                transform: translateY(-4px);
                box-shadow: 0px 15px 30px rgba(107, 93, 211, 0.15) !important; /* Đổ bóng tím khi hover */
            }

            /* 4. CHỈNH MÀU CHỮ BÊN TRONG METRIC CARD */
            div[data-testid="stMetricLabel"] * {
                color: var(--text-muted) !important; /* Tên chỉ số màu xám */
                font-size: 0.95rem !important;
                font-weight: 700 !important;
            }
            div[data-testid="stMetricValue"] {
                font-family: 'JetBrains Mono', monospace !important;
                color: var(--text-main) !important; /* Con số màu đen đậm */
                font-size: 2.2rem !important;
                font-weight: 800 !important;
            }
            div[data-testid="stMetricDelta"] * { 
                font-weight: 700 !important; 
            }

            /* Ghi chú dưới các card */
            .drone-note {
                display: block;
                color: var(--text-muted) !important;
                font-size: 0.85rem !important;
                margin-top: 10px !important;
                line-height: 1.5 !important;
            }

            /* 5. GIỮ CHO SIDEBAR (MÀU TÍM) LUÔN CÓ CHỮ TRẮNG */
            section[data-testid="stSidebar"] {
                background: var(--sidebar-gradient) !important;
                border-right: none !important;
            }
            section[data-testid="stSidebar"] * {
                color: #ffffff !important; 
            }

            /* =========================================
               SIDEBAR MENU NAVIGATION (STYLE DẠNG NÚT)
               ========================================= */
            /* Ẩn chữ "Chuyển trang" */
            section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
                display: none !important;
            }

            /* Định dạng Menu dạng Nút (Button) */
            div[role="radiogroup"] > label {
                background: transparent !important;
                border-radius: 12px !important;
                padding: 12px 15px !important;
                margin-bottom: 8px !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }

            /* Ẩn dấu chấm tròn mặc định */
            div[role="radiogroup"] > label span[data-baseweb="radio"] {
                display: none !important;
            }

            /* Chữ Menu chưa chọn */
            div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
                color: rgba(255, 255, 255, 0.7) !important;
                font-weight: 600 !important;
                font-size: 1.05rem !important;
                margin: 0 !important;
            }

            /* Hover vào Menu */
            div[role="radiogroup"] > label:hover {
                background: rgba(255, 255, 255, 0.1) !important;
            }
            div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {
                color: #ffffff !important;
            }

            /* Menu đang chọn (Active) */
            div[role="radiogroup"] > label[data-checked="true"] {
                background: #ffffff !important;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1) !important;
            }
            div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
                color: var(--accent-purple) !important;
                font-weight: 800 !important;
            }

            /* 6. SỬA LỖI NÚT UPGRADE BỊ TRẮNG TOÁT */
            .pro-upgrade-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 24px 20px;
                text-align: center;
                margin-top: 50px;
            }
            .pro-upgrade-card h4, .pro-upgrade-card p { 
                color: #ffffff !important; 
            }

            /* Ép cứng màu chữ tím cho nút */
            section[data-testid="stSidebar"] a.pro-upgrade-btn {
                background-color: #ffffff !important;
                color: #6b5dd3 !important; 
                padding: 12px 20px;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 800;
                font-size: 0.95rem;
                display: inline-block;
                width: 100%;
                box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            section[data-testid="stSidebar"] a.pro-upgrade-btn:hover {
                transform: scale(1.03);
            }

            /* Hide Default Elements */
            header[data-testid="stHeader"] { background: transparent !important; }
            footer { visibility: hidden !important; }
            .js-plotly-plot .plotly .modebar { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# CHART GENERATORS (PLOTLY)
# =====================================================

def create_sparkline(data: list, color: str) -> go.Figure:
    """Tạo biểu đồ đường cong mini cực mượt, không trục, không nền."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        line=dict(color=color, width=3, shape='spline'),
        hoverinfo='skip'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=5),
        height=60,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False)
    )
    return fig


# =====================================================
# RENDER FUNCTIONS (MAIN CONTENT)
# =====================================================

def render_top_nav():
    """Render thanh công cụ tìm kiếm và thông tin cá nhân ở trên cùng."""
    col_title, col_search, col_profile = st.columns([5, 3, 1], vertical_alignment="center")

    with col_title:
        st.markdown("<h2 style='margin-bottom: 0px; font-weight: 800;'>Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-muted); font-size: 0.95rem;'>Warranty & Operations Intelligence</p>",
                    unsafe_allow_html=True)

    with col_search:
        st.text_input("Search", placeholder="🔍 Search drones, records...", label_visibility="collapsed")

    with col_profile:
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 15px;">
                <span style="font-size: 1.3rem; cursor: pointer;">🔔</span>
                <div style="width: 42px; height: 42px; border-radius: 50%; background-color: #e6e4f7; display: flex; justify-content: center; align-items: center; font-weight: 800; color: #6b5dd3; box-shadow: 0px 4px 10px rgba(107, 93, 211, 0.2);">AD</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)


def render_page_title(title: str, subtitle: str = ""):
    st.markdown(f"<h3 style='font-weight: 800; margin-bottom: 5px;'>{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: var(--text-muted); font-size: 0.95rem;'>{subtitle}</p>", unsafe_allow_html=True)


def render_section_label(text: str):
    st.markdown(f"<h4 style='margin-top: 25px; margin-bottom: 15px; font-weight: 800;'>{text}</h4>",
                unsafe_allow_html=True)


def render_metric_card(label: str, value: str, note: str = "", delta: str = None):
    """Thẻ thông số cơ bản không có biểu đồ."""
    with st.container(border=True):
        st.metric(label=label, value=value, delta=delta)
        if note:
            st.markdown(f"<span class='drone-note'>{note}</span>", unsafe_allow_html=True)


def render_metric_with_chart(label: str, value: str, data_points: list, color: str = "#6b5dd3", delta: str = None):
    """Thẻ thông số cao cấp tích hợp biểu đồ xu hướng (Sparkline)."""
    with st.container(border=True):
        st.metric(label=label, value=value, delta=delta)
        if data_points is not None and len(data_points) > 0:
            fig = create_sparkline(data_points, color)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_risk_score(score: float, risk_level: str, note: str = ""):
    """Thẻ hiển thị điểm rủi ro với thanh tiến trình tuỳ chỉnh màu sắc tự động."""
    risk_level = str(risk_level).title()

    if risk_level == "High":
        color, delta_color = "#ef4444", "inverse"
    elif risk_level == "Medium":
        color, delta_color = "#f59e0b", "off"
    elif risk_level == "Low":
        color, delta_color = "#22c55e", "normal"
    else:
        color, delta_color = "#6b5dd3", "normal"

    progress_value = max(0.0, min(1.0, float(score) / 10))
    progress_pct = int(progress_value * 100)

    with st.container(border=True):
        st.metric(
            label="Overall Risk Score",
            value=f"{score:.1f} / 10",
            delta=f"Level: {risk_level}",
            delta_color=delta_color
        )

        st.markdown(
            f"""
            <div style="width: 100%; background-color: #f0f0f5; border-radius: 8px; margin-top: 15px; margin-bottom: 5px; height: 6px;">
                <div style="width: {progress_pct}%; background-color: {color}; height: 100%; border-radius: 8px; transition: width 0.5s ease-in-out;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if note:
            st.markdown(f"<span class='drone-note'>{note}</span>", unsafe_allow_html=True)


def render_result_badge(label: str, value: str, level: str = "info"):
    """Bảng thông báo trạng thái nổi bật."""
    message = f"**{label}:** {value}"
    badge_map = {
        "danger": st.error,
        "warning": st.warning,
        "success": st.success,
        "info": st.info
    }
    badge_func = badge_map.get(level, st.info)
    badge_func(message, icon="✓" if level == "success" else "ℹ️")


def render_banner(text: str, level: str = "info"):
    render_result_badge("", text, level)


# =====================================================
# RENDER FUNCTIONS (SIDEBAR)
# =====================================================

def render_sidebar_header():
    """Hiển thị Logo không bị nền trắng và đẹp hơn."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 35px; padding-left: 5px;">
            <span style="font-size: 2.2rem; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.2)); line-height: 1;">🚁</span>
            <h2 style="margin: 0; color: white !important; font-weight: 800; font-size: 1.6rem; letter-spacing: 0.5px;">Drone DSS</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_upgrade_card():
    """Hiển thị thẻ kêu gọi nâng cấp (Become a Pro) với hiệu ứng kính mờ."""
    st.markdown(
        """
        <div class="pro-upgrade-card">
            <h4>Become a Pro</h4>
            <p>Get access to advanced analytics and features</p>
            <a href="#" class="pro-upgrade-btn">Upgrade Now</a>
        </div>
        """,
        unsafe_allow_html=True
    )