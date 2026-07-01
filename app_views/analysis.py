"""
pages/analysis.py — Phân tích từng drone
==========================================
Chọn drone → xem KPIs + Lịch sử bay (drill-down) + biểu đồ riêng cho thiết bị đó.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import COLORS, RISK_COLOR_MAP, SEQ, startup_load_or_stop, style_chart
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_metric_card,
)


# ═══════════════════════════════════════════════════════════════════════
# CSS scoped riêng cho khu vực Drill-down lịch sử bay
# Ép text về đen + radio buttons rõ ràng
# ═══════════════════════════════════════════════════════════════════════
_DRILLDOWN_CSS = """
<style>
/* Catch-all: text trong drill-down về đen */
.dss-drill,
.dss-drill *,
.dss-drill p,
.dss-drill span,
.dss-drill label,
.dss-drill div {
    color: #1c1e2e !important;
}

/* Radio button labels — selector mạnh */
.dss-drill [data-testid="stRadio"] label,
.dss-drill [data-testid="stRadio"] label p,
.dss-drill [data-testid="stRadio"] label div,
.dss-drill [data-testid="stRadio"] label span,
.dss-drill div[role="radiogroup"] label,
.dss-drill div[role="radiogroup"] label p {
    color: #1c1e2e !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}
.dss-drill [data-testid="stRadio"] label p {
    font-size: 0.88rem !important;
}

/* Widget labels */
.dss-drill [data-testid="stWidgetLabel"],
.dss-drill [data-testid="stWidgetLabel"] *,
.dss-drill [data-testid="stWidgetLabel"] p {
    color: #1c1e2e !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    opacity: 1 !important;
}

/* Radio nhóm */
.dss-drill [data-testid="stRadio"] > div {
    background: #f7f8fc !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
}
.dss-drill [data-testid="stRadio"] label:has(input:checked) p {
    color: #1c1e2e !important;
    font-weight: 700 !important;
}

/* Caption dưới chart */
.dss-drill .chart-caption {
    color: #1c1e2e !important;
    font-size: 0.78rem !important;
    font-style: italic !important;
    margin: 8px 0 0 !important;
}
</style>
"""


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    # Inject CSS scoped cho drill-down
    st.markdown(_DRILLDOWN_CSS, unsafe_allow_html=True)

    render_page_title(
        "Drone Unit Analysis",
        "Theo dõi sức khỏe, lịch sử bay và bảo trì theo từng thiết bị.  ·  README Section 2.4",
    )

    render_section_label("Chọn thiết bị")
    drone_sel = st.selectbox(
        "Drone", sorted(df["drone_id"].unique()), label_visibility="collapsed"
    )
    df_d = df[df["drone_id"] == drone_sel].reset_index(drop=True)

    # ── Xác định trạng thái drone (giống logic ở dashboard cũ) ──────────
    latest_row     = df_d.iloc[-1]
    latest_risk    = float(latest_row["risk_score"])
    latest_maint   = str(latest_row.get("maintenance_action", ""))

    if latest_maint == "Maintenance required":
        status_label = "🔴 Cần bảo trì (Ghi đè)"
    elif latest_risk >= 6:
        status_label = "🔴 Cần bảo trì"
    elif latest_risk >= 3:
        status_label = "🟡 Cần theo dõi"
    else:
        status_label = "🟢 Online"

    # ── KPIs ────────────────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4, gap="large")
    with d1:
        render_metric_card("Tổng bản ghi",    f"{len(df_d):,}",                         drone_sel)
    with d2:
        render_metric_card("Risk Score TB",   f"{df_d['risk_score'].mean():.1f}",        "Rủi ro trung bình")
    with d3:
        render_metric_card("Tỷ lệ High Risk", f"{df_d['is_high_risk'].mean()*100:.1f}%", "Tỷ lệ nguy hiểm")
    with d4:
        render_metric_card("Battery TB",      f"{df_d['battery_level'].mean():.0f}%",    "Dung lượng pin TB")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # ── LỊCH SỬ BAY (chuyển từ dashboard.py về đây) ────────────────────
    # ═══════════════════════════════════════════════════════════════════
    render_section_label("Lịch sử bay & biến động rủi ro")

    st.markdown('<div class="dss-drill">', unsafe_allow_html=True)

    with st.container(border=True):
        # ── Header với drone ID + status badge ──────────────────────────
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:center; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.3rem;">🔎</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#1c1e2e;">
                        Lịch sử bay
                    </span>
                    <span style="background:#eef0fb; color:#4f63d2;
                                 font-family:'JetBrains Mono',monospace;
                                 font-size:0.85rem; font-weight:700;
                                 padding:3px 12px; border-radius:7px;">
                        {drone_sel}
                    </span>
                </div>
                <div style="font-size:0.85rem; font-weight:600; color:#1c1e2e;">
                    {status_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        n_total = len(df_d)

        # ── Chọn cửa sổ + chế độ hiển thị ───────────────────────────────
        cfg1, cfg2 = st.columns([1.4, 1])
        with cfg1:
            window_label = st.radio(
                "Khoảng thời gian",
                ["100 chuyến gần nhất", "500 chuyến gần nhất",
                 "2,000 chuyến gần nhất", "Toàn bộ lịch sử"],
                horizontal=True,
                key=f"analysis_win_{drone_sel}",
            )
        with cfg2:
            view_mode = st.radio(
                "Chế độ xem",
                ["Trung bình động", "Điểm thô"],
                horizontal=True,
                key=f"analysis_mode_{drone_sel}",
            )

        window_map = {
            "100 chuyến gần nhất":   100,
            "500 chuyến gần nhất":   500,
            "2,000 chuyến gần nhất": 2000,
            "Toàn bộ lịch sử":       n_total,
        }
        n_show  = min(window_map[window_label], n_total)
        df_view = df_d.tail(n_show).reset_index(drop=True)

        # ── Vẽ chart ────────────────────────────────────────────────────
        fig_history = go.Figure()

        if view_mode == "Trung bình động":
            roll_w = max(5, min(200, n_show // 20))
            df_view["rolling_mean"] = df_view["risk_score"].rolling(roll_w, min_periods=1).mean()
            df_view["rolling_min"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).min()
            df_view["rolling_max"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).max()

            # Min-Max band
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_max"],
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_min"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(79,99,210,0.10)",
                name=f"Khoảng dao động ({roll_w} chuyến)",
                hoverinfo="skip",
            ))
            # Trung bình
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_mean"],
                mode="lines",
                line=dict(color="#4f63d2", width=2.8, shape="spline", smoothing=1.0),
                name="Risk Score TB",
                hovertemplate="Chuyến %{x}<br>Risk TB: %{y:.2f}<extra></extra>",
            ))
            chart_caption = (
                f"Đường xu hướng: trung bình động {roll_w} chuyến · "
                f"Vùng nhạt thể hiện khoảng min-max."
            )

        else:  # Điểm thô — downsample nếu quá nhiều
            MAX_POINTS = 1500
            if len(df_view) > MAX_POINTS:
                step    = len(df_view) // MAX_POINTS
                df_plot = df_view.iloc[::step].copy()
                note    = f" (lấy mẫu 1/{step})"
            else:
                df_plot = df_view
                note    = ""

            colors = df_plot["risk_score"].apply(
                lambda v: "#dc2626" if v >= 6 else "#d97706" if v >= 3 else "#16a34a"
            )

            fig_history.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["risk_score"],
                mode="markers",
                marker=dict(
                    color=colors, size=5, opacity=0.6,
                    line=dict(width=0),
                ),
                name="Risk Score",
                hovertemplate="Chuyến %{x}<br>Risk: %{y}<extra></extra>",
            ))
            chart_caption = (
                f"Hiển thị {len(df_plot):,}/{len(df_view):,} điểm thô{note} · "
                "Màu đỏ ≥ 6 · vàng 3-5 · xanh < 3."
            )

        # ── Ngưỡng tham chiếu ───────────────────────────────────────────
        fig_history.add_hrect(
            y0=6, y1=10.5,
            fillcolor="rgba(220,38,38,0.06)",
            line_width=0, layer="below",
        )
        fig_history.add_hrect(
            y0=3, y1=6,
            fillcolor="rgba(217,119,6,0.045)",
            line_width=0, layer="below",
        )
        fig_history.add_hline(
            y=6, line_dash="dash", line_color="#dc2626", line_width=1.2,
            annotation_text="<b>≥ 6</b>  Ngưỡng nguy hiểm",
            annotation_position="top left",
            annotation_font=dict(color="#dc2626", size=11),
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )
        fig_history.add_hline(
            y=3, line_dash="dash", line_color="#d97706", line_width=1.2,
            annotation_text="<b>≥ 3</b>  Ngưỡng cảnh báo",
            annotation_position="top left",
            annotation_font=dict(color="#d97706", size=11),
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )

        fig_history.update_layout(
            xaxis=dict(
                title=dict(
                    text=f"Timeline — {n_show:,} chuyến gần nhất",
                    font=dict(size=11, color="#1c1e2e"),
                ),
                tickfont=dict(size=10, color="#1c1e2e"),
                showgrid=False,
                showline=True, linecolor="rgba(0,0,0,0.08)",
            ),
            yaxis=dict(
                title=dict(text="Risk Score", font=dict(size=11, color="#1c1e2e")),
                range=[-0.5, 10.5], dtick=2,
                tickfont=dict(size=10, color="#1c1e2e"),
                gridcolor="rgba(0,0,0,0.04)",
                zeroline=False,
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white", font_size=12,
                bordercolor="rgba(0,0,0,0.1)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                font=dict(size=11, color="#1c1e2e"),
                bgcolor="rgba(255,255,255,0.6)",
                borderwidth=0,
            ),
            margin=dict(t=20, b=50, l=60, r=40),
        )

        st.plotly_chart(style_chart(fig_history, 360), use_container_width=True)

        st.markdown(
            f'<p class="chart-caption">{chart_caption}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # đóng .dss-drill

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # ── 3 chart phân tích sâu (giữ nguyên từ bản gốc) ──────────────────
    # ═══════════════════════════════════════════════════════════════════
    a1, a2 = st.columns(2, gap="large")
    with a1:
        render_section_label(f"Phân phối Risk Score — {drone_sel}")
        fig = px.histogram(df_d, x="risk_score", nbins=11,
                           color_discrete_sequence=[COLORS["accent"]], opacity=0.85)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
        st.caption(
            "📊 Phân bổ điểm rủi ro của riêng drone này. So với baseline fleet, "
            "phân phối lệch phải nhiều = drone thường xuyên ở trạng thái rủi ro cao."
        )

    with a2:
        render_section_label(f"Flight Time vs Battery — {drone_sel}")
        sample = df_d.sample(min(1000, len(df_d)), random_state=42)
        fig2 = px.scatter(sample, x="flight_time", y="battery_level",
                          color="operation_risk", opacity=0.65,
                          color_discrete_map=RISK_COLOR_MAP)
        st.plotly_chart(style_chart(fig2, 340), use_container_width=True)
        st.caption(
            "🔋 Mối tương quan giữa thời gian bay và pin còn lại. Đường giảm dốc "
            "= pin tiêu hao nhanh, có thể do battery cell xuống cấp."
        )

    render_section_label("Phân bổ hành động bảo trì")
    mc = df_d["maintenance_action"].value_counts().reset_index()
    mc.columns = ["action", "count"]
    fig3 = px.pie(mc, names="action", values="count", hole=0.55,
                  color_discrete_sequence=SEQ)
    fig3.update_traces(textfont_size=12)
    st.plotly_chart(style_chart(fig3, 380), use_container_width=True)
    st.caption(
        "🥯 Donut chart cho thấy tỉ lệ các hành động bảo trì cho drone này. "
        "Mảng **Monitor / No maintenance** lớn = drone đang khoẻ; "
        "mảng **Maintenance required** lớn = drone cần được kiểm tra ngay."
    )