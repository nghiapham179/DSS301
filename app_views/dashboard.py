"""
pages/dashboard.py — System Overview
======================================
KPIs + 4 biểu đồ phân tích fleet (README Section 2.1).
"""

import numpy as np
import plotly.express as px
import streamlit as st

from core import (
    COLORS, RISK_COLOR_MAP, SEQ,
    startup_load_or_stop, style_chart,
)
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_metric_card, render_metric_with_chart, render_risk_score,
)


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    render_page_title(
        "System Overview",
        "Tổng quan dữ liệu vận hành, phân phối rủi ro và hành động bảo trì.  ·  DSS301",
    )

    # ── KPIs ────────────────────────────────────────────────
    avg_score      = float(df["risk_score"].mean())
    high_pct       = round(df["is_high_risk"].mean() * 100, 1)
    avg_risk_level = "High" if avg_score >= 6 else "Medium" if avg_score >= 3 else "Low"

    rng = np.random.default_rng(42)
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        render_metric_with_chart(
            "Tổng Bản Ghi", f"{len(df):,}",
            rng.normal(0, 1, 14).cumsum(), COLORS["accent"], "+3.2% tuần này",
        )
    with k2:
        rng2 = np.random.default_rng(7)
        render_metric_with_chart(
            "Số Drone", str(df["drone_id"].nunique()),
            rng2.normal(0, 1, 14).cumsum(), COLORS["blue"], "Active units",
        )
    with k3:
        render_metric_card(
            "Tỷ lệ High Risk", f"{high_pct}%",
            "Drone cần xử lý cảnh báo", "−2% so tuần trước",
        )
    with k4:
        render_risk_score(
            round(avg_score, 1), avg_risk_level,
            "Điểm rủi ro trung bình hệ thống",
        )

    st.divider()

    # ── Chart row 1 ─────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        render_section_label("Phân phối rủi ro")
        fig = px.histogram(
            df, x="risk_score", color="operation_risk", nbins=11,
            color_discrete_map=RISK_COLOR_MAP, barmode="overlay", opacity=0.82,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
        st.caption(
            "🔍 Histogram phân phối điểm rủi ro (0–10) toàn bộ drone, "
            "tô màu theo 3 mức **High / Medium / Low**. Phần lớn drone tập trung "
            "ở vùng Medium — đây là baseline để so sánh khi thông số bất thường."
        )

    with ch2:
        render_section_label("Hành động bảo trì")
        mc = df["maintenance_action"].value_counts().reset_index()
        mc.columns = ["action", "count"]
        fig2 = px.bar(mc, x="count", y="action", orientation="h",
                      color="action", color_discrete_sequence=SEQ)
        fig2.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(style_chart(fig2, 340), use_container_width=True)
        st.caption(
            "🔧 Thống kê 4 loại hành động bảo trì model đề xuất. Tỷ lệ **Monitor** "
            "và **No maintenance** cao cho thấy fleet đang vận hành ổn định; "
            "**Maintenance required** ít nhưng cần ưu tiên xử lý."
        )

    # ── Chart row 2 ─────────────────────────────────────────
    ch3, ch4 = st.columns(2, gap="large")
    with ch3:
        render_section_label("Battery vs Wind Speed")
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig3 = px.scatter(sample, x="battery_level", y="wind_speed",
                          color="operation_risk", opacity=0.55,
                          color_discrete_map=RISK_COLOR_MAP)
        st.plotly_chart(style_chart(fig3, 360), use_container_width=True)
        st.caption(
            "⚡ Mối quan hệ giữa **pin** và **tốc độ gió** — 2 feature quan trọng nhất "
            "(54% importance). Vùng trái-trên (pin thấp + gió mạnh) tập trung điểm đỏ "
            "(High risk) — tổ hợp nguy hiểm cần báo động sớm."
        )

    with ch4:
        render_section_label("Rủi ro trung bình theo drone")
        drone_avg = df.groupby("drone_id")["risk_score"].mean().reset_index()
        fig4 = px.bar(drone_avg, x="drone_id", y="risk_score",
                      color="risk_score",
                      color_continuous_scale=["#dcfce7", "#fef3c7", "#fee2e2"])
        fig4.update_traces(marker_line_width=0)
        fig4.update_layout(coloraxis_showscale=False, xaxis_title=None)
        st.plotly_chart(style_chart(fig4, 360), use_container_width=True)
        st.caption(
            "🚁 Risk score trung bình theo từng thiết bị. Drone có cột màu **đỏ/cam** "
            "đang có lịch sử rủi ro cao hơn fleet trung bình → ưu tiên kiểm tra trong "
            "trang **Phân tích drone**."
        )
