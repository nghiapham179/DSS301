"""
pages/analysis.py — Phân tích từng drone
==========================================
Chọn drone → xem KPIs + biểu đồ riêng cho thiết bị đó.
"""

import plotly.express as px
import streamlit as st

from core import COLORS, RISK_COLOR_MAP, SEQ, startup_load_or_stop, style_chart
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_metric_card,
)


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    render_page_title(
        "Drone Unit Analysis",
        "Theo dõi sức khỏe và lịch sử bảo trì theo từng thiết bị.  ·  README Section 2.4",
    )

    render_section_label("Chọn thiết bị")
    drone_sel = st.selectbox(
        "Drone", sorted(df["drone_id"].unique()), label_visibility="collapsed"
    )
    df_d = df[df["drone_id"] == drone_sel]

    # KPIs
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
