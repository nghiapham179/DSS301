"""
views/dashboard.py — System Overview
======================================
KPIs + 4 biểu đồ phân tích fleet (Đã nâng cấp chuẩn DSS).
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
    render_hero_panel, render_kpi_tiles, render_drone_records_table,
)


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    render_page_title(
        "System Overview",
        "Tổng quan dữ liệu vận hành, phân phối rủi ro và hành động bảo trì.  ·  DSS301",
    )

    # ── Aggregates ──────────────────────────────────────────
    avg_score = float(df["risk_score"].mean())
    high_pct  = round(df["is_high_risk"].mean() * 100, 1)

    # Distribution by operation risk level (sums ≈ 100)
    vc       = df["operation_risk"].value_counts(normalize=True) * 100
    low_pct  = float(vc.get("Low", 0.0))
    med_pct  = float(vc.get("Medium", 0.0))
    high_dpc = float(vc.get("High", 0.0))

    # Risk-score histogram → bar heights (0–100)
    counts, _   = np.histogram(df["risk_score"].dropna(), bins=24, range=(0, 10))
    mx          = counts.max() or 1
    bar_heights = [c / mx * 100 for c in counts]

    # ── Hero panel ──────────────────────────────────────────
    render_hero_panel(
        total_records=len(df), avg_score=avg_score,
        low_pct=low_pct, med_pct=med_pct, high_pct=high_dpc,
        bar_heights=bar_heights, delta_str="Real-time Sync",
    )

    # ── KPI tiles ───────────────────────────────────────────
    render_kpi_tiles([
        {"icon": "🔋", "label": "Pin trung bình",
         "value": f"{df['battery_level'].mean():.0f}%",
         "delta": "↗ Ổn định", "delta_color": "var(--c-success)"},
        {"icon": "📶", "label": "Tín hiệu TB",
         "value": f"{df['signal_strength'].mean():.0f}%",
         "delta": "↗ Ổn định", "delta_color": "var(--c-success)"},
        {"icon": "🚁", "label": "Drone hoạt động",
         "value": f"{df['drone_id'].nunique()}",
         "delta": "Toàn đội bay", "delta_color": "var(--text-secondary)"},
        {"icon": "⚠️", "label": "Tỷ lệ High Risk",
         "value": f"{high_pct}%",
         "delta": "Cần ưu tiên xử lý", "delta_color": "var(--c-danger)"},
    ])

    # ── Records table (top drones by avg risk) ──────────────
    grp = (df.groupby("drone_id")
             .agg(score=("risk_score", "mean"),
                  bat=("battery_level", "mean"),
                  wind=("wind_speed", "mean"))
             .reset_index()
             .sort_values("score", ascending=False)
             .head(6))
    rows = []
    for _, g in grp.iterrows():
        s = g["score"]
        if s >= 6:
            lvl, lab = "danger", "Nguy hiểm"
        elif s >= 3:
            lvl, lab = "warning", "Cảnh báo"
        else:
            lvl, lab = "success", "An toàn"
        rows.append({
            "id": str(g["drone_id"]),
            "battery": f"{g['bat']:.0f}%",
            "wind": f"{g['wind']:.0f} m/s",
            "status_label": lab, "status_level": lvl,
        })
    render_drone_records_table(rows, title="Danh sách Drone cần ưu tiên kiểm tra")

    st.divider()

    # ── Chart row 1 ─────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        render_section_label("Phân phối rủi ro toàn hệ thống")
        fig = px.histogram(
            df, x="risk_score", color="operation_risk", nbins=15,
            color_discrete_map=RISK_COLOR_MAP, barmode="stack", opacity=0.9,
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_title="Điểm rủi ro (0-10)", yaxis_title="Số lượng bản ghi")
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
        st.caption(
            "🔍 Histogram phân phối điểm rủi ro (0–10) theo dạng cộng dồn (Stack). "
            "Giúp nhận diện nhanh tỷ trọng các vùng High Risk tập trung ở phân khúc điểm nào."
        )

    with ch2:
        render_section_label("Thống kê hành động bảo trì")
        mc = df["maintenance_action"].value_counts().reset_index()
        mc.columns = ["action", "count"]
        # Sắp xếp để thanh dài nhất nằm trên cùng
        mc = mc.sort_values("count", ascending=True)

        fig2 = px.bar(
            mc, x="count", y="action", orientation="h",
            color="action", color_discrete_sequence=SEQ, text="count"
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Số lượng")
        st.plotly_chart(style_chart(fig2, 340), use_container_width=True)
        st.caption(
            "🔧 Hệ thống tự động phân loại mức độ bảo trì. Tỷ lệ **Maintenance required** "
            "sẽ quyết định tải lượng công việc của đội ngũ kỹ thuật mặt đất trong tuần."
        )

    # ── Chart row 2 ─────────────────────────────────────────
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    ch3, ch4 = st.columns(2, gap="large")

    with ch3:
        render_section_label("Tương quan: Pin vs Tốc độ gió")
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig3 = px.scatter(
            sample, x="battery_level", y="wind_speed",
            color="operation_risk", size="risk_score", opacity=0.6,
            color_discrete_map=RISK_COLOR_MAP,
            hover_data=["drone_id", "flight_time"]
        )
        fig3.update_layout(xaxis_title="Dung lượng pin (%)", yaxis_title="Tốc độ gió (m/s)")
        st.plotly_chart(style_chart(fig3, 360), use_container_width=True)
        st.caption(
            "⚡ **Size của chấm tròn thể hiện mức độ rủi ro**. Vùng góc trái phía trên "
            "(Pin thấp + Gió to) là khu vực cảnh báo đỏ. Hover chuột để xem ID thiết bị."
        )

    with ch4:
        render_section_label("Top 12 Drone rủi ro cao nhất")
        # Lọc ra top 12 drone có điểm rủi ro trung bình cao nhất
        drone_avg = df.groupby("drone_id")["risk_score"].mean().reset_index()
        drone_avg = drone_avg.sort_values("risk_score", ascending=False).head(12)

        fig4 = px.bar(
            drone_avg, x="drone_id", y="risk_score",
            color="risk_score",
            color_continuous_scale=["#fef3c7", "#fca5a5", "#dc2626"],
            text_auto=".1f"
        )
        fig4.update_traces(marker_line_width=0, textposition="outside")
        fig4.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Risk Score")
        st.plotly_chart(style_chart(fig4, 360), use_container_width=True)
        st.caption(
            "🚁 Danh sách thu gọn tập trung vào nhóm thiết bị có điểm rủi ro vượt ngưỡng. "
            "Đây là danh sách cần được đưa vào kế hoạch kiểm tra kỹ thuật lập tức."
        )