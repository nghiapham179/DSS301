"""
views/dashboard.py — System Overview
======================================
KPIs + 4 biểu đồ phân tích fleet (Đã nâng cấp chuẩn DSS).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    COLORS, RISK_COLOR_MAP, SEQ,
    startup_load_or_stop, style_chart,
)
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_hero_panel, render_kpi_tiles
)


# ═══════════════════════════════════════════════════════════════════════
# CSS scoped — ép TẤT CẢ text trong khu vực drill-down về màu đen
# Dùng selector mạnh + !important để override global Streamlit styles
# ═══════════════════════════════════════════════════════════════════════
_DRILLDOWN_CSS = """
<style>
/* ── Catch-all: mọi text trong drill-down về đen ─────────────── */
.dss-drill,
.dss-drill *,
.dss-drill p,
.dss-drill span,
.dss-drill label,
.dss-drill div {
    color: #1c1e2e !important;
}

/* ── Radio button labels — selector mạnh ─────────────────────── */
.dss-drill [data-testid="stRadio"] label,
.dss-drill [data-testid="stRadio"] label p,
.dss-drill [data-testid="stRadio"] label div,
.dss-drill [data-testid="stRadio"] label span,
.dss-drill [data-testid="stRadio"] [data-baseweb="radio"] div,
.dss-drill div[role="radiogroup"] label,
.dss-drill div[role="radiogroup"] label p,
.dss-drill div[role="radiogroup"] label span {
    color: #1c1e2e !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* ── Radio label kích thước đọc được ─────────────────────────── */
.dss-drill [data-testid="stRadio"] label p {
    font-size: 0.88rem !important;
}

/* ── Tiêu đề widget "Khoảng thời gian" / "Chế độ xem" ────────── */
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

/* ── Khoanh nhóm radio bằng nền nhạt ─────────────────────────── */
.dss-drill [data-testid="stRadio"] > div {
    background: #f7f8fc !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
}

/* ── Selected radio: chữ vẫn đen, nhưng đậm hơn ──────────────── */
.dss-drill [data-testid="stRadio"] label:has(input:checked) p,
.dss-drill [data-testid="stRadio"] label:has(input:checked) {
    color: #1c1e2e !important;
    font-weight: 700 !important;
}

/* ── Mini KPI labels ─────────────────────────────────────────── */
.dss-mini-kpi [data-testid="stMetricLabel"],
.dss-mini-kpi [data-testid="stMetricLabel"] *,
.dss-mini-kpi [data-testid="stMetricLabel"] p {
    color: #1c1e2e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* ── Mini KPI values ────────────────────────────────────────── */
.dss-mini-kpi [data-testid="stMetricValue"],
.dss-mini-kpi [data-testid="stMetricValue"] * {
    color: #1c1e2e !important;
    font-size: 1.45rem !important;
}

/* ── Caption dưới chart ──────────────────────────────────────── */
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
        "System Overview",
        "Tổng quan dữ liệu vận hành, phân phối rủi ro và hành động bảo trì.  ·  DSS301",
    )

    # ── 1. TỔNG QUAN DỰ ÁN & CẢNH BÁO HỌC THUẬT ─────────────────────────────
    with st.expander("📖 TỔNG QUAN DỰ ÁN & CƠ SỞ CHỌN LỌC DỮ LIỆU", expanded=False):
        st.warning(
            "💡 **Lưu ý học thuật về Độ chính xác mô hình (Accuracy >99%)**\n\n"
            "Các mô hình phân loại (Random Forest) trong dự án này đạt độ chính xác rất cao (~99.5%). Nguyên nhân cốt lõi là do tập dữ liệu (dataset) hiện tại được tổng hợp (Synthetic Data) dựa trên các quy tắc vật lý và nghiệp vụ (Rule-based) có tính hệ thống cao, ranh giới phân loại rất rõ ràng và ít dữ liệu nhiễu (noise).\n\n"
            "→ **Kết luận:** Con số 99% này phản ánh việc mô hình đã **khai phá và biểu diễn lại xuất sắc bộ luật nghiệp vụ (Rules Extraction)**, thay vì là độ chính xác tuyệt đối trên tập dữ liệu vận hành nhiễu loạn ngoài thực tế. Nếu áp dụng vào luồng dữ liệu thật, accuracy kỳ vọng sẽ dao động ở mức 85-92%."
        )
        st.markdown(
            """
            ### 1. Tổng quan dự án Drone DSS
            **Drone DSS** là Hệ thống Hỗ trợ Ra quyết định ứng dụng Machine Learning (Random Forest) nhằm giám sát hạm đội UAV theo thời gian thực. Hệ thống tự động phân tích dữ liệu cảm biến để đánh giá rủi ro và chỉ định bảo trì.

            ### 2. Cơ sở khoa học & Ý nghĩa phân tích của 10 Features
            Việc chọn lọc 10 thông số đầu vào được phân tích dựa trên ma trận tương quan (gần như độc lập tuyến tính hoàn toàn), bao quát 3 "trụ cột" an toàn bay:
            * **🔋 Năng lượng & Động lực học (Battery Level, Flight Time, Speed, Altitude):** Đại diện cho sức khỏe vật lý và tải trọng vận hành.
            * **🌤️ Tác động môi trường (Wind Speed, Temperature, Humidity, Pressure):** Các biến xúc tác ngoại cảnh tác động lên khí động học và tuổi thọ linh kiện.
            * **📡 Viễn thông & Điều hướng (Signal Strength, GPS Accuracy):** Các biến sinh tử (Override Variables). Mất tín hiệu đồng nghĩa với trạng thái High Risk tức thời.
            """
        )

    # ── Aggregates ──────────────────────────────────────────
    avg_score = float(df["risk_score"].mean())
    high_pct  = round(df["is_high_risk"].mean() * 100, 1)

    vc       = df["operation_risk"].value_counts(normalize=True) * 100
    low_pct  = float(vc.get("Low", 0.0))
    med_pct  = float(vc.get("Medium", 0.0))
    high_dpc = float(vc.get("High", 0.0))

    # ── Hero panel ──────────────────────────────────────────
    score_counts = df["risk_score"].round().value_counts().to_dict()
    counts = [score_counts.get(i, 0) for i in range(11)]
    mx = max(counts) if max(counts) > 0 else 1
    bar_heights = [(c / mx) * 100 for c in counts]

    render_hero_panel(
        total_records=len(df),
        avg_score=avg_score,
        low_pct=low_pct,
        med_pct=med_pct,
        high_pct=high_dpc,
        bar_heights=bar_heights,
        counts=counts,
        delta_str="Real-time Sync",
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

    # ── 2. BẢNG DRONE ĐỘNG & LOGIC RISK SCORE CHUẨN ─────────────────────────
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    render_section_label("Quản lý Hạm đội & Theo dõi chi tiết (Drill-down)")

    latest_records = df.drop_duplicates(subset=['drone_id'], keep='last')

    drone_stats = df.groupby('drone_id').agg(
        total_flights=('drone_id', 'count'),
        high_risk_count=('is_high_risk', 'sum'),
        mode_risk=('operation_risk', lambda x: x.mode()[0])
    ).reset_index()

    drone_stats['high_risk_pct'] = (drone_stats['high_risk_count'] / drone_stats['total_flights']) * 100

    merged_df = pd.merge(
        drone_stats,
        latest_records[['drone_id', 'risk_score', 'maintenance_action', 'battery_level']],
        on='drone_id'
    )

    def determine_status(row):
        if row['maintenance_action'] == 'Maintenance required':
            return '🔴 Cần bảo trì (Ghi đè)'
        elif row['risk_score'] >= 6:
            return '🔴 Cần bảo trì'
        elif row['risk_score'] >= 3:
            return '🟡 Cần theo dõi'
        else:
            return '🟢 Online'

    merged_df['Trạng thái hiện tại'] = merged_df.apply(determine_status, axis=1)

    display_df = merged_df[['drone_id', 'total_flights', 'mode_risk', 'high_risk_pct', 'battery_level', 'Trạng thái hiện tại']].copy()
    display_df.columns = ['Drone ID', 'Tổng số chuyến', 'Risk phổ biến', '% High Risk', 'Pin hiện tại (%)', 'Trạng thái']
    display_df['% High Risk'] = display_df['% High Risk'].round(1).astype(str) + '%'
    display_df['Pin hiện tại (%)'] = display_df['Pin hiện tại (%)'].round(1).astype(str) + '%'

    st.caption("🖱️ *Click chọn một dòng bất kỳ để xem lịch sử bay chi tiết của Drone đó.*")

    # ── 3. TÍNH NĂNG DRILL-DOWN (Click vào bảng) ────────────────────────────
    selected_event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if len(selected_event.selection.rows) > 0:
        selected_idx  = selected_event.selection.rows[0]
        selected_drone = display_df.iloc[selected_idx]['Drone ID']
        selected_status = display_df.iloc[selected_idx]['Trạng thái']

        # Wrap toàn bộ phần drill-down trong div.dss-drill để CSS scoped áp dụng
        st.markdown('<div class="dss-drill">', unsafe_allow_html=True)

        with st.container(border=True):
            # ── Header với drone ID + status badge ──────────────────────
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
                            {selected_drone}
                        </span>
                    </div>
                    <div style="font-size:0.85rem; font-weight:600; color:#1c1e2e;">
                        {selected_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            drone_history = df[df['drone_id'] == selected_drone].reset_index(drop=True)
            n_total = len(drone_history)

            # ── Chọn cửa sổ + chế độ hiển thị ───────────────────────────
            cfg1, cfg2 = st.columns([1.4, 1])
            with cfg1:
                window_label = st.radio(
                    "Khoảng thời gian",
                    ["100 chuyến gần nhất", "500 chuyến gần nhất",
                     "2,000 chuyến gần nhất", "Toàn bộ lịch sử"],
                    horizontal=True,
                    key=f"win_{selected_drone}",
                )
            with cfg2:
                view_mode = st.radio(
                    "Chế độ xem",
                    ["Trung bình động", "Điểm thô"],
                    horizontal=True,
                    key=f"mode_{selected_drone}",
                )

            window_map = {
                "100 chuyến gần nhất":   100,
                "500 chuyến gần nhất":   500,
                "2,000 chuyến gần nhất": 2000,
                "Toàn bộ lịch sử":       n_total,
            }
            n_show  = min(window_map[window_label], n_total)
            df_view = drone_history.tail(n_show).reset_index(drop=True)

            # ── Vẽ chart ────────────────────────────────────────────────
            fig_history = go.Figure()

            if view_mode == "Trung bình động":
                roll_w = max(5, min(200, n_show // 20))
                df_view["rolling_mean"] = df_view["risk_score"].rolling(roll_w, min_periods=1).mean()
                df_view["rolling_min"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).min()
                df_view["rolling_max"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).max()

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
                fig_history.add_trace(go.Scatter(
                    x=df_view.index, y=df_view["rolling_mean"],
                    mode="lines",
                    line=dict(color="#4f63d2", width=2.8, shape="spline", smoothing=1.0),
                    name=f"Risk Score TB",
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

            # ── Ngưỡng tham chiếu: hrect mờ + line ────────────────────────
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
                    title=dict(
                        text="Risk Score",
                        font=dict(size=11, color="#1c1e2e"),
                    ),
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

            # ── Mini KPIs cho drone được chọn ───────────────────────────
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="dss-mini-kpi">', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tổng chuyến",     f"{n_total:,}")
            k2.metric("Risk score TB",   f"{drone_history['risk_score'].mean():.2f} / 10")
            k3.metric("% High Risk",     f"{drone_history['is_high_risk'].mean()*100:.1f}%")
            k4.metric("Pin TB",          f"{drone_history['battery_level'].mean():.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # đóng .dss-drill

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
        drone_avg = df.groupby("drone_id")["risk_score"].mean().reset_index()
        drone_avg = drone_avg.sort_values("risk_score", ascending=False).head(12)

        fig4 = px.bar(
            drone_avg, x="drone_id", y="risk_score",
            color="risk_score",
            color_continuous_scale=["#16a34a", "#d97706", "#dc2626"],
            text_auto=".1f"
        )
        fig4.update_traces(marker_line_width=0, textposition="outside")
        fig4.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Risk Score")
        st.plotly_chart(style_chart(fig4, 360), use_container_width=True)
        st.caption(
            "🚁 Danh sách thu gọn tập trung vào nhóm thiết bị có điểm rủi ro vượt ngưỡng. "
            "Đây là danh sách cần được đưa vào kế hoạch kiểm tra kỹ thuật lập tức."
        )