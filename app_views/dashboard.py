"""
views/dashboard.py — System Overview
======================================
KPIs + Fleet status table + 4 biểu đồ phân tích fleet.
(Phần drill-down lịch sử bay đã chuyển sang views/analysis.py)
"""

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
# Chart layout preset — dùng chung cho mọi chart trong page
# Fix vấn đề hover tooltip đè legend + tooltip position lệch
# ═══════════════════════════════════════════════════════════════════════
def _apply_chart_style(fig, unified=True):
    """
    Apply consistent styling for hover + legend.

    unified=True → hovermode "x unified": tooltip là 1 box duy nhất bám
    theo trục X, text LUÔN nằm trong box (fix triệt để bug tách rời).
    unified=False → dùng cho scatter (mode "closest").
    """
    fig.update_layout(
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#d1d5db",
            font=dict(
                size=13,
                family="'Inter', system-ui, sans-serif",
                color="#1c1e2e",
            ),
            align="left",
            namelength=-1,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#1c1e2e"),
            title=dict(font=dict(size=11, color="#64697a")),
        ),
        margin=dict(t=20, b=100, l=50, r=20),
    )
    # Tắt spike line (thanh nét đứt) mà không ghi đè title của axis
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    render_page_title(
        "System Overview",
        "Tổng quan dữ liệu vận hành, phân phối rủi ro và hành động bảo trì.  ·  DSS301",
    )

    # ── 1. TỔNG QUAN DỰ ÁN & CẢNH BÁO HỌC THUẬT ─────────────────────────────
    with st.expander("📖 TỔNG QUAN DỰ ÁN & CƠ SỞ CHỌN LỌC DỮ LIỆU", expanded=False):
        st.warning(
            "💡 **Lưu ý học thuật về Độ chính xác mô hình (Accuracy >99%)**\n\n"
            "Nhãn của tập dữ liệu này là **synthetic** — được sinh tất định từ chính "
            "các features bằng bộ luật nghiệp vụ (rule-based). Vì vậy accuracy ~99% trên "
            "random split là **hệ quả tất yếu**: mô hình đã khai phá lại bộ luật sinh nhãn "
            "(rules extraction), không phải năng lực dự đoán trên dữ liệu vận hành thật "
            "(hiện tượng *label leakage* — Kapoor & Narayanan 2023, *Patterns*).\n\n"
            "→ **Cách xử lý (thay vì hạ accuracy nhân tạo):** giữ nhãn sạch và bổ sung "
            "3 đánh giá trung thực — xem tab **Model Info → 🧪 Nghiên cứu**:\n"
            "1. **GroupKFold theo drone_id** — đo khả năng tổng quát sang drone chưa từng thấy (leakage gap);\n"
            "2. **Noise robustness** — tiêm nhiễu nhãn phụ thuộc đặc trưng vào *tập train duy nhất* "
            "(0→30%), test giữ sạch, so độ bền RF / DT / LR (Frénay & Verleysen 2014);\n"
            "3. **Cost-sensitive decision** — quy tắc Bayes tối thiểu chi phí kỳ vọng (Elkan 2001) "
            "ưu tiên không bỏ sót High-risk, đang được áp dụng thật trong trang Dự đoán."
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

    # ── 2. BẢNG DRONE FLEET ─────────────────────────────────────────────────
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    render_section_label("Quản lý Hạm đội — Trạng thái hiện tại")

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

    display_df = merged_df[['drone_id', 'total_flights', 'mode_risk', 'high_risk_pct',
                             'battery_level', 'Trạng thái hiện tại']].copy()
    display_df.columns = ['Drone ID', 'Tổng số chuyến', 'Risk phổ biến',
                           '% High Risk', 'Pin hiện tại (%)', 'Trạng thái']
    display_df['% High Risk'] = display_df['% High Risk'].round(1).astype(str) + '%'
    display_df['Pin hiện tại (%)'] = display_df['Pin hiện tại (%)'].round(1).astype(str) + '%'

    st.caption("💡 *Xem lịch sử bay chi tiết của từng drone tại trang Phân tích Drone.*")

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ── Chart row 1 ─────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        render_section_label("Phân phối rủi ro toàn hệ thống")
        fig = px.histogram(
            df, x="risk_score", color="operation_risk", nbins=15,
            color_discrete_map=RISK_COLOR_MAP, barmode="stack", opacity=0.9,
        )
        fig.update_traces(
            marker_line_width=0,
            # unified mode: %{x} hiện ở tiêu đề box, mỗi nhóm 1 dòng
            hovertemplate="<b>%{fullData.name}:</b> %{y:,} bản ghi<extra></extra>",
        )
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Điểm rủi ro (0-10)",
                    standoff=10,  # khoảng cách title tới trục
                ),
            ),
            yaxis_title="Số lượng bản ghi",
            bargap=0.08,
            legend_title_text="Operation Risk",
        )
        _apply_chart_style(fig)
        st.plotly_chart(
            style_chart(fig, 440),  # tăng chiều cao để có chỗ cho legend
            width="stretch",
            config={"displayModeBar": False},
        )
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
        fig2.update_traces(
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>%{x:,} bản ghi<extra></extra>",
        )
        fig2.update_layout(
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Số lượng",
        )
        _apply_chart_style(fig2, unified=False)  # bar ngang: dùng closest
        fig2.update_layout(hovermode="y unified")  # tooltip bám trục Y
        # Chart này không có legend nên margin bottom nhỏ hơn
        fig2.update_layout(margin=dict(t=20, b=30, l=50, r=40))
        st.plotly_chart(
            style_chart(fig2, 380),
            width="stretch",
            config={"displayModeBar": False},
        )
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
        # size= không chấp nhận NaN — loại phòng thủ các dòng thiếu risk_score
        sample = sample.dropna(subset=["risk_score", "battery_level", "wind_speed"])
        fig3 = px.scatter(
            sample, x="battery_level", y="wind_speed",
            color="operation_risk", size="risk_score", opacity=0.6,
            color_discrete_map=RISK_COLOR_MAP,
            hover_data={
                "drone_id": True,
                "flight_time": ":.1f",
                "battery_level": ":.0f",
                "wind_speed": ":.1f",
                "risk_score": ":.1f",
                "operation_risk": False,
            },
        )
        fig3.update_traces(
            marker=dict(line=dict(width=0)),
        )
        fig3.update_layout(
            xaxis=dict(
                title=dict(text="Dung lượng pin (%)", standoff=10),
            ),
            yaxis_title="Tốc độ gió (m/s)",
            legend_title_text="Operation Risk",
        )
        _apply_chart_style(fig3, unified=False)  # scatter: dùng closest
        st.plotly_chart(
            style_chart(fig3, 440),  # tăng cao vì có legend
            width="stretch",
            config={"displayModeBar": False},
        )
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
        fig4.update_traces(
            marker_line_width=0,
            textposition="outside",
            hovertemplate="<b>Risk Score TB:</b> %{y:.2f} / 10<extra></extra>",
        )
        fig4.update_layout(
            coloraxis_showscale=False,
            xaxis_title=None,
            yaxis_title="Risk Score",
        )
        _apply_chart_style(fig4)
        # Chart này không có legend, chỉ có color axis đã ẩn
        fig4.update_layout(margin=dict(t=20, b=30, l=50, r=20))
        st.plotly_chart(
            style_chart(fig4, 400),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "🚁 Danh sách thu gọn tập trung vào nhóm thiết bị có điểm rủi ro vượt ngưỡng. "
            "Đây là danh sách cần được đưa vào kế hoạch kiểm tra kỹ thuật lập tức."
        )