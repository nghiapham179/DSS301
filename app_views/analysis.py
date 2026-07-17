"""
pages/analysis.py — Phan tich tung drone
==========================================
Chon drone -> Panel thong tin DJI + co so khoa hoc chon model
             + KPIs + Lich su bay (drill-down) + bieu do rieng.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import COLORS, RISK_COLOR_MAP, SEQ, startup_load_or_stop, style_chart
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_metric_card,
)


# ══════════════════════════════════════════════════════════════════════════════
# THONG SO KY THUAT 3 DONG DJI PHO BIEN NHAT VIET NAM
# Nguon: DJI official specs (dji.com)
# ══════════════════════════════════════════════════════════════════════════════
DRONE_SPECS = {
    "DJI Mini 3": {
        "tier":        "Phổ thông",
        "tier_color":  "#3ad17e",
        "icon":        "🛩️",
        "weight":      "248 g",
        "flight_time": "38 phút",
        "wind_res":    "10.7 m/s (cấp 5)",
        "camera":      "1/1.3″ CMOS · 12MP · 4K HDR",
        "sensor":      "Cảm biến xuống dưới",
        "transmission":"10 km (O2)",
        "use_case":    "Người mới, du lịch, vlog cá nhân",
        "why": (
            "Đại diện phân khúc drone tiêu dùng nhẹ (< 249g — miễn đăng ký ở "
            "nhiều nước). Trọng lượng thấp khiến nó nhạy cảm nhất với gió và "
            "nhiễu tín hiệu — là 'trường hợp xấu nhất' lý tưởng để kiểm định "
            "khả năng dự báo rủi ro của mô hình."
        ),
    },
    "DJI Air 3": {
        "tier":        "Tầm trung",
        "tier_color":  "#f0b429",
        "icon":        "🚁",
        "weight":      "720 g",
        "flight_time": "46 phút",
        "wind_res":    "12 m/s (cấp 6)",
        "camera":      "Kép 1/1.3″ CMOS · 48MP · 24mm + 70mm",
        "sensor":      "Cảm biến đa hướng (omnidirectional)",
        "transmission":"20 km (O4)",
        "use_case":    "Bán chuyên, khảo sát, quay phim tầm trung",
        "why": (
            "Cân bằng giữa trọng tải và thời lượng bay. Cảm biến đa hướng cho "
            "dữ liệu môi trường phong phú hơn — phù hợp kiểm tra mô hình trên "
            "phân khúc phổ biến nhất thị trường VN, nơi khối lượng vận hành lớn."
        ),
    },
    "DJI Mavic 3 Pro": {
        "tier":        "Cao cấp",
        "tier_color":  "#f0584f",
        "icon":        "🛸",
        "weight":      "958 g",
        "flight_time": "43 phút",
        "wind_res":    "12 m/s (cấp 6)",
        "camera":      "Bộ ba Hasselblad · 4/3 CMOS · 20MP",
        "sensor":      "Cảm biến đa hướng + hồng ngoại đáy",
        "transmission":"15 km (O3+)",
        "use_case":    "Chuyên nghiệp, điện ảnh, khảo sát công nghiệp",
        "why": (
            "Đại diện thiết bị nặng, nhiều cảm biến, giá trị cao — nơi quyết "
            "định bảo trì sai sẽ tốn kém nhất. Khả năng kháng gió tốt cho phép "
            "kiểm định mô hình ở vùng vận hành khắc nghiệt mà Mini 3 không đạt được."
        ),
    },
}

# Anh xa drone_id -> ten model DJI (khop train_model.py)
DRONE_NAME_MAP = {
    "Drone_1":  "DJI Mini 3",  "Drone_4":  "DJI Mini 3",
    "Drone_7":  "DJI Mini 3",  "Drone_10": "DJI Mini 3",
    "Drone_2":  "DJI Air 3",   "Drone_5":  "DJI Air 3",  "Drone_8": "DJI Air 3",
    "Drone_3":  "DJI Mavic 3 Pro", "Drone_6": "DJI Mavic 3 Pro",
    "Drone_9":  "DJI Mavic 3 Pro",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"


# ══════════════════════════════════════════════════════════════════════════════
# CSS scoped cho drill-down (ep text den)
# ══════════════════════════════════════════════════════════════════════════════
_DRILLDOWN_CSS = """
<style>
.dss-drill, .dss-drill *, .dss-drill p, .dss-drill span,
.dss-drill label, .dss-drill div { color: #1c1e2e !important; }
.dss-drill [data-testid="stRadio"] label,
.dss-drill [data-testid="stRadio"] label p,
.dss-drill div[role="radiogroup"] label,
.dss-drill div[role="radiogroup"] label p {
    color: #1c1e2e !important; font-weight: 500 !important; opacity: 1 !important;
}
.dss-drill [data-testid="stRadio"] label p { font-size: 0.88rem !important; }
.dss-drill [data-testid="stWidgetLabel"],
.dss-drill [data-testid="stWidgetLabel"] * {
    color: #1c1e2e !important; font-weight: 700 !important;
    font-size: 0.75rem !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; opacity: 1 !important;
}
.dss-drill [data-testid="stRadio"] > div {
    background: #f7f8fc !important; border-radius: 10px !important;
    padding: 10px 14px !important; border: 1px solid rgba(0,0,0,0.08) !important;
}
.dss-drill .chart-caption {
    color: #1c1e2e !important; font-size: 0.78rem !important;
    font-style: italic !important; margin: 8px 0 0 !important;
}
</style>
"""


def _load_per_drone_metrics():
    """Doc per_drone tu metrics JSON cua operation_risk_model (neu co)."""
    p = MODEL_DIR / "operation_risk_model_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("per_drone")
    except Exception:
        return None


def _render_drone_info_panel(model_name: str):
    """Panel thong tin ky thuat + co so khoa hoc cho 1 dong DJI."""
    spec = DRONE_SPECS.get(model_name)
    if not spec:
        return

    # Header card voi tier badge
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1b1e30,#262a44);
                    border-radius:16px;padding:20px 24px;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;align-items:center;gap:14px;">
              <span style="font-size:2.2rem;">{spec['icon']}</span>
              <div>
                <div style="color:#fff;font-size:1.3rem;font-weight:700;
                            letter-spacing:-0.02em;">{model_name}</div>
                <div style="color:rgba(255,255,255,.55);font-size:0.8rem;">
                    {spec['use_case']}</div>
              </div>
            </div>
            <span style="background:{spec['tier_color']}22;color:{spec['tier_color']};
                         border:1px solid {spec['tier_color']}55;border-radius:999px;
                         padding:5px 16px;font-size:0.8rem;font-weight:700;">
                {spec['tier']}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bang thong so ky thuat — 6 cot
    specs_grid = [
        ("⚖️", "Trọng lượng",   spec["weight"]),
        ("⏱️", "Thời gian bay",  spec["flight_time"]),
        ("💨", "Kháng gió",      spec["wind_res"]),
        ("📷", "Camera",         spec["camera"]),
        ("📡", "Truyền tín hiệu",spec["transmission"]),
        ("🎯", "Cảm biến",       spec["sensor"]),
    ]
    cells = ""
    for icon, label, val in specs_grid:
        cells += (
            f"<div style='background:#fff;border:1px solid rgba(0,0,0,.07);"
            f"border-radius:12px;padding:13px 16px;'>"
            f"<div style='font-size:0.68rem;color:#64697a;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;'>"
            f"{icon} {label}</div>"
            f"<div style='font-size:0.92rem;color:#1c1e2e;font-weight:700;'>{val}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);"
        f"gap:12px;margin-bottom:16px;'>{cells}</div>",
        unsafe_allow_html=True,
    )


def render():
    render_top_nav()
    _, df = startup_load_or_stop()

    st.markdown(_DRILLDOWN_CSS, unsafe_allow_html=True)

    render_page_title(
        "Drone Unit Analysis",
        "Thông tin thiết bị, cơ sở khoa học và lịch sử vận hành theo từng drone.  ·  DSS301",
    )

    # ── Chon thiet bi ────────────────────────────────────────────────────
    render_section_label("Chọn thiết bị")
    drone_sel = st.selectbox(
        "Drone", sorted(df["drone_id"].unique()), label_visibility="collapsed"
    )
    df_d = df[df["drone_id"] == drone_sel].reset_index(drop=True)
    model_name = DRONE_NAME_MAP.get(drone_sel, "N/A")

    # ── PANEL THONG TIN DRONE + CO SO KHOA HOC ──────────────────────────
    _render_drone_info_panel(model_name)

    # ── Trang thai drone ────────────────────────────────────────────────
    latest_row   = df_d.iloc[-1]
    latest_risk  = float(latest_row["risk_score"])
    latest_maint = str(latest_row.get("maintenance_action", ""))

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
        render_metric_card("Tổng bản ghi",    f"{len(df_d):,}",                          model_name)
    with d2:
        render_metric_card("Risk Score TB",   f"{df_d['risk_score'].mean():.1f}",        "Rủi ro trung bình")
    with d3:
        render_metric_card("Tỷ lệ High Risk", f"{df_d['is_high_risk'].mean()*100:.1f}%", "Tỷ lệ nguy hiểm")
    with d4:
        render_metric_card("Battery TB",      f"{df_d['battery_level'].mean():.0f}%",    "Dung lượng pin TB")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # ── LICH SU BAY (drill-down) ────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════
    render_section_label("Lịch sử bay & biến động rủi ro")
    st.markdown('<div class="dss-drill">', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:center; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.3rem;">🔎</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#1c1e2e;">
                        Lịch sử bay</span>
                    <span style="background:#eef0fb; color:#4f63d2;
                                 font-family:'JetBrains Mono',monospace;
                                 font-size:0.85rem; font-weight:700;
                                 padding:3px 12px; border-radius:7px;">
                        {drone_sel} · {model_name}</span>
                </div>
                <div style="font-size:0.85rem; font-weight:600; color:#1c1e2e;">
                    {status_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        n_total = len(df_d)
        cfg1, cfg2 = st.columns([1.4, 1])
        with cfg1:
            window_label = st.radio(
                "Khoảng thời gian",
                ["100 chuyến gần nhất", "500 chuyến gần nhất",
                 "2,000 chuyến gần nhất", "Toàn bộ lịch sử"],
                horizontal=True, key=f"analysis_win_{drone_sel}",
            )
        with cfg2:
            view_mode = st.radio(
                "Chế độ xem", ["Trung bình động", "Điểm thô"],
                horizontal=True, key=f"analysis_mode_{drone_sel}",
            )

        window_map = {
            "100 chuyến gần nhất": 100, "500 chuyến gần nhất": 500,
            "2,000 chuyến gần nhất": 2000, "Toàn bộ lịch sử": n_total,
        }
        n_show  = min(window_map[window_label], n_total)
        df_view = df_d.tail(n_show).reset_index(drop=True)

        fig_history = go.Figure()
        if view_mode == "Trung bình động":
            roll_w = max(5, min(200, n_show // 20))
            df_view["rolling_mean"] = df_view["risk_score"].rolling(roll_w, min_periods=1).mean()
            df_view["rolling_min"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).min()
            df_view["rolling_max"]  = df_view["risk_score"].rolling(roll_w, min_periods=1).max()
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_max"],
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_min"],
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(79,99,210,0.10)",
                name=f"Khoảng dao động ({roll_w} chuyến)", hoverinfo="skip"))
            fig_history.add_trace(go.Scatter(
                x=df_view.index, y=df_view["rolling_mean"], mode="lines",
                line=dict(color="#4f63d2", width=2.8, shape="spline", smoothing=1.0),
                name="Risk Score TB",
                hovertemplate="Chuyến %{x}<br>Risk TB: %{y:.2f}<extra></extra>"))
            chart_caption = (f"Đường xu hướng: trung bình động {roll_w} chuyến · "
                             f"Vùng nhạt thể hiện khoảng min-max.")
        else:
            MAX_POINTS = 1500
            if len(df_view) > MAX_POINTS:
                step    = len(df_view) // MAX_POINTS
                df_plot = df_view.iloc[::step].copy()
                note    = f" (lấy mẫu 1/{step})"
            else:
                df_plot = df_view
                note    = ""
            colors = df_plot["risk_score"].apply(
                lambda v: "#dc2626" if v >= 6 else "#d97706" if v >= 3 else "#16a34a")
            fig_history.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["risk_score"], mode="markers",
                marker=dict(color=colors, size=5, opacity=0.6, line=dict(width=0)),
                name="Risk Score",
                hovertemplate="Chuyến %{x}<br>Risk: %{y}<extra></extra>"))
            chart_caption = (f"Hiển thị {len(df_plot):,}/{len(df_view):,} điểm thô{note} · "
                             "Màu đỏ ≥ 6 · vàng 3-5 · xanh < 3.")

        fig_history.add_hrect(y0=6, y1=10.5, fillcolor="rgba(220,38,38,0.06)",
                              line_width=0, layer="below")
        fig_history.add_hrect(y0=3, y1=6, fillcolor="rgba(217,119,6,0.045)",
                              line_width=0, layer="below")
        fig_history.add_hline(y=6, line_dash="dash", line_color="#dc2626", line_width=1.2,
                              annotation_text="<b>≥ 6</b>  Ngưỡng nguy hiểm",
                              annotation_position="top left",
                              annotation_font=dict(color="#dc2626", size=11),
                              annotation_bgcolor="rgba(255,255,255,0.85)")
        fig_history.add_hline(y=3, line_dash="dash", line_color="#d97706", line_width=1.2,
                              annotation_text="<b>≥ 3</b>  Ngưỡng cảnh báo",
                              annotation_position="top left",
                              annotation_font=dict(color="#d97706", size=11),
                              annotation_bgcolor="rgba(255,255,255,0.85)")
        fig_history.update_layout(
            xaxis=dict(title=dict(text=f"Timeline — {n_show:,} chuyến gần nhất",
                                  font=dict(size=11, color="#1c1e2e")),
                       tickfont=dict(size=10, color="#1c1e2e"),
                       showgrid=False, showline=True, linecolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title=dict(text="Risk Score", font=dict(size=11, color="#1c1e2e")),
                       range=[-0.5, 10.5], dtick=2,
                       tickfont=dict(size=10, color="#1c1e2e"),
                       gridcolor="rgba(0,0,0,0.04)", zeroline=False),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="rgba(0,0,0,0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=11, color="#1c1e2e"),
                        bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
            margin=dict(t=20, b=50, l=60, r=40))
        fig_history.update_xaxes(showspikes=False)
        fig_history.update_yaxes(showspikes=False)

        st.plotly_chart(style_chart(fig_history, 360), width="stretch")
        st.markdown(f'<p class="chart-caption">{chart_caption}</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # ── CO SO KHOA HOC + PER-DRONE ACCURACY ─────────────────────────────
    # ═══════════════════════════════════════════════════════════════════
    render_section_label("Cơ sở khoa học chọn thiết bị & hiệu năng mô hình")

    col_why, col_acc = st.columns([1.2, 1], gap="large")

    with col_why:
        spec = DRONE_SPECS.get(model_name, {})
        st.markdown(
            f"""
            <div style="background:#f7f8fc;border:1px solid rgba(0,0,0,.07);
                        border-radius:14px;padding:18px 20px;height:100%;">
              <div style="font-size:0.72rem;font-weight:700;color:#4f63d2;
                          text-transform:uppercase;letter-spacing:0.06em;
                          margin-bottom:10px;">
                🔬 Vì sao chọn {model_name}?</div>
              <p style="font-size:0.9rem;color:#1c1e2e;line-height:1.7;margin:0;">
                {spec.get('why', '')}</p>
              <div style="margin-top:14px;padding-top:14px;
                          border-top:1px solid rgba(0,0,0,.07);
                          font-size:0.8rem;color:#64697a;line-height:1.6;">
                <b>Chiến lược nghiên cứu:</b> 3 dòng DJI được chọn để bao phủ
                <b>3 phân khúc thị trường</b> (phổ thông → cao cấp) với dải trọng
                lượng 248g–958g và kháng gió cấp 5–6. Cách này giúp kiểm chứng
                mô hình DSS hoạt động ổn định trên toàn phổ thiết bị thực tế
                tại Việt Nam, thay vì chỉ tối ưu cho một loại drone.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_acc:
        per_drone = _load_per_drone_metrics()
        if per_drone and drone_sel in per_drone:
            d = per_drone[drone_sel]
            st.markdown(
                f"""
                <div style="background:#fff;border:1px solid rgba(0,0,0,.07);
                            border-radius:14px;padding:18px 20px;height:100%;">
                  <div style="font-size:0.72rem;font-weight:700;color:#4f63d2;
                              text-transform:uppercase;letter-spacing:0.06em;
                              margin-bottom:14px;">
                    🎯 Độ chính xác mô hình trên {drone_sel}</div>
                  <div style="display:flex;gap:20px;margin-bottom:8px;">
                    <div>
                      <div style="font-size:2rem;font-weight:700;color:#16a34a;
                                  font-family:'JetBrains Mono',monospace;">
                        {d['accuracy']*100:.1f}%</div>
                      <div style="font-size:0.7rem;color:#64697a;">Accuracy</div>
                    </div>
                    <div>
                      <div style="font-size:2rem;font-weight:700;color:#4f63d2;
                                  font-family:'JetBrains Mono',monospace;">
                        {d['f1']*100:.1f}%</div>
                      <div style="font-size:0.7rem;color:#64697a;">F1-score</div>
                    </div>
                  </div>
                  <div style="font-size:0.78rem;color:#64697a;
                              padding-top:12px;border-top:1px solid rgba(0,0,0,.07);">
                    Đánh giá trên <b>{d['n_test']:,}</b> bản ghi test của riêng
                    drone này (operation_risk). Con số cao và đồng đều giữa các
                    drone chứng tỏ mô hình không thiên vị thiết bị nào.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Chưa có dữ liệu per-drone. Chạy lại `train_model.py` "
                "(bản mới) để sinh chỉ số hiệu năng theo từng drone.",
                icon="ℹ️",
            )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # ── 3 CHART PHAN TICH SAU ──────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════
    a1, a2 = st.columns(2, gap="large")
    with a1:
        render_section_label(f"Phân phối Risk Score — {drone_sel}")
        fig = px.histogram(df_d, x="risk_score", nbins=11,
                           color_discrete_sequence=[COLORS["accent"]], opacity=0.85)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_chart(fig, 340), width="stretch")
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
        st.plotly_chart(style_chart(fig2, 340), width="stretch")
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
    st.plotly_chart(style_chart(fig3, 380), width="stretch")
    st.caption(
        "🥯 Donut chart cho thấy tỉ lệ các hành động bảo trì cho drone này. "
        "Mảng **Monitor / No maintenance** lớn = drone đang khoẻ; "
        "mảng **Maintenance required** lớn = drone cần được kiểm tra ngay."
    )