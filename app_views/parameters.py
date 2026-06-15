"""
views/parameters.py — Điều chỉnh thông số
============================================
2 tab:
  • Tab Slider: mô phỏng nhanh, không lưu
  • Tab Form:   nhập dữ liệu thực địa + dropdown chọn tình huống mẫu,
                lưu vào custom_drone_data.csv kết hợp danh sách ID từ gốc.
"""

import pandas as pd
import streamlit as st

from core import (
    build_input_df, predict_drone, flight_decision, battery_status,
    translate, risk_score_estimate, risk_to_level, save_custom,
    make_radar_chart, startup_load_or_stop,
    TEMPLATES, TEMPLATE_LOOKUP, load_data, CUSTOM_DATA_PATH,
)
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_risk_score, render_result_badge, render_banner,
)


def get_all_drone_ids():
    """Quét và lấy danh sách toàn bộ Drone ID từ cả drone_data_clean.csv và custom_drone_data.csv."""
    ids = set()

    # 1. Quét dữ liệu từ dataset sạch gốc
    try:
        df_base = load_data()
        for col in ["Drone_ID", "drone_id"]:
            if col in df_base.columns:
                ids.update(df_base[col].dropna().astype(str).tolist())
    except Exception:
        pass

    # 2. Quét dữ liệu từ lịch sử custom đã lưu trước đó
    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            df_custom = pd.read_csv(CUSTOM_DATA_PATH)
            if "drone_id" in df_custom.columns:
                ids.update(df_custom["drone_id"].dropna().astype(str).tolist())
        except Exception:
            pass

    return sorted(list(ids))


def render():
    render_top_nav()
    startup_load_or_stop()

    render_page_title(
        "Điều chỉnh thông số drone",
        "Mô phỏng hoặc nhập dữ liệu thực địa — hệ thống hỗ trợ ra quyết định vận hành.",
    )

    tab_slider, tab_form = st.tabs([
        "🎛️  Mô phỏng nhanh (Slider)",
        "📝  Nhập dữ liệu thực địa (Form)",
    ])

    with tab_slider:
        _render_slider_tab()

    with tab_form:
        _render_form_tab()


# ─── TAB 1: SLIDER ──────────────────────────────────────────────────────────

def _render_slider_tab():
    render_banner(
        "Kéo slider để mô phỏng tình huống — kết quả cập nhật tức thì, không lưu CSV.",
        "info",
    )

    render_section_label("Thông số đầu vào  (10 features)")
    sl1, sl2 = st.columns(2, gap="large")

    with sl1:
        battery_level   = st.slider("🔋 Pin còn lại (%)",       0.0, 100.0,  60.0, 1.0)
        wind_speed      = st.slider("💨 Tốc độ gió (m/s)",       0.0,  50.0,  15.0, 0.5)
        signal_strength = st.slider("📶 Cường độ tín hiệu (%)", 0.0, 100.0,  70.0, 1.0)
        temperature     = st.slider("🌡️ Nhiệt độ (°C)",        -40.0, 50.0,  25.0, 0.5)
        gps_accuracy    = st.slider("📍 GPS accuracy (m)",       0.0,  10.0,   5.0, 0.1)

    with sl2:
        flight_time = st.slider("⏱️ Thời gian bay (phút)",   0.0,  60.0,  30.0, 0.5)
        altitude    = st.slider("🏔️ Độ cao (m)",             0.0, 500.0, 250.0, 5.0)
        speed       = st.slider("🚀 Tốc độ bay",              0.0, 100.0,  50.0, 1.0)
        humidity    = st.slider("💧 Độ ẩm (%)",                0.0, 100.0,  50.0, 1.0)
        pressure    = st.slider("🌬️ Áp suất (hPa)",         950.0,1050.0, 1000.0, 1.0)

    inp = build_input_df(
        battery_level=battery_level, flight_time=flight_time,
        signal_strength=signal_strength, temperature=temperature,
        wind_speed=wind_speed, gps_accuracy=gps_accuracy,
        altitude=altitude, speed=speed, humidity=humidity, pressure=pressure,
    )
    risk_pred, maint_pred, rec_pred, conf = predict_drone(inp)
    flight_st, flight_reason, flight_lv = flight_decision(
        battery_level, signal_strength, wind_speed, gps_accuracy,
        flight_time, temperature, altitude, speed, risk_pred, maint_pred,
    )
    risk_vn, maint_vn = translate(risk_pred, maint_pred)
    bat_label, bat_lv = battery_status(battery_level)
    est_score         = risk_score_estimate(battery_level, wind_speed, signal_strength, temperature)

    st.divider()
    render_section_label("Kết quả dự đoán")

    col_score, col_radar = st.columns([1, 1], gap="large")
    with col_score:
        render_risk_score(est_score, risk_pred, f"Model confidence: {conf}%")
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1: render_result_badge("Mức rủi ro",     risk_vn,   risk_to_level(risk_pred))
        with b2: render_result_badge("Tình trạng pin", bat_label, bat_lv)
        b3, b4 = st.columns(2)
        with b3: render_result_badge("Bảo trì",        maint_vn,  "info")
        with b4: render_result_badge("Trạng thái bay", flight_st, flight_lv)

    with col_radar:
        render_section_label("Radar — thông số chuẩn hoá 0–1")
        params = {
            "Pin":      battery_level / 100,
            "Tín hiệu": signal_strength / 100,
            "GPS":       1 - gps_accuracy / 10,
            "Gió":       1 - wind_speed / 50,
            "Bay":       1 - flight_time / 60,
            "Độ cao":   1 - altitude / 500,
            "Tốc độ":   1 - speed / 100,
            "Nhiệt độ": 1 - abs(temperature - 20) / 65,
        }
        st.plotly_chart(
            make_radar_chart(params, safe_threshold=0.6),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    render_banner(flight_reason, flight_lv)

    st.divider()
    render_section_label("Khuyến nghị cuối cùng")
    render_banner(rec_pred, risk_to_level(risk_pred))


# ─── TAB 2: FORM (with template dropdown) ───────────────────────────────────

NONE_OPTION = "— Tự nhập thủ công —"


def _render_form_tab():
    render_banner(
        "Nhập dữ liệu thực địa chi tiết — kết quả sẽ được **lưu vào** "
        "`Data/custom_drone_data.csv` để theo dõi lịch sử.",
        "info",
    )

    # ── Dropdown chọn tình huống mẫu (OUTSIDE form so it reruns immediately) ──
    render_section_label("Áp dụng tình huống mẫu (tuỳ chọn)")

    template_choice = st.selectbox(
        "Tình huống mẫu",
        [NONE_OPTION] + list(TEMPLATE_LOOKUP.keys()),
        key="template_selector",
        label_visibility="collapsed",
        help="Chọn 1 tình huống → các ô bên dưới tự động điền theo phiếu mẫu. "
             "Bạn vẫn có thể chỉnh tay sau khi áp dụng.",
        )

    # If template selected, get defaults; otherwise use system defaults
    if template_choice != NONE_OPTION:
        t = TEMPLATE_LOOKUP[template_choice]
        defaults = t["params"]
        st.caption(f"💡 **{t['icon']} {t['name']}** — {t['desc']}")
    else:
        defaults = None

    def _d(key, fallback):
        return float(defaults[key]) if defaults and key in defaults else fallback

    # ── Dropdown chọn Drone ID (OUTSIDE form so it reruns immediately) ──
    render_section_label("Thông tin thiết bị vận hành")
    existing_ids = get_all_drone_ids()
    dropdown_options = ["➕ -- Tạo Drone Mới --"] + existing_ids

    selected_drone_opt = st.selectbox(
        "Chọn Drone ID từ Dataset hoặc khởi tạo thực thể mới",
        options=dropdown_options,
        key="drone_id_selector",
        label_visibility="collapsed",
        help="Hệ thống tự động đồng bộ danh sách ID từ cả drone_data_clean.csv và custom_drone_data.csv"
    )

    if selected_drone_opt == "➕ -- Tạo Drone Mới --":
        drone_id = st.text_input(
            "Nhập ID Drone mới", value="Drone_Custom_001",
            help="Mã định danh drone mới, ví dụ: Drone_003, UAV_Beta",
        )
    else:
        drone_id = selected_drone_opt
        st.info(f"📍 Đang thiết lập thêm bản ghi thông số thực địa mới cho thiết bị: **{drone_id}**")

    # ── Form (auto-fills from dropdown selection above) ───────────────────────
    with st.form("drone_manual_form"):
        st.markdown(f"**Drone Target:** `{drone_id}`")

        render_section_label("Thông số cảm biến (10 features)")
        f1, f2 = st.columns(2, gap="large")
        with f1:
            bl = st.number_input("🔋 Pin còn lại (%)",        0.0, 100.0, _d("battery_level", 60.0), 1.0)
            ft = st.number_input("⏱️ Thời gian bay (phút)",    0.0, 300.0, _d("flight_time", 30.0), 1.0)
            ss = st.number_input("📶 Cường độ tín hiệu (%)",  0.0, 100.0, _d("signal_strength", 70.0), 1.0)
            tp = st.number_input("🌡️ Nhiệt độ (°C)",         -50.0,  80.0, _d("temperature", 25.0), 0.5)
            ws = st.number_input("💨 Tốc độ gió (m/s)",        0.0, 100.0, _d("wind_speed", 15.0), 0.5)
        with f2:
            ga = st.number_input("📍 GPS accuracy (m)",    0.0,  50.0, _d("gps_accuracy", 5.0), 0.1)
            al = st.number_input("🏔️ Độ cao (m)",          0.0,1000.0, _d("altitude", 250.0), 5.0)
            sp = st.number_input("🚀 Tốc độ bay",           0.0, 200.0, _d("speed", 50.0), 1.0)
            hm = st.number_input("💧 Độ ẩm (%)",            0.0, 100.0, _d("humidity", 50.0), 1.0)
            pr = st.number_input("🌬️ Áp suất (hPa)",     900.0,1100.0, _d("pressure", 1000.0), 1.0)

        submitted = st.form_submit_button("Predict & Save", type="primary")

    if not submitted:
        return

    inp = build_input_df(
        battery_level=bl, flight_time=ft,
        signal_strength=ss, temperature=tp,
        wind_speed=ws, gps_accuracy=ga,
        altitude=al, speed=sp, humidity=hm, pressure=pr,
    )
    risk_pred, maint_pred, rec_pred, conf = predict_drone(inp)
    flight_st, flight_reason, flight_lv = flight_decision(
        bl, ss, ws, ga, ft, tp, al, sp, risk_pred, maint_pred,
    )
    risk_vn, maint_vn = translate(risk_pred, maint_pred)
    bat_label, bat_lv = battery_status(bl)

    saved = save_custom(
        drone_id, inp, risk_pred, maint_pred, rec_pred,
        bat_label, flight_st, flight_reason,
    )

    template_note = f" (tình huống: {template_choice})" if template_choice != NONE_OPTION else ""
    render_banner(
        f"✅ Dự đoán hoàn tất — **{drone_id}**{template_note} đã được lưu vào "
        f"Data/custom_drone_data.csv.",
        "success",
    )

    render_section_label("Kết quả")
    r1, r2, r3, r4 = st.columns(4, gap="large")
    with r1: render_result_badge("Rủi ro vận hành",  risk_vn,   risk_to_level(risk_pred))
    with r2: render_result_badge("Hành động bảo trì", maint_vn, "info")
    with r3: render_result_badge("Tình trạng pin",   bat_label, bat_lv)
    with r4: render_result_badge("Trạng thái bay",   flight_st, flight_lv)

    render_banner(flight_reason, flight_lv)
    render_banner(rec_pred, risk_to_level(risk_pred))

    with st.expander("📄 Xem bản ghi vừa lưu (18 cột)"):
        st.dataframe(saved.tail(1), use_container_width=True)
        st.caption(
            f"File: Data/custom_drone_data.csv  ·  "
            f"Tổng bản ghi hiện tại: {len(saved):,}"
        )