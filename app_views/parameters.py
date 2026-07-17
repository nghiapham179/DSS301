"""
app_views/parameters.py — Không gian Dự đoán & Nhập liệu (Prediction Workspace)
================================================================================
3 tab:
  1. Tab Slider: Mô phỏng nhanh qua thanh trượt, không lưu. Hiển thị Radar Chart.
  2. Tab Form:   Nhập dữ liệu thực địa, chọn tình huống mẫu, lưu vào custom_drone_data.csv.
  3. Tab Batch:  Tải lên file CSV chứa nhiều bản ghi để dự đoán hàng loạt.
"""

import pandas as pd
import streamlit as st
import joblib

from core import (
    build_input_df, predict_drone, flight_decision, battery_status,
    translate, risk_score_estimate, risk_to_level, save_custom,
    make_radar_chart, startup_load_or_stop, cost_sensitive_risk_labels,
    TEMPLATES, TEMPLATE_LOOKUP, load_data, CUSTOM_DATA_PATH,
    DRONE_PROFILES, DEFAULT_PROFILE, DRONE_NAME_MAP, get_profile,
)
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_risk_score, render_result_badge, render_banner,
)

# ─── UTILS CHO TAB 2 & 3 ────────────────────────────────────────────────────

def get_all_drone_ids():
    """Quét và lấy danh sách toàn bộ Drone ID từ cả drone_data_clean.csv và custom_drone_data.csv."""
    ids = set()
    try:
        df_base = load_data()
        for col in ["Drone_ID", "drone_id"]:
            if col in df_base.columns:
                ids.update(df_base[col].dropna().astype(str).tolist())
    except Exception:
        pass

    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            df_custom = pd.read_csv(CUSTOM_DATA_PATH)
            if "drone_id" in df_custom.columns:
                ids.update(df_custom["drone_id"].dropna().astype(str).tolist())
        except Exception:
            pass
    return sorted(list(ids))

@st.cache_resource
def load_ml_pipeline():
    """Tải model trực tiếp phục vụ cho Batch Prediction siêu tốc."""
    try:
        risk_model = joblib.load("Model/operation_risk_model.joblib")
        maint_model = joblib.load("Model/maintenance_action_model.joblib")
        recom_model = joblib.load("Model/recommendation_model.joblib")
        risk_le = joblib.load("Model/operation_risk_model_label_encoder.joblib")
        maint_le = joblib.load("Model/maintenance_action_model_label_encoder.joblib")
        recom_le = joblib.load("Model/recommendation_model_label_encoder.joblib")
        return (risk_model, risk_le), (maint_model, maint_le), (recom_model, recom_le)
    except Exception as e:
        return None, str(e)

def validate_and_format_data(df_raw):
    """Ép khuôn 10 features chuẩn cho file CSV upload."""
    EXPECTED_FEATURES = [
        "battery_level", "flight_time", "signal_strength", "temperature",
        "wind_speed", "gps_accuracy", "altitude", "speed", "humidity", "pressure"
    ]
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_cols:
        return None, f"Thiếu các cột bắt buộc: {', '.join(missing_cols)}"
    df_clean = df[EXPECTED_FEATURES]
    try:
        df_clean = df_clean.astype(float)
    except ValueError:
        return None, "Dữ liệu chứa ký tự không hợp lệ (không phải số)."
    return df_clean, "Success"

NONE_OPTION = "— Tự nhập thủ công —"


# ─── HỒ SƠ QUY TẮC THEO DÒNG DRONE ──────────────────────────────────────────

def _profile_selector(key: str):
    """Selectbox chọn dòng drone → trả về profile quy tắc tương ứng."""
    options = [DEFAULT_PROFILE["name"]] + list(DRONE_PROFILES.keys())
    choice = st.selectbox(
        "Hồ sơ quy tắc vận hành (ngưỡng an toàn theo dòng drone)",
        options, key=key,
    )
    return DRONE_PROFILES.get(choice, DEFAULT_PROFILE)


def _render_profile_specs(p: dict):
    """Bảng ngưỡng của hồ sơ đang chọn — để phi công biết luật đang áp dụng."""
    with st.expander(f"{p['icon']} Ngưỡng an toàn đang áp dụng — {p['name']}",
                     expanded=False):
        rows = [
            ("💨 Gió — cấm bay / giám sát",
             f"> {p['wind_nofly']:.1f} m/s  /  > {p['wind_monitor']:.1f} m/s"),
            ("🌡️ Nhiệt độ — dải vận hành / vùng thoải mái",
             f"{p['temp_min']:.0f}…{p['temp_max']:.0f}°C  /  "
             f"{p['temp_monitor_low']:.0f}…{p['temp_monitor_high']:.0f}°C"),
            ("🔋 Pin — khẩn cấp (cấm bay) / quay về trạm",
             f"< {p['battery_critical']:.0f}%  /  < {p['battery_rth']:.0f}%"),
            ("⏱️ Thời lượng bay — tối đa / giám sát",
             (f"{p['max_flight_time']:.0f} phút" if p["max_flight_time"] else "—")
             + f"  /  > {p['flight_time_monitor']:.0f} phút"),
            ("🚀 Tốc độ — tối đa / giám sát",
             (f"{p['max_speed']:.0f} m/s" if p["max_speed"] else "—")
             + f"  /  > {p['speed_monitor']:.0f}"),
            ("🏔️ Độ cao — trần cấp phép / giám sát",
             (f"{p['legal_alt']:.0f} m" if p["legal_alt"] else "—")
             + f"  /  > {p['alt_monitor']:.0f} m"),
            ("📶 Tín hiệu — cấm bay / giám sát",
             f"< {p['signal_nofly']:.0f}%  /  < {p['signal_monitor']:.0f}%"),
            ("📍 GPS accuracy — cấm bay / giám sát",
             f"> {p['gps_nofly']:.0f} m  /  > {p['gps_monitor']:.0f} m"),
        ]
        st.table(pd.DataFrame(rows, columns=["Thông số", "Ngưỡng"]))
        st.caption(f"📚 Nguồn ngưỡng: **{p['source']}**"
                   + (f" · Khối lượng: {p['weight_g']} g" if p["weight_g"] else "")
                   + ". Drone càng nhẹ, ngưỡng pin quay về càng cao (dự phòng "
                     "năng lượng trước gió).")


def _rule_override_note(risk_pred: str, flight_lv: str):
    """
    Hiển thị khi tầng quy tắc (knowledge-driven) nghiêm khắc hơn ML
    (data-driven) — minh họa thứ tự ưu tiên trong DSS hybrid: ràng buộc
    an toàn cứng theo spec hãng luôn ghi đè khuyến nghị từ mô hình.
    """
    if flight_lv == "danger" and risk_pred in ("Low", "Medium"):
        render_banner(
            "⚖️ **Tầng quy tắc an toàn ghi đè ML:** mô hình ML đánh giá rủi ro "
            f"**{risk_pred}** (theo thang dữ liệu huấn luyện), nhưng thông số "
            "vi phạm ngưỡng cứng theo spec của dòng drone đang chọn → hệ thống "
            "ưu tiên quyết định của tầng quy tắc (kiến trúc DSS hybrid: "
            "knowledge-driven > data-driven cho ràng buộc an toàn).",
            "warning",
        )


# ─── RENDER GIAO DIỆN CHÍNH ─────────────────────────────────────────────────

def render():
    render_top_nav()
    startup_load_or_stop()

    render_page_title(
        "Không gian Dự đoán (Prediction Workspace)",
        "Hỗ trợ ra quyết định thông qua mô phỏng, nhập liệu thực địa, và phân tích hàng loạt.",
    )

    # Khởi tạo 3 tab tính năng
    tab_slider, tab_form, tab_batch = st.tabs([
        "🎛️  Mô phỏng nhanh (Slider)",
        "📝  Nhập liệu thực địa (Form)",
        "📂  Dự đoán hàng loạt (CSV)"
    ])

    with tab_slider: _render_slider_tab()
    with tab_form:   _render_form_tab()
    with tab_batch:  _render_batch_tab()


# ─── TAB 1: SLIDER ──────────────────────────────────────────────────────────

def _render_slider_tab():
    render_banner("Kéo slider để mô phỏng tình huống — kết quả cập nhật tức thì, không lưu lại.", "info")

    render_section_label("Dòng drone & hồ sơ quy tắc")
    prof = _profile_selector("slider_profile")
    _render_profile_specs(prof)

    render_section_label("Thông số đầu vào (10 features)")
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
        profile=prof,
    )
    risk_vn, maint_vn = translate(risk_pred, maint_pred)
    bat_label, bat_lv = battery_status(battery_level, profile=prof)
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
        st.plotly_chart(make_radar_chart(params, safe_threshold=0.6), width="stretch", config={"displayModeBar": False})

    _rule_override_note(risk_pred, flight_lv)
    render_banner(flight_reason, flight_lv)
    st.divider()
    render_section_label("Khuyến nghị cuối cùng")
    render_banner(rec_pred, risk_to_level(risk_pred))


# ─── TAB 2: FORM (LƯU DỮ LIỆU) ──────────────────────────────────────────────

def _render_form_tab():
    render_banner("Nhập dữ liệu thực địa chi tiết — kết quả sẽ được **lưu vào** `Data/custom_drone_data.csv`.", "info")

    render_section_label("Áp dụng tình huống mẫu (tuỳ chọn)")
    template_choice = st.selectbox(
        "Tình huống mẫu", [NONE_OPTION] + list(TEMPLATE_LOOKUP.keys()),
        key="template_selector", label_visibility="collapsed"
    )

    if template_choice != NONE_OPTION:
        t = TEMPLATE_LOOKUP[template_choice]
        defaults = t["params"]
        st.caption(f"💡 **{t['icon']} {t['name']}** — {t['desc']}")
    else:
        defaults = None

    def _d(key, fallback):
        return float(defaults[key]) if defaults and key in defaults else fallback

    render_section_label("Thông tin thiết bị vận hành")
    existing_ids = get_all_drone_ids()
    dropdown_options = ["➕ -- Tạo Drone Mới --"] + existing_ids

    selected_drone_opt = st.selectbox(
        "Chọn Drone ID từ Dataset hoặc khởi tạo thực thể mới",
        options=dropdown_options, key="drone_id_selector", label_visibility="collapsed"
    )

    if selected_drone_opt == "➕ -- Tạo Drone Mới --":
        drone_id = st.text_input("Nhập ID Drone mới", value="Drone_Custom_001")
        prof = _profile_selector("form_profile")
    else:
        drone_id = selected_drone_opt
        # Drone trong hạm đội đã biết dòng máy → tự áp hồ sơ quy tắc tương ứng
        known_model = DRONE_NAME_MAP.get(drone_id)
        if known_model:
            prof = get_profile(drone_id)
            st.info(f"📍 Thiết bị **{drone_id}** thuộc dòng **{known_model}** — "
                    f"tự động áp dụng hồ sơ quy tắc của dòng máy này.")
        else:
            prof = _profile_selector("form_profile")
            st.info(f"📍 Thiết lập thông số thực địa mới cho thiết bị: **{drone_id}**")
    _render_profile_specs(prof)

    with st.form("drone_manual_form"):
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

    if submitted:
        inp = build_input_df(
            battery_level=bl, flight_time=ft, signal_strength=ss, temperature=tp,
            wind_speed=ws, gps_accuracy=ga, altitude=al, speed=sp, humidity=hm, pressure=pr,
        )
        risk_pred, maint_pred, rec_pred, conf = predict_drone(inp)
        flight_st, flight_reason, flight_lv = flight_decision(
            bl, ss, ws, ga, ft, tp, al, sp, risk_pred, maint_pred,
            profile=prof,
        )
        risk_vn, maint_vn = translate(risk_pred, maint_pred)
        bat_label, bat_lv = battery_status(bl, profile=prof)

        saved = save_custom(
            drone_id, inp, risk_pred, maint_pred, rec_pred,
            bat_label, flight_st, flight_reason,
            drone_model=prof["name"] if prof is not DEFAULT_PROFILE else None,
        )

        template_note = f" (tình huống: {template_choice})" if template_choice != NONE_OPTION else ""
        render_banner(f"✅ Dự đoán hoàn tất — **{drone_id}**{template_note} đã được lưu vào Data/custom_drone_data.csv.", "success")

        render_section_label("Kết quả")
        r1, r2, r3, r4 = st.columns(4, gap="large")
        with r1: render_result_badge("Rủi ro vận hành",  risk_vn,   risk_to_level(risk_pred))
        with r2: render_result_badge("Hành động bảo trì", maint_vn, "info")
        with r3: render_result_badge("Tình trạng pin",   bat_label, bat_lv)
        with r4: render_result_badge("Trạng thái bay",   flight_st, flight_lv)

        _rule_override_note(risk_pred, flight_lv)
        render_banner(flight_reason, flight_lv)
        render_banner(rec_pred, risk_to_level(risk_pred))

        with st.expander("📄 Xem bản ghi vừa lưu (49 cột chuẩn)"):
            st.dataframe(saved.tail(1), width="stretch")
            st.caption(f"File: Data/custom_drone_data.csv  ·  Tổng bản ghi hiện tại: {len(saved):,}")


# ─── TAB 3: BATCH PREDICTION (UPLOAD CSV) ───────────────────────────────────

def _render_batch_tab():
    render_banner("Tải lên file CSV chứa dữ liệu cảm biến thô để tự động phân tích toàn hạm đội.", "info")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Kéo thả file .CSV dữ liệu Drone vào đây", type=["csv"])
        st.caption("💡 *File CSV cần chứa đủ 10 cột thông số cảm biến. Tên cột có thể lộn xộn vị trí, hệ thống sẽ tự sắp xếp.*")

    if uploaded_file is not None:
        st.divider()
        render_section_label("Tiến trình Xử lý & Dự đoán")

        with st.status("Đang phân tích dữ liệu...", expanded=True) as status:
            try:
                df_raw = pd.read_csv(uploaded_file)
                df_clean, msg = validate_and_format_data(df_raw)

                if df_clean is None:
                    status.update(label="Thất bại", state="error", expanded=True)
                    st.error(msg)
                    return

                pipeline_data = load_ml_pipeline()
                if pipeline_data[0] is None:
                    status.update(label="Lỗi Model", state="error", expanded=True)
                    st.error(f"Lỗi tải Model: {pipeline_data[1]}")
                    return

                (risk_model, risk_le), (maint_model, maint_le), (recom_model, recom_le) = pipeline_data

                # operation_risk: quy tắc Bayes tối thiểu chi phí (Elkan 2001)
                # — đồng bộ với tab Slider/Form (core.predict_drone)
                risk_labels, _ = cost_sensitive_risk_labels(risk_model, risk_le, df_clean)
                df_raw['PREDICTED_Operation_Risk'] = risk_labels
                X_batch = df_clean.to_numpy()   # model fit không tên cột — tránh sklearn warning
                df_raw['PREDICTED_Maintenance_Action'] = maint_le.inverse_transform(maint_model.predict(X_batch))
                df_raw['PREDICTED_Recommendation'] = recom_le.inverse_transform(recom_model.predict(X_batch))

                status.update(label=f"Hoàn tất xử lý {len(df_raw):,} bản ghi!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Lỗi không xác định", state="error", expanded=True)
                st.error(f"Đã xảy ra lỗi: {str(e)}")
                return

        st.success("✅ Phân tích thành công!")
        st.dataframe(df_raw.head(10), width="stretch")

        csv_data = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Tải File Kết Quả Phân Tích (.CSV)",
            data=csv_data,
            file_name="drone_batch_predictions.csv",
            mime="text/csv",
            type="primary",
            width="stretch"
        )