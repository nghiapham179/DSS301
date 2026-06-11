import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import các component từ UI mới
from ui import (
    load_css,
    render_top_nav,
    render_page_title,
    render_section_label,
    render_metric_card,
    render_metric_with_chart,
    render_risk_score,
    render_result_badge,
    render_sidebar_header,
    render_sidebar_upgrade_card,
    render_banner
)

# ================== CONFIG ==================

st.set_page_config(
    page_title="Drone DSS",
    page_icon="🚁",
    layout="wide"
)

load_css()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "drone_data_clean.csv"
CUSTOM_DATA_PATH = BASE_DIR / "Data" / "custom_drone_data.csv"
MODEL_DIR = BASE_DIR / "Model"


# ================== LOAD MODEL + DATA ==================

@st.cache_resource
def load_models():
    return {
        "risk": (
            joblib.load(MODEL_DIR / "operation_risk_model.joblib"),
            joblib.load(MODEL_DIR / "operation_risk_model_label_encoder.joblib")
        ),
        "maint": (
            joblib.load(MODEL_DIR / "maintenance_action_model.joblib"),
            joblib.load(MODEL_DIR / "maintenance_action_model_label_encoder.joblib")
        ),
        "rec": (
            joblib.load(MODEL_DIR / "recommendation_model.joblib"),
            joblib.load(MODEL_DIR / "recommendation_model_label_encoder.joblib")
        )
    }


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    models = load_models()
    df = load_data()
except FileNotFoundError as e:
    render_banner("Không tìm thấy file data hoặc model. Kiểm tra lại folder Data/Model và tên file.", "danger")
    st.code(str(e))
    st.stop()
except Exception as e:
    render_banner("Có lỗi khi load dữ liệu hoặc model.", "danger")
    st.code(str(e))
    st.stop()

# ================== FEATURES ==================

FEATURES = [
    "battery_level",
    "flight_time",
    "signal_strength",
    "temperature",
    "wind_speed",
    "gps_accuracy",
    "altitude",
    "speed",
    "humidity",
    "pressure"
]


# ================== HELPER FUNCTIONS ==================

def get_flight_status(battery_level, signal_strength, wind_speed, gps_accuracy, flight_time, temperature, altitude,
                      speed, risk_pred, maint_pred):
    maint_text = str(maint_pred).lower()

    if (
            battery_level < 20 or signal_strength < 35 or wind_speed > 35 or gps_accuracy > 10 or temperature > 45 or temperature < -10):
        return ("Cấm Bay", "Drone không nên bay do pin thấp, tín hiệu yếu, gió mạnh, GPS kém hoặc nhiệt độ nguy hiểm.",
                "danger")

    if (
            "urgent" in maint_text or "required" in maint_text or "maintenance" in maint_text or "inspect" in maint_text or "inspection" in maint_text):
        return ("Yêu Cầu Bảo Trì", "Drone cần được kiểm tra hoặc bảo trì trước khi thực hiện nhiệm vụ tiếp theo.",
                "danger")

    if risk_pred == "High":
        return ("Quay Về Trạm", "Rủi ro vận hành cao. Drone nên quay về trạm hoặc dừng nhiệm vụ để kiểm tra.",
                "warning")

    if (
            battery_level < 40 or signal_strength < 60 or wind_speed > 25 or gps_accuracy > 7 or flight_time > 40 or altitude > 350 or speed > 75):
        return ("Bay Kèm Giám Sát", "Drone có thể bay nhưng cần theo dõi sát vì một số thông số đang ở mức cần chú ý.",
                "warning")

    return ("Đủ Điều Kiện Bay", "Drone đủ điều kiện vận hành bình thường.", "success")


def save_custom_drone_data(drone_id, input_data, risk_pred, maint_pred, rec_pred, battery_status, flight_status,
                           flight_reason):
    save_df = input_data.copy()
    save_df.insert(0, "drone_id", drone_id)
    save_df["operation_risk"] = risk_pred
    save_df["maintenance_action"] = maint_pred
    save_df["recommendation"] = rec_pred
    save_df["battery_status"] = battery_status
    save_df["flight_status"] = flight_status
    save_df["flight_reason"] = flight_reason
    save_df["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    CUSTOM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            old_df = pd.read_csv(CUSTOM_DATA_PATH)
            if old_df.empty:
                final_df = save_df
            else:
                final_df = pd.concat([old_df, save_df], ignore_index=True)
        except:
            final_df = save_df
    else:
        final_df = save_df

    final_df.to_csv(CUSTOM_DATA_PATH, index=False, encoding="utf-8-sig")
    return save_df


def translate_results(risk_pred, maint_pred):
    vn_risk_map = {"High": "Nguy Hiểm", "Medium": "Cảnh Báo", "Low": "An Toàn"}
    vn_maint_map = {
        "No maintenance required": "Không yêu cầu bảo trì",
        "Inspection recommended": "Khuyến nghị kiểm tra",
        "Maintenance required": "Yêu cầu bảo trì",
        "Monitor": "Cần giám sát chặt"
    }
    return (vn_risk_map.get(risk_pred, risk_pred), vn_maint_map.get(maint_pred, maint_pred))


def get_risk_level_from_score(score):
    if score >= 6: return "High"
    if score >= 3: return "Medium"
    return "Low"


def get_level_from_risk(risk_pred):
    if risk_pred == "High": return "danger"
    if risk_pred == "Medium": return "warning"
    if risk_pred == "Low": return "success"
    return "info"


def style_plotly(fig, height=None):
    """Cập nhật style Plotly sang Premium Light Theme"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#808191", family="Nunito Sans"),
        margin=dict(t=35, b=20, l=20, r=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(color="#11142d")
        )
    )

    if height:
        fig.update_layout(height=height)

    fig.update_xaxes(
        gridcolor="rgba(0,0,0,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)"
    )

    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)"
    )

    return fig


def build_input_dataframe(battery_level, flight_time, signal_strength, temperature, wind_speed, gps_accuracy, altitude,
                          speed, humidity, pressure):
    return pd.DataFrame([{
        "battery_level": battery_level, "flight_time": flight_time, "signal_strength": signal_strength,
        "temperature": temperature, "wind_speed": wind_speed, "gps_accuracy": gps_accuracy,
        "altitude": altitude, "speed": speed, "humidity": humidity, "pressure": pressure
    }])


def predict_drone(input_data):
    X_input = input_data[FEATURES]
    rf_r, le_r = models["risk"]
    rf_m, le_m = models["maint"]
    rf_rc, le_rc = models["rec"]

    risk_pred = le_r.inverse_transform(rf_r.predict(X_input))[0]
    maint_pred = le_m.inverse_transform(rf_m.predict(X_input))[0]
    rec_pred = le_rc.inverse_transform(rf_rc.predict(X_input))[0]

    risk_conf = 0
    if hasattr(rf_r, "predict_proba"):
        risk_conf = round(float(rf_r.predict_proba(X_input).max()) * 100, 1)

    return risk_pred, maint_pred, rec_pred, risk_conf


# ================== TOP NAV & SIDEBAR ==================

# Gọi Navbar gọn gàng
render_top_nav()

with st.sidebar:
    render_sidebar_header()

    # CẬP NHẬT: Thêm Icon trực tiếp vào các lựa chọn của Menu
    page = st.radio(
        "Chuyển trang",
        [
            "🏠 Dashboard",
            "🎯 Dự đoán",
            "📝 Nhập dữ liệu drone",
            "📊 Phân tích drone"
        ]
    )

    render_sidebar_upgrade_card()

# ================== PAGE 1: DASHBOARD ==================

# CẬP NHẬT: Khớp điều kiện với chuỗi có chứa Icon
if page == "🏠 Dashboard":
    render_page_title("System Overview", "Tổng quan dữ liệu vận hành, rủi ro và hành động bảo trì.")

    avg_score = df["risk_score"].mean()
    high_pct = round(df["is_high_risk"].mean() * 100, 1)
    avg_risk_level = get_risk_level_from_score(avg_score)

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        fake_data_1 = np.random.randn(10).cumsum()
        render_metric_with_chart("Tổng Bản Ghi", f"{len(df):,}", fake_data_1, "#6b5dd3", "Dữ liệu vận hành")

    with col2:
        fake_data_2 = np.random.randn(10).cumsum()
        render_metric_with_chart("Số Drone", f"{df['drone_id'].nunique()}", fake_data_2, "#3f8cff", "Active units")

    with col3:
        render_metric_card("Tỷ lệ High Risk", f"{high_pct}%", "Cần xử lý cảnh báo", "- Cải thiện 2%")

    with col4:
        render_risk_score(avg_score, avg_risk_level, f"Điểm trung bình toàn hệ thống")

    st.divider()

    c_chart1, c_chart2 = st.columns(2, gap="large")

    with c_chart1:
        render_section_label("Phân Phối Rủi Ro (Risk Distribution)")
        fig = px.histogram(
            df, x="risk_score", color="operation_risk", nbins=11,
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
            barmode="overlay"
        )
        fig = style_plotly(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c_chart2:
        render_section_label("Hành Động Bảo Trì (Maintenance Action)")
        mc = df["maintenance_action"].value_counts().reset_index()
        mc.columns = ["action", "count"]
        fig2 = px.bar(
            mc, x="count", y="action", orientation="h", color="action",
            color_discrete_sequence=["#6b5dd3", "#3f8cff", "#ff754c", "#22c55e"]
        )
        fig2.update_layout(showlegend=False)
        fig2 = style_plotly(fig2, height=360)
        st.plotly_chart(fig2, use_container_width=True)

    c_chart3, c_chart4 = st.columns(2, gap="large")

    with c_chart3:
        render_section_label("Battery vs Wind Speed")
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        fig3 = px.scatter(
            sample_df, x="battery_level", y="wind_speed", color="operation_risk", opacity=0.6,
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
        )
        fig3 = style_plotly(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with c_chart4:
        render_section_label("Trung Bình Rủi Ro Theo Drone")
        drone_avg = df.groupby("drone_id")["risk_score"].mean().reset_index()
        fig4 = px.bar(
            drone_avg, x="drone_id", y="risk_score", color="risk_score",
            color_continuous_scale="Purples"
        )
        fig4 = style_plotly(fig4, height=380)
        st.plotly_chart(fig4, use_container_width=True)

# ================== PAGE 2: DỰ ĐOÁN ==================

elif page == "🎯 Dự đoán":
    render_page_title("Real-time Drone Prediction", "Mô phỏng thông số vận hành và nhận tư vấn thời gian thực.")
    render_banner("Điều chỉnh các slider bên dưới để xem hệ thống ra quyết định bảo trì và trạng thái bay.", "info")

    render_section_label("Input Parameters")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        battery_level = st.slider("Pin còn lại (%)", 0.0, 100.0, 60.0, 1.0)
        wind_speed = st.slider("Tốc độ gió", 0.0, 50.0, 15.0, 0.5)
        signal_strength = st.slider("Cường độ tín hiệu (%)", 0.0, 100.0, 70.0, 1.0)
        temperature = st.slider("Nhiệt độ (°C)", -40.0, 50.0, 25.0, 0.5)
        gps_accuracy = st.slider("Độ chính xác GPS", 0.0, 10.0, 5.0, 0.1)

    with col2:
        flight_time = st.slider("Thời gian bay (phút)", 0.0, 60.0, 30.0, 0.5)
        altitude = st.slider("Độ cao (m)", 0.0, 500.0, 250.0, 5.0)
        speed = st.slider("Tốc độ bay", 0.0, 100.0, 50.0, 1.0)
        humidity = st.slider("Độ ẩm (%)", 0.0, 100.0, 50.0, 1.0)
        pressure = st.slider("Áp suất (hPa)", 950.0, 1050.0, 1000.0, 1.0)

    input_data = build_input_dataframe(battery_level, flight_time, signal_strength, temperature, wind_speed,
                                       gps_accuracy, altitude, speed, humidity, pressure)
    risk_pred, maint_pred, rec_pred, risk_conf = predict_drone(input_data)
    flight_status, flight_reason, flight_level = get_flight_status(battery_level, signal_strength, wind_speed,
                                                                   gps_accuracy, flight_time, temperature, altitude,
                                                                   speed, risk_pred, maint_pred)

    risk_pred_vn, maint_pred_vn = translate_results(risk_pred, maint_pred)
    bat_status = "Tốt" if battery_level > 40 else "Trung Bình" if battery_level > 20 else "Yếu"

    est_score = min(10, round(
        (1 - battery_level / 100) * 4 + (wind_speed / 50) * 3 + (1 - signal_strength / 100) * 2 + (
                    abs(temperature) / 50) * 1, 1))

    st.divider()
    render_section_label("Decision Output")

    col_score, col_result = st.columns([1, 2], gap="large")

    with col_score:
        render_risk_score(est_score, risk_pred, f"Model confidence: {risk_conf}%")

    with col_result:
        r1, r2 = st.columns(2)
        with r1: render_result_badge("Mức Rủi Ro", risk_pred_vn, get_level_from_risk(risk_pred))
        with r2: render_result_badge("Tình Trạng Pin", bat_status,
                                     "success" if bat_status == "Tốt" else "warning" if bat_status == "Trung Bình" else "danger")

        r3, r4 = st.columns(2)
        with r3: render_result_badge("Hành Động Bảo Trì", maint_pred_vn, "info")
        with r4: render_result_badge("Trạng Thái Bay", flight_status, flight_level)

        render_banner(flight_reason, flight_level)

    st.divider()
    render_section_label("Final Recommendation")
    render_banner(rec_pred, get_level_from_risk(risk_pred))

# ================== PAGE 3: NHẬP DỮ LIỆU DRONE ==================

elif page == "📝 Nhập dữ liệu drone":
    render_page_title("Manual Drone Data Input", "Lưu trữ dữ liệu theo thời gian thực vào Data/custom_drone_data.csv.")

    with st.form("manual_drone_input_form"):
        render_section_label("Drone Information")
        drone_id_input = st.text_input("Drone ID", value="Drone_Custom_001")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            battery_level = st.number_input("Pin còn lại (%)", 0.0, 100.0, 60.0, 1.0)
            flight_time = st.number_input("Thời gian bay (phút)", 0.0, 300.0, 30.0, 1.0)
            signal_strength = st.number_input("Cường độ tín hiệu (%)", 0.0, 100.0, 70.0, 1.0)
            temperature = st.number_input("Nhiệt độ (°C)", -50.0, 80.0, 25.0, 0.5)
            wind_speed = st.number_input("Tốc độ gió", 0.0, 100.0, 15.0, 0.5)

        with col2:
            gps_accuracy = st.number_input("Độ chính xác GPS", 0.0, 50.0, 5.0, 0.1)
            altitude = st.number_input("Độ cao (m)", 0.0, 1000.0, 250.0, 5.0)
            speed = st.number_input("Tốc độ bay", 0.0, 200.0, 50.0, 1.0)
            humidity = st.number_input("Độ ẩm (%)", 0.0, 100.0, 50.0, 1.0)
            pressure = st.number_input("Áp suất (hPa)", 900.0, 1100.0, 1000.0, 1.0)

        submitted = st.form_submit_button("Predict & Save", type="primary")

    if submitted:
        input_data = build_input_dataframe(battery_level, flight_time, signal_strength, temperature, wind_speed,
                                           gps_accuracy, altitude, speed, humidity, pressure)
        risk_pred, maint_pred, rec_pred, risk_conf = predict_drone(input_data)
        flight_status, flight_reason, flight_level = get_flight_status(battery_level, signal_strength, wind_speed,
                                                                       gps_accuracy, flight_time, temperature, altitude,
                                                                       speed, risk_pred, maint_pred)
        risk_pred_vn, maint_pred_vn = translate_results(risk_pred, maint_pred)
        bat_status = "Tốt" if battery_level > 40 else "Trung Bình" if battery_level > 20 else "Yếu"

        saved_row = save_custom_drone_data(drone_id_input, input_data, risk_pred, maint_pred, rec_pred, bat_status,
                                           flight_status, flight_reason)

        render_banner("Dự đoán hoàn tất. Dữ liệu đã được lưu thành công.", "success")
        render_section_label("Prediction Result")

        r1, r2, r3, r4 = st.columns(4, gap="large")
        with r1: render_result_badge("Mức Rủi Ro", risk_pred_vn, get_level_from_risk(risk_pred))
        with r2: render_result_badge("Bảo Trì", maint_pred_vn, "info")
        with r3: render_result_badge("Tình Trạng Pin", bat_status,
                                     "success" if bat_status == "Tốt" else "warning" if bat_status == "Trung Bình" else "danger")
        with r4: render_result_badge("Trạng Thái Bay", flight_status, flight_level)

        render_banner(flight_reason, flight_level)
        render_banner(rec_pred, get_level_from_risk(risk_pred))

        with st.expander("Xem dữ liệu vừa lưu"):
            st.dataframe(saved_row, use_container_width=True)

# ================== PAGE 4: PHÂN TÍCH DRONE ==================

elif page == "📊 Phân tích drone":
    render_page_title("Drone Unit Analysis", "Theo dõi sức khỏe và vòng đời bảo trì theo từng thiết bị.")
    render_section_label("Select Drone")

    drone_sel = st.selectbox("Chọn thiết bị", sorted(df["drone_id"].unique()), label_visibility="collapsed")
    df_d = df[df["drone_id"] == drone_sel]
    st.write("")

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        render_metric_card("Tổng bản ghi", f"{len(df_d):,}", drone_sel)
    with c2:
        render_metric_card("Risk Score TB", f"{df_d['risk_score'].mean():.1f}", "Rủi ro trung bình")
    with c3:
        render_metric_card("High Risk", f"{round(df_d['is_high_risk'].mean() * 100, 1)}%", "Tỷ lệ nguy hiểm")
    with c4:
        render_metric_card("Battery TB", f"{df_d['battery_level'].mean():.0f}%", "Dung lượng pin TB")

    st.divider()
    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_section_label(f"Mức Độ Rủi Ro - {drone_sel}")
        fig = px.histogram(df_d, x="risk_score", nbins=11, color_discrete_sequence=["#6b5dd3"])
        fig = style_plotly(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        render_section_label(f"Flight Time vs Battery - {drone_sel}")
        sample_drone = df_d.sample(min(1000, len(df_d)), random_state=42)
        fig2 = px.scatter(
            sample_drone, x="flight_time", y="battery_level", color="operation_risk",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
        )
        fig2 = style_plotly(fig2, height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.write("")
    render_section_label("Phân Bổ Hành Động Bảo Trì")
    mc = df_d["maintenance_action"].value_counts().reset_index()
    mc.columns = ["action", "count"]

    fig3 = px.pie(
        mc, names="action", values="count", hole=0.6,
        color_discrete_sequence=["#6b5dd3", "#3f8cff", "#ff754c", "#22c55e"]
    )
    fig3 = style_plotly(fig3, height=420)
    st.plotly_chart(fig3, use_container_width=True)