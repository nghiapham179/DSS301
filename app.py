import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from ui import (
    load_css,
    render_top_nav,
    render_main_title,
    open_card,
    close_card,
    render_metric_card,
    render_section_title,
    render_score_box
)


# ================== CONFIG ==================

st.set_page_config(
    page_title="Drone DSS",
    page_icon="🚁",
    layout="wide"
)

load_css()
render_top_nav()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "drone_data_clean.csv"
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
    st.error("Không tìm thấy file data hoặc model. Kiểm tra lại tên folder Data/Model và tên file.")
    st.code(str(e))
    st.stop()
except Exception as e:
    st.error("Có lỗi khi load dữ liệu hoặc model.")
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


# ================== SIDEBAR ==================

st.sidebar.title("Drone DSS")
page = st.sidebar.radio(
    "Chọn trang",
    [
        "Dashboard",
        "Dự đoán",
        "Phân tích drone"
    ]
)


# ================== PAGE 1: DASHBOARD ==================

if page == "Dashboard":
    render_main_title(
        "Hệ thống hỗ trợ ra quyết định bảo trì và vận hành Drone",
        f"Tổng cộng {len(df):,} bản ghi • {df['drone_id'].nunique()} drone được phân tích"
    )

    avg_score = df["risk_score"].mean()
    high_pct = round(df["is_high_risk"].mean() * 100, 1)

    if avg_score >= 6:
        score_status = "High Risk"
        score_color = "#E24B4A"
    elif avg_score >= 3:
        score_status = "Adequate"
        score_color = "#c56b00"
    else:
        score_status = "Stable"
        score_color = "#2e7d32"

    open_card()

    col_score, col_gauge = st.columns([1.1, 2.4])

    with col_score:
        render_score_box(
            f"{avg_score:.1f}/10",
            score_status,
            f"{high_pct}% bản ghi thuộc nhóm High Risk",
            score_color
        )

    with col_gauge:
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={"text": "Overall Operation Risk Score"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": score_color},
                "steps": [
                    {"range": [0, 3], "color": "#e8f5e9"},
                    {"range": [3, 6], "color": "#fff3e0"},
                    {"range": [6, 10], "color": "#ffebee"}
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 3},
                    "thickness": 0.75,
                    "value": avg_score
                }
            }
        ))

        fig_score.update_layout(
            height=260,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_score, use_container_width=True)

    close_card()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("Tổng bản ghi", f"{len(df):,}", "records")

    with c2:
        render_metric_card("Số drone", df["drone_id"].nunique(), "drone units")

    with c3:
        render_metric_card("Risk score TB", f"{avg_score:.1f}", "out of 10")

    with c4:
        render_metric_card("Tỷ lệ High Risk", f"{high_pct}%", "cần xử lý")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        open_card()

        render_section_title(
            "Phân phối Risk Score",
            "Biểu đồ thể hiện mức rủi ro vận hành của drone."
        )

        fig = px.histogram(
            df,
            x="risk_score",
            color="operation_risk",
            nbins=11,
            color_discrete_map={
                "High": "#E24B4A",
                "Medium": "#EF9F27",
                "Low": "#639922"
            },
            barmode="overlay"
        )

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        close_card()

    with col2:
        open_card()

        render_section_title(
            "Maintenance Action",
            "Phân bố các hành động bảo trì được đề xuất."
        )

        mc = df["maintenance_action"].value_counts().reset_index()
        mc.columns = ["action", "count"]

        fig2 = px.bar(
            mc,
            x="count",
            y="action",
            orientation="h",
            color="action"
        )

        fig2.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig2, use_container_width=True)

        close_card()

    col3, col4 = st.columns(2)

    with col3:
        open_card()

        render_section_title(
            "Battery Level vs Wind Speed",
            "So sánh pin và tốc độ gió theo mức rủi ro."
        )

        sample_df = df.sample(min(3000, len(df)), random_state=42)

        fig3 = px.scatter(
            sample_df,
            x="battery_level",
            y="wind_speed",
            color="operation_risk",
            opacity=0.45,
            color_discrete_map={
                "High": "#E24B4A",
                "Medium": "#EF9F27",
                "Low": "#639922"
            }
        )

        fig3.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig3, use_container_width=True)

        close_card()

    with col4:
        open_card()

        render_section_title(
            "Risk Score trung bình theo Drone",
            "Mức rủi ro trung bình của từng drone."
        )

        drone_avg = df.groupby("drone_id")["risk_score"].mean().reset_index()

        fig4 = px.bar(
            drone_avg,
            x="drone_id",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="RdYlGn_r"
        )

        fig4.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig4, use_container_width=True)

        close_card()


# ================== PAGE 2: DỰ ĐOÁN ==================

elif page == "Dự đoán":
    render_main_title(
        "Nhập thông số Drone để nhận quyết định",
        "Hệ thống dự đoán Operation Risk, Maintenance Action và Recommendation."
    )

    open_card()

    col1, col2 = st.columns(2)

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

    close_card()

    input_data = pd.DataFrame([{
        "battery_level": battery_level,
        "flight_time": flight_time,
        "signal_strength": signal_strength,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "gps_accuracy": gps_accuracy,
        "altitude": altitude,
        "speed": speed,
        "humidity": humidity,
        "pressure": pressure
    }])

    X_input = input_data[FEATURES]

    rf_r, le_r = models["risk"]
    rf_m, le_m = models["maint"]
    rf_rc, le_rc = models["rec"]

    risk_pred = le_r.inverse_transform(rf_r.predict(X_input))[0]
    maint_pred = le_m.inverse_transform(rf_m.predict(X_input))[0]
    rec_pred = le_rc.inverse_transform(rf_rc.predict(X_input))[0]

    if hasattr(rf_r, "predict_proba"):
        risk_conf = round(float(rf_r.predict_proba(X_input).max()) * 100, 1)
    else:
        risk_conf = 0

    color_map = {
        "High": "#E24B4A",
        "Medium": "#EF9F27",
        "Low": "#639922"
    }

    icon_map = {
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟢"
    }

    est_score = min(10, round(
        (1 - battery_level / 100) * 4 +
        (wind_speed / 50) * 3 +
        (1 - signal_strength / 100) * 2 +
        (abs(temperature) / 50) * 1,
        1
    ))

    open_card()

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=est_score,
        title={"text": "Risk Score ước tính"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": color_map.get(risk_pred, "#378ADD")},
            "steps": [
                {"range": [0, 2], "color": "#EAF3DE"},
                {"range": [2, 6], "color": "#FAEEDA"},
                {"range": [6, 10], "color": "#FCEBEB"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 6
            }
        }
    ))

    fig_gauge.update_layout(
        height=280,
        margin=dict(t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    r1, r2, r3 = st.columns(3)

    r1.metric(
        f"{icon_map.get(risk_pred, '⚪')} Operation Risk",
        risk_pred,
        f"Confidence: {risk_conf}%"
    )

    r2.metric("Maintenance Action", maint_pred)

    bat_status = "Good" if battery_level > 40 else "Medium" if battery_level > 20 else "Low"

    bat_icon = {
        "Good": "🟢",
        "Medium": "🟡",
        "Low": "🔴"
    }

    r3.metric(
        f"{bat_icon[bat_status]} Battery Status",
        bat_status
    )

    st.subheader("Khuyến nghị cuối cùng")

    if risk_pred == "High":
        st.error(rec_pred)
    elif risk_pred == "Medium":
        st.warning(rec_pred)
    else:
        st.success(rec_pred)

    close_card()


# ================== PAGE 3: PHÂN TÍCH DRONE ==================

elif page == "Phân tích drone":
    render_main_title(
        "Phân tích theo từng Drone",
        "Theo dõi mức rủi ro, pin và hành động bảo trì của từng drone."
    )

    open_card()

    drone_sel = st.selectbox(
        "Chọn drone",
        sorted(df["drone_id"].unique())
    )

    close_card()

    df_d = df[df["drone_id"] == drone_sel]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("Số bản ghi", f"{len(df_d):,}", drone_sel)

    with c2:
        render_metric_card("Risk score TB", f"{df_d['risk_score'].mean():.1f}", "average risk")

    with c3:
        render_metric_card("High Risk", f"{round(df_d['is_high_risk'].mean() * 100, 1)}%", "risk ratio")

    with c4:
        render_metric_card("Battery TB", f"{df_d['battery_level'].mean():.0f}%", "average battery")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        open_card()

        render_section_title(
            f"Risk Score - {drone_sel}",
            "Phân phối điểm rủi ro của drone được chọn."
        )

        fig = px.histogram(
            df_d,
            x="risk_score",
            nbins=11,
            color_discrete_sequence=["#378ADD"]
        )

        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        close_card()

    with col2:
        open_card()

        render_section_title(
            f"Flight Time vs Battery - {drone_sel}",
            "Mối quan hệ giữa thời gian bay và pin."
        )

        sample_drone = df_d.sample(min(1000, len(df_d)), random_state=42)

        fig2 = px.scatter(
            sample_drone,
            x="flight_time",
            y="battery_level",
            color="operation_risk",
            color_discrete_map={
                "High": "#E24B4A",
                "Medium": "#EF9F27",
                "Low": "#639922"
            }
        )

        fig2.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig2, use_container_width=True)

        close_card()

    open_card()

    render_section_title(
        "Phân bố Maintenance Action",
        "Tỷ lệ các hành động bảo trì được đề xuất."
    )

    mc = df_d["maintenance_action"].value_counts().reset_index()
    mc.columns = ["action", "count"]

    fig3 = px.pie(
        mc,
        names="action",
        values="count",
        hole=0.55
    )

    fig3.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    close_card()