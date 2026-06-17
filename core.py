"""
core.py — Drone DSS shared logic
=================================
Bao gồm:
  • Paths, palette, FEATURES constants
  • Cached loaders: models, data, metrics, feature importance
  • Helper functions: predict, flight_decision, battery_status, save_custom,
    style_chart, make_radar_chart, translate, …

Tất cả các page trong views/ đều import từ đây để tránh lặp code.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─── PATHS ──────────────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).resolve().parent
DATA_PATH        = BASE_DIR / "Data" / "drone_data_clean.csv"
CUSTOM_DATA_PATH = BASE_DIR / "Data" / "custom_drone_data.csv"
MODEL_DIR        = BASE_DIR / "Model"


# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────

COLORS = {
    "accent":  "#4f63d2",
    "blue":    "#3f8cff",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger":  "#dc2626",
}
RISK_COLOR_MAP = {
    "High":   COLORS["danger"],
    "Medium": COLORS["warning"],
    "Low":    COLORS["success"],
}
SEQ = [COLORS["accent"], COLORS["blue"], COLORS["warning"], COLORS["success"]]


# ─── FEATURES (README Section 10) ───────────────────────────────────────────

FEATURES = [
    "battery_level", "flight_time", "signal_strength",
    "temperature", "wind_speed", "gps_accuracy",
    "altitude", "speed", "humidity", "pressure",
]


# ─── TEMPLATES (8 kịch bản chuẩn) ───────────────────────────────────────────

TEMPLATES = {
    "Tối ưu (Mọi điều kiện hoàn hảo)": {
        "icon": "🌟", "name": "Điều kiện tối ưu", "desc": "Mọi thông số đều ở mức lý tưởng, an toàn tuyệt đối để cất cánh.",
        "params": dict(battery_level=95, flight_time=15, signal_strength=98, temperature=25,
                       wind_speed=3, gps_accuracy=1.2, altitude=50, speed=15, humidity=40, pressure=1012)
    },
    "Pin cạn kiệt (< 20%)": {
        "icon": "🔋", "name": "Cạn kiệt pin", "desc": "Mô phỏng rủi ro khi dung lượng pin xuống mức nguy hiểm.",
        "params": dict(battery_level=15, flight_time=8, signal_strength=85, temperature=22,
                       wind_speed=5, gps_accuracy=2.0, altitude=30, speed=10, humidity=50, pressure=1010)
    },
    "Gió bão (Nguy hiểm)": {
        "icon": "🌪️", "name": "Gió bão/Thời tiết xấu", "desc": "Tốc độ gió và điều kiện môi trường vượt giới hạn chịu đựng.",
        "params": dict(battery_level=80, flight_time=20, signal_strength=60, temperature=18,
                       wind_speed=38, gps_accuracy=6.5, altitude=120, speed=25, humidity=70, pressure=995)
    },
    "Quá nhiệt (> 45°C)": {
        "icon": "🔥", "name": "Cảnh báo quá nhiệt", "desc": "Nhiệt độ môi trường cực kỳ cao, ảnh hưởng đến motor và pin.",
        "params": dict(battery_level=75, flight_time=30, signal_strength=88, temperature=48,
                       wind_speed=4, gps_accuracy=1.8, altitude=40, speed=12, humidity=30, pressure=1008)
    },
    "Lạnh giá, dễ sập nguồn (< -10°C)": {
        "icon": "❄️", "name": "Môi trường băng giá", "desc": "Nhiệt độ âm sâu có thể gây hiện tượng sụt pin đột ngột.",
        "params": dict(battery_level=60, flight_time=10, signal_strength=82, temperature=-12,
                       wind_speed=8, gps_accuracy=2.5, altitude=80, speed=15, humidity=45, pressure=1020)
    },
    "Mất kết nối (Signal < 35%)": {
        "icon": "📡", "name": "Mất sóng/Nhiễu tín hiệu", "desc": "Kết nối giữa bộ điều khiển và drone bị gián đoạn nghiêm trọng.",
        "params": dict(battery_level=85, flight_time=25, signal_strength=25, temperature=24,
                       wind_speed=6, gps_accuracy=8.0, altitude=200, speed=18, humidity=55, pressure=1011)
    },
    "Độ cao/Tốc độ vượt mức (High Risk)": {
        "icon": "🚀", "name": "Vượt thông số bay", "desc": "Drone bay quá cao và quá nhanh, rủi ro va chạm mất kiểm soát cao.",
        "params": dict(battery_level=88, flight_time=35, signal_strength=70, temperature=20,
                       wind_speed=12, gps_accuracy=4.0, altitude=400, speed=85, humidity=60, pressure=1005)
    },
    "Bảo trì định kỳ (Thời gian bay cao)": {
        "icon": "🛠️", "name": "Yêu cầu bảo dưỡng", "desc": "Drone đã hoạt động liên tục trong thời gian dài, cần kiểm tra hao mòn.",
        "params": dict(battery_level=45, flight_time=50, signal_strength=90, temperature=26,
                       wind_speed=5, gps_accuracy=1.5, altitude=100, speed=20, humidity=50, pressure=1010)
    },
}

# Alias để fix lỗi import trong parameters.py
TEMPLATE_LOOKUP = TEMPLATES


# ─── CACHED LOADERS ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models() -> dict:
    """Load all 3 RF models + label encoders."""
    names = {
        "risk":  "operation_risk_model",
        "maint": "maintenance_action_model",
        "rec":   "recommendation_model",
    }
    return {
        key: (
            joblib.load(MODEL_DIR / f"{name}.joblib"),
            joblib.load(MODEL_DIR / f"{name}_label_encoder.joblib"),
        )
        for key, name in names.items()
    }


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load data gốc và tự động gộp với dữ liệu thực địa (custom_drone_data)."""
    # 1. Đọc data gốc
    df_base = pd.read_csv(DATA_PATH)

    # 2. Đọc data thực địa vừa nhập (nếu có)
    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            df_custom = pd.read_csv(CUSTOM_DATA_PATH)

            # Đồng bộ tên định danh (Khớp cột drone_id của file custom với file gốc)
            if 'drone_id' in df_custom.columns and 'Drone_ID' in df_base.columns:
                df_custom = df_custom.rename(columns={'drone_id': 'Drone_ID'})

            # Gộp 2 tập dữ liệu lại thành 1
            df_combined = pd.concat([df_base, df_custom], ignore_index=True)
            return df_combined
        except Exception:
            pass

    return df_base


@st.cache_data
def load_metrics(model_name: str):
    """Load <model>_metrics.json generated by train_model.py."""
    path = MODEL_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    """Load feature_importance.csv generated by train_model.py."""
    path = MODEL_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ─── INPUT / PREDICTION ─────────────────────────────────────────────────────

def build_input_df(**kw) -> pd.DataFrame:
    """Build single-row DataFrame from kwargs matching FEATURES."""
    return pd.DataFrame([{k: kw[k] for k in FEATURES}])


def predict_drone(inp: pd.DataFrame):
    """Run all 3 classifiers. Returns (risk, maint, rec, confidence%)."""
    models = load_models()
    X = inp[FEATURES]
    rf_r,  le_r  = models["risk"]
    rf_m,  le_m  = models["maint"]
    rf_rc, le_rc = models["rec"]

    risk  = le_r.inverse_transform(rf_r.predict(X))[0]
    maint = le_m.inverse_transform(rf_m.predict(X))[0]
    rec   = le_rc.inverse_transform(rf_rc.predict(X))[0]

    conf = 0.0
    if hasattr(rf_r, "predict_proba"):
        conf = round(float(rf_r.predict_proba(X).max()) * 100, 1)

    return risk, maint, rec, conf


# ─── DECISION RULES (README Section 11) ─────────────────────────────────────

def flight_decision(battery, signal, wind, gps, flight_time,
                    temp, alt, speed, risk, maint):
    """Rule-based flight status. Returns (label, reason, level)."""
    ml = str(maint).lower()

    # Cấm Bay
    reasons_critical = []
    if battery < 20:     reasons_critical.append("pin dưới 20%")
    if signal  < 35:     reasons_critical.append("tín hiệu dưới 35%")
    if wind    > 35:     reasons_critical.append("gió trên 35 m/s")
    if gps     > 10:     reasons_critical.append("GPS accuracy trên 10 m")
    if temp    > 45:     reasons_critical.append("nhiệt độ trên 45°C")
    if temp    < -10:    reasons_critical.append("nhiệt độ dưới -10°C")
    if reasons_critical:
        return ("Cấm Bay",
                "Điều kiện nguy hiểm: " + ", ".join(reasons_critical) + ".",
                "danger")

    if any(kw in ml for kw in ("maintenance required", "inspection recommended",
                               "inspect", "urgent")):
        return ("Yêu Cầu Bảo Trì",
                "Drone cần kiểm tra hoặc bảo trì trước chuyến bay tiếp theo.",
                "danger")

    if risk == "High":
        return ("Quay Về Trạm",
                "Rủi ro vận hành cao. Drone nên quay về trạm để kiểm tra.",
                "warning")

    reasons_monitor = []
    if flight_time > 40: reasons_monitor.append("thời gian bay trên 40 phút")
    if alt > 350:        reasons_monitor.append("độ cao trên 350 m")
    if speed > 75:       reasons_monitor.append("tốc độ trên 75")
    if battery < 40:     reasons_monitor.append("pin dưới 40%")
    if signal < 60:      reasons_monitor.append("tín hiệu dưới 60%")
    if wind > 25:        reasons_monitor.append("gió trên 25 m/s")
    if gps > 7:          reasons_monitor.append("GPS accuracy trên 7 m")
    if reasons_monitor:
        return ("Bay Kèm Giám Sát",
                "Cần theo dõi sát: " + ", ".join(reasons_monitor) + ".",
                "warning")

    return ("Đủ Điều Kiện Bay", "Drone đủ điều kiện vận hành bình thường.", "success")


def battery_status(level: float):
    """(label, alert_level)"""
    if level > 40: return "Tốt", "success"
    if level > 20: return "Trung Bình", "warning"
    return "Yếu", "danger"


def risk_to_level(risk: str) -> str:
    return {"High": "danger", "Medium": "warning", "Low": "success"}.get(risk, "info")


# ─── TRANSLATION ────────────────────────────────────────────────────────────

_VN_RISK  = {"High": "Nguy Hiểm", "Medium": "Cảnh Báo", "Low": "An Toàn"}
_VN_MAINT = {
    "No maintenance required": "Không yêu cầu bảo trì",
    "Inspection recommended":  "Khuyến nghị kiểm tra",
    "Maintenance required":    "Yêu cầu bảo trì",
    "Monitor":                 "Cần giám sát chặt",
}


def translate(risk: str, maint: str):
    return _VN_RISK.get(risk, risk), _VN_MAINT.get(maint, maint)


def risk_score_estimate(battery, wind, signal, temp) -> float:
    """Weighted risk score 0–10."""
    return min(10.0, round(
        (1 - battery / 100) * 4.0
        + (wind / 50)        * 3.0
        + (1 - signal / 100) * 2.0
        + (abs(temp) / 50)   * 1.0,
        1,
        ))


# ─── PERSISTENCE ────────────────────────────────────────────────────────────

# Danh sách 49 cột chuẩn từ drone_data_clean.csv
STANDARD_COLUMNS = [
    "latitude", "longitude", "altitude", "speed", "heading", "battery_level",
    "drone_id", "flight_time", "signal_strength", "temperature", "humidity",
    "pressure", "wind_speed", "wind_direction", "gps_accuracy", "battery_status",
    "charging_action", "battery_bin", "flight_time_bin", "risk_score",
    "operation_risk", "is_high_risk", "maintenance_action", "recommendation",
    "ext_drone_id", "application", "drone_size", "drone_model", "manufacturer",
    "propeller_count", "max_carry_weight_kg", "actual_carry_weight_kg",
    "payload_type", "payload_description", "supp_altitude_m",
    "supp_flight_duration_min", "distance_flown_km", "operator_id",
    "flight_date", "supp_battery_remaining_pct", "supp_gps_accuracy_m",
    "supp_wind_speed_mps", "obstacles_encountered", "flight_status",
    "regulatory_approval_id", "operation_notes", "payload_load_ratio",
    "combined_maintenance_priority", "combined_decision_reason"
]

def save_custom(drone_id, inp, risk, maint, rec, bat_label, status, reason):
    """Lưu bản ghi vào custom_drone_data.csv theo đúng 49 cột chuẩn."""
    row = inp.copy()

    # 1. Ánh xạ các giá trị hiện có vào đúng tên cột chuẩn
    row["drone_id"] = drone_id
    row["operation_risk"] = risk
    row["maintenance_action"] = maint
    row["recommendation"] = rec
    row["battery_status"] = bat_label
    row["flight_status"] = status
    row["combined_decision_reason"] = reason  # Đổi tên khớp format
    row["flight_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Tự động tính toán thêm các cột có thể suy luận
    row["risk_score"] = risk_score_estimate(
        row["battery_level"].iloc[0], row["wind_speed"].iloc[0],
        row["signal_strength"].iloc[0], row["temperature"].iloc[0]
    )
    row["is_high_risk"] = 1 if risk == "High" else 0

    # 3. Điền rỗng (NaN) cho các cột không có trong form để giữ chuẩn format
    for col in STANDARD_COLUMNS:
        if col not in row.columns:
            row[col] = pd.NA

    # Sắp xếp lại đúng thứ tự 49 cột
    row = row[STANDARD_COLUMNS]

    CUSTOM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            old = pd.read_csv(CUSTOM_DATA_PATH)
            if not old.empty:
                # Ép file cũ theo đúng chuẩn 49 cột nếu file cũ bị thiếu
                for col in STANDARD_COLUMNS:
                    if col not in old.columns:
                        old[col] = pd.NA
                row = pd.concat([old[STANDARD_COLUMNS], row], ignore_index=True)
        except (pd.errors.EmptyDataError, Exception):
            pass

    row.to_csv(CUSTOM_DATA_PATH, index=False, encoding="utf-8-sig")

    # Xóa cache để cập nhật các biểu đồ/Dashboard ngay lập tức
    load_data.clear()

    return row


# ─── CHART STYLING ──────────────────────────────────────────────────────────

def style_chart(fig: go.Figure, height: int = None) -> go.Figure:
    """Unified Plotly theme."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64697a", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(t=28, b=16, l=12, r=12),
        legend=dict(
            bgcolor="rgba(255,255,255,.8)",
            bordercolor="rgba(0,0,0,.06)",
            borderwidth=1,
            font=dict(color="#1c1e2e", size=11),
        ),
        **({"height": height} if height else {}),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,.04)", zerolinecolor="rgba(0,0,0,.04)",
                     linecolor="rgba(0,0,0,.08)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,.04)", zerolinecolor="rgba(0,0,0,.04)",
                     linecolor="rgba(0,0,0,.08)")
    return fig


def make_radar_chart(params: dict, safe_threshold: float = 0.6) -> go.Figure:
    """Radar chart so sánh thông số hiện tại vs ngưỡng an toàn."""
    labels = list(params.keys())
    vals   = [round(v, 3) for v in params.values()]
    labels_c = labels + [labels[0]]
    vals_c   = vals + [vals[0]]
    thresh_c = [safe_threshold] * len(labels_c)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=thresh_c, theta=labels_c,
        fill="toself", fillcolor="rgba(22,163,74,.08)",
        line=dict(color="rgba(22,163,74,.4)", width=1.5, dash="dot"),
        name="Ngưỡng an toàn",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=labels_c,
        fill="toself", fillcolor="rgba(79,99,210,.15)",
        line=dict(color="#4f63d2", width=2.5),
        name="Thông số hiện tại",
        marker=dict(size=6, color="#4f63d2"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=9, color="#9da3b5"),
                            gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.08)"),
            angularaxis=dict(tickfont=dict(size=10, color="#1c1e2e"),
                             gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.08)"),
        ),
        legend=dict(font=dict(size=11, color="#64697a"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        margin=dict(t=24, b=50, l=40, r=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui"),
    )
    return fig


# ─── STARTUP CHECK ──────────────────────────────────────────────────────────

def startup_load_or_stop():
    """
    Try to load models + data, show error banner and stop if either fails.
    Returns (models, df) on success.
    """
    from ui import render_banner
    try:
        models = load_models()
        df     = load_data()
        return models, df
    except FileNotFoundError as e:
        render_banner(
            "Không tìm thấy file data hoặc model.  "
            "Kiểm tra thư mục Data/ và Model/, sau đó chạy:  `python train_model.py`",
            "danger",
        )
        st.code(str(e))
        st.stop()
    except Exception as e:
        render_banner("Lỗi không xác định khi tải dữ liệu hoặc model.", "danger")
        st.code(str(e))
        st.stop()