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


# Cột số theo data dictionary — ép kiểu khi đọc để một ô rác không làm
# hỏng kiểu của cả cột (dataset gốc có 1 giá trị rác trong wind_direction).
NUMERIC_COLUMNS = [
    "latitude", "longitude", "altitude", "speed", "heading", "battery_level",
    "flight_time", "signal_strength", "temperature", "humidity", "pressure",
    "wind_speed", "wind_direction", "gps_accuracy", "risk_score", "is_high_risk",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ép các cột số về dạng số; giá trị không hợp lệ → NaN."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load data gốc và tự động gộp với dữ liệu thực địa (custom_drone_data)."""
    # 1. Đọc data gốc — low_memory=False: đọc cả cột 1 lần để suy kiểu nhất
    #    quán (tránh DtypeWarning do pandas suy kiểu theo từng chunk).
    df_base = _coerce_numeric(pd.read_csv(DATA_PATH, low_memory=False))

    # 2. Đọc data thực địa vừa nhập (nếu có)
    if CUSTOM_DATA_PATH.exists() and CUSTOM_DATA_PATH.stat().st_size > 0:
        try:
            df_custom = _coerce_numeric(
                pd.read_csv(CUSTOM_DATA_PATH, low_memory=False))

            # Đồng bộ tên định danh (Khớp cột drone_id của file custom với file gốc)
            if 'drone_id' in df_custom.columns and 'Drone_ID' in df_base.columns:
                df_custom = df_custom.rename(columns={'drone_id': 'Drone_ID'})

            # Gộp 2 tập dữ liệu lại thành 1
            df_combined = pd.concat([df_base, df_custom], ignore_index=True)

            # Vá dữ liệu custom cũ thiếu cột suy luận (risk_score NaN làm
            # crash scatter 'size=' trên Dashboard): tính lại từ 4 features
            # theo đúng công thức risk_score_estimate.
            if "risk_score" in df_combined.columns:
                m = df_combined["risk_score"].isna()
                if m.any():
                    est = (
                        (1 - df_combined.loc[m, "battery_level"] / 100) * 4.0
                        + (df_combined.loc[m, "wind_speed"] / 50) * 3.0
                        + (1 - df_combined.loc[m, "signal_strength"] / 100) * 2.0
                        + (df_combined.loc[m, "temperature"].abs() / 50) * 1.0
                    ).clip(upper=10.0).round(1)
                    df_combined.loc[m, "risk_score"] = est
            if "is_high_risk" in df_combined.columns:
                df_combined["is_high_risk"] = df_combined["is_high_risk"].fillna(
                    (df_combined.get("operation_risk") == "High").astype(int)
                )
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


@st.cache_data
def load_research_json(filename: str):
    """Load file JSON kết quả nghiên cứu (noise_robustness / cost_sensitive)."""
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cost_matrix_for(le) -> "np.ndarray | None":
    """
    Trả về ma trận chi phí C(true, pred) khớp thứ tự classes của label
    encoder, hoặc None nếu chưa có cost_sensitive.json / thứ tự lệch.
    """
    cfg = load_research_json("cost_sensitive.json")
    if not cfg:
        return None
    if list(le.classes_) != list(cfg.get("classes", [])):
        return None
    return np.array(cfg["cost_matrix"], dtype=float)


def cost_sensitive_risk_labels(model, le, X):
    """
    Dự đoán operation_risk theo quy tắc Bayes tối thiểu chi phí kỳ vọng
    (Elkan 2001): j* = argmin_j Σ_i P(i|x)·C(i,j).
    Fallback về argmax nếu chưa có ma trận chi phí.
    Trả về (labels, proba) — proba theo thứ tự le.classes_.
    """
    # Model được fit trên numpy array (không có tên cột) — ép về array để
    # khớp, tránh UserWarning "X has feature names" của sklearn.
    X = np.asarray(X)
    proba = model.predict_proba(X)
    C = _cost_matrix_for(le)
    if C is None:
        idx = proba.argmax(axis=1)
    else:
        idx = (proba @ C).argmin(axis=1)
    return le.inverse_transform(idx), proba


# ─── INPUT / PREDICTION ─────────────────────────────────────────────────────

def build_input_df(**kw) -> pd.DataFrame:
    """Build single-row DataFrame from kwargs matching FEATURES."""
    return pd.DataFrame([{k: kw[k] for k in FEATURES}])


def predict_drone(inp: pd.DataFrame):
    """
    Run all 3 classifiers. Returns (risk, maint, rec, confidence%).

    operation_risk dùng quy tắc Bayes tối thiểu chi phí kỳ vọng (Elkan
    2001) thay vì argmax — ưu tiên không bỏ sót High risk. Ma trận chi
    phí lấy từ Model/cost_sensitive.json (sinh bởi train_model.py).
    """
    models = load_models()
    # .to_numpy(): model fit trên array không tên cột — thứ tự cột đã được
    # FEATURES bảo đảm; tránh UserWarning "X has feature names" của sklearn.
    X = inp[FEATURES].to_numpy()
    rf_r,  le_r  = models["risk"]
    rf_m,  le_m  = models["maint"]
    rf_rc, le_rc = models["rec"]

    risk_labels, proba = cost_sensitive_risk_labels(rf_r, le_r, X)
    risk  = risk_labels[0]
    maint = le_m.inverse_transform(rf_m.predict(X))[0]
    rec   = le_rc.inverse_transform(rf_rc.predict(X))[0]

    # Confidence = xác suất của lớp risk ĐÃ CHỌN (không phải max chung)
    risk_idx = list(le_r.classes_).index(risk)
    conf = round(float(proba[0, risk_idx]) * 100, 1)

    return risk, maint, rec, conf


# ─── PER-MODEL OPERATION RULE PROFILES ──────────────────────────────────────
# Ngưỡng vận hành theo TỪNG DÒNG DRONE, lấy từ thông số kỹ thuật chính hãng
# DJI (dji.com — trang Specs của từng model). Nguyên tắc suy ra ngưỡng:
#   • wind_nofly     = sức gió chịu tối đa theo spec (max wind resistance)
#   • wind_monitor   ≈ 75–80% giới hạn spec (vùng đệm an toàn)
#   • temp_min/max   = dải nhiệt vận hành theo spec (Operating Temperature)
#   • battery_rth    = % pin phải quay về trạm — drone càng nhẹ, dự trữ càng
#                      cao vì gió tiêu hao pin nhanh hơn (nguyên tắc dự phòng
#                      năng lượng; DJI Fly mặc định cảnh báo low-battery 20%)
#   • max_flight_time= thời lượng pin tối đa theo spec
#   • legal_alt      = trần bay 120 m theo điều kiện cấp phép bay phổ biến
#                      cho UAV dân dụng tại Việt Nam
#
# DEFAULT_PROFILE giữ NGUYÊN bộ ngưỡng chung cũ của hệ thống (tương thích
# ngược — thang đo của dataset synthetic rộng hơn spec thật, vd gió 0–50 m/s).

DRONE_NAME_MAP = {
    "Drone_1":  "DJI Mini 3",  "Drone_4": "DJI Mini 3",
    "Drone_7":  "DJI Mini 3",  "Drone_10": "DJI Mini 3",
    "Drone_2":  "DJI Air 3",   "Drone_5": "DJI Air 3",
    "Drone_8":  "DJI Air 3",
    "Drone_3":  "DJI Mavic 3 Pro", "Drone_6": "DJI Mavic 3 Pro",
    "Drone_9":  "DJI Mavic 3 Pro",
}

DEFAULT_PROFILE = dict(
    name="Mặc định (ngưỡng chung)", icon="⚙️", weight_g=None,
    wind_nofly=35.0,  wind_monitor=25.0,
    temp_min=-10.0,   temp_max=45.0,
    temp_monitor_low=-10.0, temp_monitor_high=45.0,   # không có vùng đệm
    battery_critical=20.0,  battery_rth=40.0,
    max_flight_time=None,   flight_time_monitor=40.0,
    max_speed=None,         speed_monitor=75.0,
    legal_alt=None,         alt_monitor=350.0,
    gps_nofly=10.0, gps_monitor=7.0,
    signal_nofly=35.0, signal_monitor=60.0,
    source="Bộ ngưỡng chung của hệ thống (theo thang dataset synthetic)",
)

DRONE_PROFILES = {
    "DJI Mini 3": dict(
        DEFAULT_PROFILE,
        name="DJI Mini 3", icon="🛩️", weight_g=248,
        wind_nofly=10.7,  wind_monitor=8.0,     # Level 5 Beaufort theo spec
        temp_min=-10.0,   temp_max=40.0,
        temp_monitor_low=0.0, temp_monitor_high=35.0,
        battery_critical=15.0, battery_rth=30.0,  # nhẹ nhất → dự trữ cao nhất
        max_flight_time=38.0,  flight_time_monitor=30.0,
        max_speed=16.0,        speed_monitor=13.0,   # m/s, S-mode
        legal_alt=120.0,       alt_monitor=100.0,
        source="DJI Mini 3 — Specs chính hãng (dji.com)",
    ),
    "DJI Air 3": dict(
        DEFAULT_PROFILE,
        name="DJI Air 3", icon="✈️", weight_g=720,
        wind_nofly=12.0,  wind_monitor=9.5,
        temp_min=-10.0,   temp_max=40.0,
        temp_monitor_low=0.0, temp_monitor_high=35.0,
        battery_critical=12.0, battery_rth=25.0,
        max_flight_time=46.0,  flight_time_monitor=37.0,
        max_speed=21.0,        speed_monitor=17.0,
        legal_alt=120.0,       alt_monitor=100.0,
        source="DJI Air 3 — Specs chính hãng (dji.com)",
    ),
    "DJI Mavic 3 Pro": dict(
        DEFAULT_PROFILE,
        name="DJI Mavic 3 Pro", icon="🚁", weight_g=958,
        wind_nofly=12.0,  wind_monitor=9.5,
        temp_min=-10.0,   temp_max=40.0,
        temp_monitor_low=0.0, temp_monitor_high=35.0,
        battery_critical=10.0, battery_rth=20.0,  # pin lớn, chịu gió tốt nhất
        max_flight_time=43.0,  flight_time_monitor=34.0,
        max_speed=21.0,        speed_monitor=17.0,
        legal_alt=120.0,       alt_monitor=100.0,
        source="DJI Mavic 3 Pro — Specs chính hãng (dji.com)",
    ),
}


def get_profile(key: str) -> dict:
    """
    Lấy hồ sơ quy tắc theo tên dòng drone HOẶC drone_id (Drone_1..10).
    Không khớp → DEFAULT_PROFILE.
    """
    if key in DRONE_PROFILES:
        return DRONE_PROFILES[key]
    model = DRONE_NAME_MAP.get(str(key))
    return DRONE_PROFILES.get(model, DEFAULT_PROFILE)


# ─── DECISION RULES (README Section 11) ─────────────────────────────────────

def flight_decision(battery, signal, wind, gps, flight_time,
                    temp, alt, speed, risk, maint, profile=None):
    """
    Rule-based flight status theo hồ sơ quy tắc của từng dòng drone.
    profile=None → DEFAULT_PROFILE (giữ nguyên hành vi cũ).
    Returns (label, reason, level).
    """
    p  = profile or DEFAULT_PROFILE
    ml = str(maint).lower()

    # Cấm Bay — ràng buộc an toàn cứng theo spec hãng
    rc = []
    if battery < p["battery_critical"]:
        rc.append(f"pin dưới {p['battery_critical']:.0f}% (ngưỡng khẩn của {p['name']})")
    if signal < p["signal_nofly"]:
        rc.append(f"tín hiệu dưới {p['signal_nofly']:.0f}%")
    if wind > p["wind_nofly"]:
        rc.append(f"gió trên {p['wind_nofly']:.1f} m/s (sức gió chịu tối đa theo spec)")
    if gps > p["gps_nofly"]:
        rc.append(f"GPS accuracy trên {p['gps_nofly']:.0f} m")
    if temp > p["temp_max"]:
        rc.append(f"nhiệt độ trên {p['temp_max']:.0f}°C (dải vận hành theo spec)")
    if temp < p["temp_min"]:
        rc.append(f"nhiệt độ dưới {p['temp_min']:.0f}°C (dải vận hành theo spec)")
    if p["max_flight_time"] and flight_time > p["max_flight_time"]:
        rc.append(f"vượt thời lượng pin tối đa {p['max_flight_time']:.0f} phút")
    if p["max_speed"] and speed > p["max_speed"]:
        rc.append(f"vượt tốc độ tối đa {p['max_speed']:.0f} m/s theo spec")
    if p["legal_alt"] and alt > p["legal_alt"]:
        rc.append(f"vượt trần bay cấp phép {p['legal_alt']:.0f} m")
    if rc:
        return ("Cấm Bay",
                "Điều kiện nguy hiểm: " + ", ".join(rc) + ".",
                "danger")

    if any(kw in ml for kw in ("maintenance required", "inspection recommended",
                               "inspect", "urgent")):
        return ("Yêu Cầu Bảo Trì",
                "Drone cần kiểm tra hoặc bảo trì trước chuyến bay tiếp theo.",
                "danger")

    # Quay Về Trạm — pin chạm ngưỡng RTH của dòng máy, hoặc ML báo High
    if battery < p["battery_rth"]:
        return ("Quay Về Trạm",
                f"Pin {battery:.0f}% dưới ngưỡng quay về trạm "
                f"{p['battery_rth']:.0f}% của {p['name']}.",
                "warning")
    if risk == "High":
        return ("Quay Về Trạm",
                "Rủi ro vận hành cao. Drone nên quay về trạm để kiểm tra.",
                "warning")

    rm = []
    if flight_time > p["flight_time_monitor"]:
        rm.append(f"thời gian bay trên {p['flight_time_monitor']:.0f} phút")
    if alt > p["alt_monitor"]:
        rm.append(f"độ cao trên {p['alt_monitor']:.0f} m")
    if speed > p["speed_monitor"]:
        rm.append(f"tốc độ trên {p['speed_monitor']:.0f}")
    if signal < p["signal_monitor"]:
        rm.append(f"tín hiệu dưới {p['signal_monitor']:.0f}%")
    if wind > p["wind_monitor"]:
        rm.append(f"gió trên {p['wind_monitor']:.1f} m/s")
    if gps > p["gps_monitor"]:
        rm.append(f"GPS accuracy trên {p['gps_monitor']:.0f} m")
    if temp < p["temp_monitor_low"] or temp > p["temp_monitor_high"]:
        rm.append(f"nhiệt độ ngoài vùng thoải mái "
                  f"{p['temp_monitor_low']:.0f}–{p['temp_monitor_high']:.0f}°C")
    if rm:
        return ("Bay Kèm Giám Sát",
                "Cần theo dõi sát: " + ", ".join(rm) + ".",
                "warning")

    return ("Đủ Điều Kiện Bay", "Drone đủ điều kiện vận hành bình thường.", "success")


def battery_status(level: float, profile=None):
    """(label, alert_level) — ngưỡng theo hồ sơ dòng drone."""
    p = profile or DEFAULT_PROFILE
    if level > p["battery_rth"]:      return "Tốt", "success"
    if level > p["battery_critical"]: return "Trung Bình", "warning"
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

def save_custom(drone_id, inp, risk, maint, rec, bat_label, status, reason,
                drone_model=None, extra_cols=None):
    """
    Lưu bản ghi vào custom_drone_data.csv theo đúng 49 cột chuẩn.
    extra_cols: dict cột chuẩn bổ sung (vd lat/long, distance_flown_km,
    operation_notes...) — dùng cho telemetry Phiên bay trực tiếp.
    """
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
    if drone_model:
        row["drone_model"] = drone_model
        row["manufacturer"] = "DJI" if str(drone_model).startswith("DJI") else pd.NA
    if extra_cols:
        for k, v in extra_cols.items():
            if k in STANDARD_COLUMNS:
                row[k] = v

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