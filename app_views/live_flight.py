"""
app_views/live_flight.py — Phiên bay trực tiếp (Live Flight Session)
=====================================================================
Mô phỏng luồng telemetry vận hành thật đổ về DSS:

  • Bấm "Bắt đầu phiên bay" → mỗi chu kỳ (3 phút thật / 10 giây demo)
    hệ thống nhận 1 DÒNG DỮ LIỆU ĐÚNG FORMAT 49 CỘT của dataset
    (sinh bởi simulator suy biến trạng thái) và phân tích ngay:
        ML (predict_drone, cost-sensitive)  +  Rule (flight_decision
        theo hồ sơ dòng drone)  +  Tầng TREND (EWMA + dự báo pin)
    → verdict: BAY TIẾP / GIÁM SÁT / QUAY VỀ / HẠ CÁNH NGAY.

  • Mỗi tick được lưu qua save_custom (49 cột chuẩn → gộp thẳng vào
    Dashboard / Phân tích drone) + bản sao live_session_log.csv.

Kỹ thuật: st.fragment(run_every=...) làm "nhịp tim" — chỉ fragment
tự chạy lại, không reload cả trang. KHÔNG sửa logic nào của core.

Tầng trend (tính chất nghiên cứu):
  • EWMA control chart trên risk score (Roberts 1959, Technometrics)
    phát hiện drift rủi ro tăng dần mà snapshot đơn lẻ chưa thấy.
  • Dự báo pin tick kế tiếp (ngoại suy tuyến tính) → khuyến nghị quay
    về TRƯỚC khi vi phạm ngưỡng RTH (proactive thay vì reactive).
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    BASE_DIR, COLORS, FEATURES,
    build_input_df, predict_drone, flight_decision, battery_status,
    risk_score_estimate, risk_to_level, translate, save_custom, style_chart,
    startup_load_or_stop, get_profile, DRONE_PROFILES, DEFAULT_PROFILE,
    DRONE_NAME_MAP,
)
from ui import (
    render_top_nav, render_page_title, render_section_label,
    render_banner, render_result_badge,
)

LIVE_LOG_PATH = BASE_DIR / "Data" / "live_session_log.csv"

TICK_MODES = {
    "⚡ Demo — 10 giây/tick (1 tick ≈ 3 phút mô phỏng)": 10,
    "🕒 Thực tế — 3 phút/tick": 180,
}
SIM_DT_MIN   = 3.0    # moi tick tuong duong 3 phut bay
EWMA_LAMBDA  = 0.30   # trong so EWMA (Roberts 1959)
EWMA_LIMIT   = 6.0    # nguong canh bao tren thang risk score 0-10


# ─── SESSION STATE ──────────────────────────────────────────────────────────

def _init_lf():
    if "lf" not in st.session_state:
        st.session_state.lf = {"status": "IDLE"}
    return st.session_state.lf


def _reset_lf():
    st.session_state.lf = {"status": "IDLE"}


# ─── SIMULATOR — suy biến trạng thái mỗi tick 3 phút ────────────────────────

def _sim_step(s: dict, base: dict, rng: np.random.Generator) -> dict:
    """
    Sinh trạng thái tick kế tiếp từ trạng thái hiện tại (random walk có
    quán tính). Pin tụt nhanh hơn khi gió/tốc độ lớn; gió theo AR(1).
    """
    n = dict(s)

    # Gió: AR(1) quanh gió nền — có quán tính, không nhảy loạn
    n["wind_speed"] = float(np.clip(
        0.70 * s["wind_speed"] + 0.30 * base["wind_mean"] + rng.normal(0, 1.2),
        0, 50))

    # Pin: hao 0.8%/phút nền + thành phần gió + tốc độ
    drain_per_min = 0.8 + 0.045 * n["wind_speed"] + 0.02 * s["speed"] \
                    + rng.normal(0, 0.05)
    n["battery_level"] = float(np.clip(
        s["battery_level"] - max(0.3, drain_per_min) * SIM_DT_MIN, 0, 100))

    n["flight_time"] = float(s["flight_time"] + SIM_DT_MIN)

    # Tín hiệu: kéo về nền, suy hao nhẹ theo độ cao
    sig_base = 92.0 - s["altitude"] / 25.0
    n["signal_strength"] = float(np.clip(
        0.75 * s["signal_strength"] + 0.25 * sig_base + rng.normal(0, 2.5),
        0, 100))

    n["gps_accuracy"] = float(np.clip(
        1.5 + 0.06 * n["wind_speed"] + rng.normal(0, 0.3), 0.5, 10))
    n["temperature"]  = float(np.clip(s["temperature"] + rng.normal(0, 0.4), -40, 50))
    n["humidity"]     = float(np.clip(s["humidity"] + rng.normal(0, 1.5), 0, 100))
    n["pressure"]     = float(np.clip(s["pressure"] + rng.normal(0, 0.6), 950, 1050))
    n["altitude"]     = float(np.clip(
        0.85 * s["altitude"] + 0.15 * base["cruise_alt"] + rng.normal(0, 4), 0, 500))
    # Toc do: autopilot giu quanh cruise — nhieu nho va kep bien do ±15%
    # (drone khong tu vuot toc do; muon test rule toc do thi dat cruise cao)
    n["speed"]        = float(np.clip(
        0.85 * s["speed"] + 0.15 * base["cruise_speed"] + rng.normal(0, 0.6),
        base["cruise_speed"] * 0.85, base["cruise_speed"] * 1.15))

    # Vị trí: bay theo heading (random walk ±12°), bước = speed × 3 phút
    n["heading"] = float((s["heading"] + rng.normal(0, 12)) % 360)
    step_km = s["speed"] * SIM_DT_MIN * 60 / 1000.0
    rad = np.deg2rad(n["heading"])
    n["latitude"]  = float(s["latitude"]  + step_km * np.cos(rad) / 111.32)
    n["longitude"] = float(s["longitude"] + step_km * np.sin(rad)
                           / (111.32 * np.cos(np.deg2rad(s["latitude"]))))
    n["distance_km"] = float(s["distance_km"] + step_km)
    return n


# ─── QUYẾT ĐỊNH MỖI TICK — tái dùng pipeline + tầng trend ───────────────────

def _analyze_tick(lf: dict) -> dict:
    """Chạy ML + rule + trend cho trạng thái hiện tại. Trả về record tick."""
    s, p = lf["state"], lf["profile"]

    inp = build_input_df(**{k: s[k] for k in FEATURES})
    risk, maint, rec, conf = predict_drone(inp)                      # tai dung
    d_label, d_reason, d_lv = flight_decision(                        # tai dung
        s["battery_level"], s["signal_strength"], s["wind_speed"],
        s["gps_accuracy"], s["flight_time"], s["temperature"],
        s["altitude"], s["speed"], risk, maint, profile=p,
    )

    # ── Tầng trend ───────────────────────────────────────────────────────
    score = risk_score_estimate(s["battery_level"], s["wind_speed"],
                                s["signal_strength"], s["temperature"])
    prev_ewma  = lf.get("ewma")
    ewma = score if prev_ewma is None else \
        round(EWMA_LAMBDA * score + (1 - EWMA_LAMBDA) * prev_ewma, 2)
    lf["ewma"] = ewma

    hist = lf["history"]
    k = min(3, len(hist))
    if k >= 1:
        slope = (s["battery_level"] - hist[-k]["battery"]) / k
        forecast = round(s["battery_level"] + slope, 1)
    else:
        forecast = s["battery_level"]

    # ── Verdict + leo thang ──────────────────────────────────────────────
    trend_note = ""
    if d_lv == "danger":                       # Cấm Bay / Yêu Cầu Bảo Trì
        verdict, v_lv, new_status = "🛑 HẠ CÁNH NGAY", "danger", "GROUNDED"
        lf["monitor_streak"] = 0
    elif d_label == "Quay Về Trạm":
        verdict, v_lv, new_status = "🔙 KẾT THÚC — QUAY VỀ TRẠM", "warning", "RETURNING"
        lf["monitor_streak"] = 0
    elif d_label == "Bay Kèm Giám Sát":
        lf["monitor_streak"] = lf.get("monitor_streak", 0) + 1
        if lf["monitor_streak"] >= 2:
            verdict, v_lv, new_status = "🔙 QUAY VỀ TRẠM", "warning", "RETURNING"
            trend_note = "2 chu kỳ cảnh báo liên tiếp — leo thang quay về."
        elif forecast < p["battery_rth"]:
            verdict, v_lv, new_status = "🔙 QUAY VỀ TRẠM", "warning", "RETURNING"
            trend_note = (f"Dự báo pin tick kế {forecast:.0f}% < ngưỡng RTH "
                          f"{p['battery_rth']:.0f}% — quay về chủ động.")
        else:
            verdict, v_lv, new_status = "👁️ BAY TIẾP — GIÁM SÁT", "warning", "FLYING"
    else:                                       # Đủ Điều Kiện Bay
        lf["monitor_streak"] = 0
        if forecast < p["battery_rth"]:
            verdict, v_lv, new_status = "🔙 QUAY VỀ TRẠM", "warning", "RETURNING"
            trend_note = (f"Dự báo pin tick kế {forecast:.0f}% < ngưỡng RTH "
                          f"{p['battery_rth']:.0f}% — quay về chủ động (proactive).")
        elif ewma > EWMA_LIMIT:
            verdict, v_lv, new_status = "👁️ BAY TIẾP — GIÁM SÁT", "warning", "FLYING"
            trend_note = f"EWMA risk {ewma:.1f} vượt giới hạn {EWMA_LIMIT} — drift rủi ro tăng."
        else:
            verdict, v_lv, new_status = "✅ BAY TIẾP", "success", "FLYING"

    return dict(
        risk=risk, maint=maint, rec=rec, conf=conf, score=score, ewma=ewma,
        forecast=forecast, d_label=d_label, d_reason=d_reason,
        verdict=verdict, v_lv=v_lv, new_status=new_status,
        trend_note=trend_note, inp=inp,
    )


def _advance_tick(lf: dict):
    """1 nhịp 3 phút: sinh dữ liệu → phân tích → lưu 49 cột → cập nhật phiên."""
    rng = lf["rng"]
    lf["state"] = _sim_step(lf["state"], lf["base"], rng)
    lf["tick"] += 1
    lf["last_tick_ts"] = time.time()
    s = lf["state"]

    event = lf.pop("pending_event", "")
    a = _analyze_tick(lf)

    # Lưu dòng 49 cột chuẩn — tái dùng save_custom → gộp vào toàn hệ thống
    bat_label, _ = battery_status(s["battery_level"], profile=lf["profile"])
    notes = (f"{lf['session_id']} tick {lf['tick']} — {a['verdict']}"
             + (f" | {a['trend_note']}" if a["trend_note"] else "")
             + (f" | Sự cố: {event}" if event else ""))
    saved = save_custom(
        lf["drone_id"], a["inp"], a["risk"], a["maint"], a["rec"],
        bat_label, a["d_label"], a["d_reason"],
        drone_model=lf["profile"]["name"] if lf["profile"] is not DEFAULT_PROFILE else None,
        extra_cols={
            "latitude":  round(s["latitude"], 6),
            "longitude": round(s["longitude"], 6),
            "heading":   round(s["heading"], 1),
            "distance_flown_km": round(s["distance_km"], 2),
            "application": lf["mission"],
            "operator_id": "OP_LIVE",
            "operation_notes": notes,
        },
    )
    # Bản sao log riêng của phiên (cùng schema 49 cột)
    try:
        row = saved.tail(1)
        header = not LIVE_LOG_PATH.exists() or LIVE_LOG_PATH.stat().st_size == 0
        row.to_csv(LIVE_LOG_PATH, mode="a", header=header,
                   index=False, encoding="utf-8-sig")
    except Exception:
        pass

    lf["history"].append(dict(
        tick=lf["tick"], time=pd.Timestamp.now().strftime("%H:%M:%S"),
        battery=round(s["battery_level"], 1), wind=round(s["wind_speed"], 1),
        signal=round(s["signal_strength"], 1), temp=round(s["temperature"], 1),
        gps=round(s["gps_accuracy"], 2), alt=round(s["altitude"], 0),
        speed=round(s["speed"], 1), flight_time=round(s["flight_time"], 0),
        lat=s["latitude"], lon=s["longitude"], dist=round(s["distance_km"], 2),
        risk=a["risk"], maint=a["maint"], decision=a["d_label"],
        verdict=a["verdict"], v_lv=a["v_lv"], reason=a["d_reason"],
        trend=a["trend_note"], score=a["score"], ewma=a["ewma"],
        forecast=a["forecast"], conf=a["conf"], event=event,
    ))

    if a["new_status"] != "FLYING":
        lf["status"]    = a["new_status"]
        lf["end_reason"] = (a["verdict"] + " — " + a["d_reason"]
                            + (" " + a["trend_note"] if a["trend_note"] else ""))


# ─── UI: MÀN HÌNH CẤU HÌNH (IDLE) ───────────────────────────────────────────

def _render_setup(lf: dict):
    render_banner(
        "Cấu hình phiên bay → bấm **Bắt đầu**. Mỗi chu kỳ, hệ thống nhận "
        "**1 dòng telemetry đúng format 49 cột của dataset** và phân tích "
        "bằng chính pipeline hiện có (ML cost-sensitive + rule theo dòng "
        "drone + tầng xu hướng EWMA).",
        "info",
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_section_label("Thiết bị & nhiệm vụ")
        fleet = sorted(DRONE_NAME_MAP.keys(), key=lambda x: int(x.split("_")[1]))
        drone_id = st.selectbox("Drone thực hiện phiên bay",
                                fleet + ["Drone_LIVE_Custom"], key="lf_drone")
        if drone_id in DRONE_NAME_MAP:
            profile = get_profile(drone_id)
            st.caption(f"Dòng máy: **{profile['name']}** {profile['icon']} — "
                       f"tự áp hồ sơ quy tắc (RTH {profile['battery_rth']:.0f}%, "
                       f"gió tối đa {profile['wind_nofly']:.1f} m/s).")
        else:
            model = st.selectbox("Chọn dòng máy",
                                 [DEFAULT_PROFILE["name"]] + list(DRONE_PROFILES),
                                 key="lf_model")
            profile = DRONE_PROFILES.get(model, DEFAULT_PROFILE)
        mission = st.text_input("Nhiệm vụ (application)",
                                "Live Test Flight", key="lf_mission")
        mode = st.radio("Chu kỳ telemetry", list(TICK_MODES), key="lf_mode")

    with c2:
        render_section_label("Điều kiện xuất phát")
        battery = st.slider("🔋 Pin (%)",            20.0, 100.0, 95.0, 1.0, key="lf_b")
        wind    = st.slider("💨 Gió nền (m/s)",       0.0,  20.0,  5.0, 0.5, key="lf_w")
        temp    = st.slider("🌡️ Nhiệt độ (°C)",     -15.0,  45.0, 28.0, 0.5, key="lf_t")
        alt     = st.slider("🏔️ Độ cao hành trình (m)", 20.0, 150.0, 80.0, 5.0, key="lf_a")
        speed   = st.slider("🚀 Tốc độ hành trình (m/s)", 5.0, 25.0, 12.0, 0.5, key="lf_s")

    if st.button("🛫 Bắt đầu phiên bay", type="primary", key="lf_start"):
        state = dict(
            battery_level=battery, flight_time=0.0, signal_strength=95.0,
            temperature=temp, wind_speed=wind, gps_accuracy=1.2,
            altitude=alt, speed=speed, humidity=60.0, pressure=1010.0,
            latitude=21.0278, longitude=105.8342, heading=float(np.random.uniform(0, 360)),
            distance_km=0.0,
        )
        # Kiểm tra tiền phiên bằng chính rule engine (chưa cần ML)
        lbl, why, lv = flight_decision(
            battery, 95.0, wind, 1.2, 0.0, temp, alt, speed,
            "Low", "Monitor", profile=profile)
        if lv == "danger":
            render_banner(f"⛔ Không đủ điều kiện cất cánh: {why}", "danger")
            return
        st.session_state.lf = dict(
            status="FLYING", session_id=pd.Timestamp.now().strftime("LIVE_%Y%m%d_%H%M%S"),
            drone_id=drone_id, profile=profile, mission=mission,
            interval=TICK_MODES[mode], tick=0, last_tick_ts=time.time(),
            state=state, base=dict(wind_mean=wind, cruise_alt=alt, cruise_speed=speed),
            history=[], monitor_streak=0, ewma=None, end_reason=None,
            rng=np.random.default_rng(),
        )
        st.rerun()


# ─── UI: MÀN HÌNH BAY (FLYING — bên trong fragment) ─────────────────────────

def _render_live(lf: dict):
    s, p = lf["state"], lf["profile"]
    hist = lf["history"]
    last = hist[-1] if hist else None

    if last:
        render_banner(f"**{last['verdict']}** — {last['reason']}"
                      + (f"  ·  _{last['trend']}_" if last["trend"] else ""),
                      last["v_lv"])
    else:
        render_banner(
            f"🛫 Phiên **{lf['session_id']}** đã bắt đầu — dòng telemetry đầu "
            f"tiên sẽ về sau {lf['interval']}s (1 tick = {SIM_DT_MIN:.0f} phút bay).",
            "info")

    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    rth = p["battery_rth"]
    with m1:
        st.metric("🔋 Pin", f"{s['battery_level']:.1f}%",
                  delta=f"RTH tại {rth:.0f}%", delta_color="off")
    with m2:
        st.metric("💨 Gió", f"{s['wind_speed']:.1f} m/s",
                  delta=f"cấm bay > {p['wind_nofly']:.1f}", delta_color="off")
    with m3:
        st.metric("⏱️ Đã bay", f"{s['flight_time']:.0f} phút",
                  delta=f"tick {lf['tick']}", delta_color="off")
    with m4:
        st.metric("📈 EWMA risk", f"{(lf['ewma'] or 0):.1f}/10",
                  delta=f"giới hạn {EWMA_LIMIT}", delta_color="off")
    with m5:
        st.metric("🗺️ Quãng đường", f"{s['distance_km']:.1f} km",
                  delta=f"{lf['drone_id']}", delta_color="off")

    if hist:
        df = pd.DataFrame(hist)
        ch1, ch2 = st.columns([3, 2], gap="large")
        with ch1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["tick"], y=df["battery"], name="Pin (%)",
                                     mode="lines+markers",
                                     line=dict(color=COLORS["accent"], width=2.5)))
            fig.add_trace(go.Scatter(x=df["tick"], y=df["wind"], name="Gió (m/s)",
                                     mode="lines+markers",
                                     line=dict(color=COLORS["blue"], width=2)))
            fig.add_trace(go.Scatter(x=df["tick"], y=df["ewma"] * 10, name="EWMA risk ×10",
                                     mode="lines", line=dict(color=COLORS["danger"],
                                                             width=2, dash="dot")))
            fig.add_hline(y=rth, line=dict(color=COLORS["warning"], dash="dash"),
                          annotation_text=f"RTH {rth:.0f}%")
            fig.update_layout(xaxis_title="Tick (3 phút/tick)",
                              legend=dict(orientation="h", y=1.15, x=0.5,
                                          xanchor="center"))
            st.plotly_chart(style_chart(fig, 320), use_container_width=True,
                            config={"displayModeBar": False})
        with ch2:
            st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"})[
                       ["latitude", "longitude"]], size=30, zoom=11)

    render_section_label("Tiêm sự cố (kiểm thử phản ứng hệ thống)")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("💥 Gió giật +12 m/s", key="lf_ev_wind"):
        s["wind_speed"] = min(50.0, s["wind_speed"] + 12)
        lf["pending_event"] = "Gió giật +12 m/s"
    if b2.button("📡 Suy hao tín hiệu −30%", key="lf_ev_sig"):
        s["signal_strength"] = max(0.0, s["signal_strength"] - 30)
        lf["pending_event"] = "Suy hao tín hiệu -30%"
    if b3.button("🔋 Sụt pin −15%", key="lf_ev_bat"):
        s["battery_level"] = max(0.0, s["battery_level"] - 15)
        lf["pending_event"] = "Sụt pin -15%"
    if b4.button("🛬 Kết thúc phiên", key="lf_end", type="primary"):
        lf["status"] = "LANDED"
        lf["end_reason"] = "Người vận hành chủ động kết thúc phiên."
        st.rerun(scope="app")

    if hist:
        render_section_label("Nhật ký telemetry (mới nhất trên cùng)")
        show = pd.DataFrame(hist)[::-1][[
            "tick", "time", "battery", "wind", "signal", "gps",
            "risk", "decision", "verdict", "forecast", "event"]]
        show.columns = ["Tick", "Giờ", "Pin %", "Gió", "Tín hiệu", "GPS",
                        "ML Risk", "Rule", "Verdict", "Dự báo pin", "Sự cố"]
        st.dataframe(show.head(10), use_container_width=True, hide_index=True)


# ─── UI: MÀN HÌNH TỔNG KẾT (RETURNING / GROUNDED / LANDED) ──────────────────

def _render_summary(lf: dict):
    icon = {"RETURNING": "🔙", "GROUNDED": "🛑", "LANDED": "🛬"}.get(lf["status"], "✅")
    lv   = {"RETURNING": "warning", "GROUNDED": "danger"}.get(lf["status"], "success")
    render_banner(f"{icon} **Phiên {lf['session_id']} đã kết thúc** — "
                  f"{lf.get('end_reason', '')}", lv)

    hist = lf["history"]
    if hist:
        df = pd.DataFrame(hist)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Tổng tick", f"{lf['tick']}",
                           delta=f"≈ {lf['tick']*SIM_DT_MIN:.0f} phút bay",
                           delta_color="off")
        with c2: st.metric("Pin cuối", f"{df['battery'].iloc[-1]:.1f}%",
                           delta=f"xuất phát {df['battery'].iloc[0]:.1f}%",
                           delta_color="off")
        with c3: st.metric("Quãng đường", f"{df['dist'].iloc[-1]:.1f} km",
                           delta=lf["drone_id"], delta_color="off")
        with c4: st.metric("EWMA risk cuối", f"{df['ewma'].iloc[-1]:.1f}/10",
                           delta=f"đỉnh {df['ewma'].max():.1f}", delta_color="off")

        render_section_label("Toàn bộ nhật ký phiên")
        st.dataframe(df[::-1].drop(columns=["v_lv", "lat", "lon"]),
                     use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Tải nhật ký phiên (.csv)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{lf['session_id']}.csv", mime="text/csv")
        st.caption(
            "💾 Từng tick cũng đã được lưu **đúng 49 cột chuẩn** vào "
            "`Data/custom_drone_data.csv` (hiển thị ngay trong Dashboard & "
            "Phân tích drone) và `Data/live_session_log.csv`.")

    if st.button("🔄 Phiên bay mới", type="primary", key="lf_new"):
        _reset_lf()
        st.rerun()


# ─── RENDER (entry) ─────────────────────────────────────────────────────────

def render():
    render_top_nav()
    startup_load_or_stop()

    render_page_title(
        "Phiên bay trực tiếp",
        "Telemetry theo chu kỳ 3 phút → phân tích tức thời: bay tiếp hay dừng?  ·  DSS301",
    )

    lf = _init_lf()

    if lf["status"] == "IDLE":
        _render_setup(lf)
    elif lf["status"] == "FLYING":
        # Fragment tự chạy lại theo chu kỳ — "nhịp tim" của phiên bay.
        # Guard thời gian: rerun do bấm nút sẽ KHÔNG tạo tick sớm.
        @st.fragment(run_every=lf["interval"])
        def _heartbeat():
            _lf = st.session_state.lf
            if _lf["status"] != "FLYING":
                st.rerun(scope="app")
            if time.time() - _lf["last_tick_ts"] >= _lf["interval"] - 0.5:
                _advance_tick(_lf)
                if _lf["status"] != "FLYING":
                    st.rerun(scope="app")
            _render_live(_lf)
        _heartbeat()
    else:
        _render_summary(lf)
