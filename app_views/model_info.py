"""
app_views/model_info.py — Model Info
===================================
3 tab:
  1. Tổng quan       — accuracy 3 model, ranking, side-by-side comparison
  2. Chi tiết model  — chọn model → xem F1 / confusion matrix riêng
  3. Feature impact  — feature importance averaged
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    load_metrics, load_feature_importance, load_research_json, style_chart,
    startup_load_or_stop,
)
from ui import (
    render_page_title, render_section_label, render_banner,
    render_class_f1_table, render_confusion_matrix_heatmap,
    render_feature_importance_chart,
    render_callout, render_compare_bars,
    render_decision_card,
)

# Màu thuật toán (theo design tokens)
ALGO_COLORS = {
    "Random Forest":       "#4f63d2",   # primary
    "Decision Tree":       "#0284c7",   # info
    "Logistic Regression": "#94a3b8",   # neutral
}

# Tác vụ → (nhãn tiếng Việt, đối tượng sử dụng, câu hỏi quyết định)
TASK_META = {
    "operation_risk":     ("Rủi ro vận hành", "Điều phối viên",
                           "Chuyến bay này rủi ro tới mức nào?"),
    "maintenance_action": ("Hành động bảo trì", "Kỹ thuật bảo trì",
                           "Cần làm gì với thiết bị này?"),
    "recommendation":     ("Khuyến nghị", "Phi công",
                           "Phi công nên làm gì tiếp theo?"),
}


MODEL_DEFS = [
    ("operation_risk_model",     "operation_risk",     "Dự đoán mức rủi ro tổng thể (High / Medium / Low)"),
    ("recommendation_model",     "recommendation",     "Khuyến nghị vận hành cuối cùng cho phi công"),
    ("maintenance_action_model", "maintenance_action", "Khuyến nghị hành động bảo trì (Monitor / Inspect / ...)"),
]

COMPARISON_MODELS = ["Random Forest", "Decision Tree", "Logistic Regression"]


def render():
    startup_load_or_stop()

    render_page_title(
        "Model Info",
        "So sánh 3 thuật toán ML (Random Forest · Decision Tree · Logistic Regression) "
        "trên 3 tác vụ quyết định.",
    )

    all_metrics = {}
    for mname, _, _ in MODEL_DEFS:
        m = load_metrics(mname)
        if m is not None:
            all_metrics[mname] = m

    if not all_metrics:
        render_banner(
            "Chưa có file _metrics.json. Chạy `python train_model.py` trước.",
            "warning",
        )
        return

    tab_overview, tab_detail, tab_impact = st.tabs([
        "📊  Tổng quan & So sánh",
        "🔍  Chi tiết theo model",
        "🎯  Feature Impact",
    ])

    with tab_overview:
        _render_overview_tab(all_metrics)

    with tab_detail:
        _render_detail_tab(all_metrics)

    with tab_impact:
        _render_impact_tab(all_metrics)


# ─── TAB 1: OVERVIEW & COMPARISON ───────────────────────────────────────────

def _collect_algo_matrix(all_metrics: dict):
    """
    Gom số liệu 3 thuật toán × 3 tác vụ từ *_metrics.json (dữ liệu THẬT).
    Trả về list dict: {task, task_vn, algo, accuracy, precision, recall, f1}
    """
    out = []
    for mname, task, _ in MODEL_DEFS:
        m = all_metrics.get(mname)
        if m is None:
            continue
        task_vn = TASK_META.get(task, (task, "", ""))[0]
        out.append(dict(task=task, task_vn=task_vn, algo="Random Forest",
                        accuracy=m.get("accuracy", 0), precision=m.get("precision", 0),
                        recall=m.get("recall", 0), f1=m.get("f1", 0)))
        for algo, v in (m.get("comparison") or {}).items():
            out.append(dict(task=task, task_vn=task_vn, algo=algo,
                            accuracy=v.get("accuracy", 0), precision=v.get("precision", 0),
                            recall=v.get("recall", 0), f1=v.get("f1", 0)))
    return out


def _render_overview_tab(all_metrics: dict):
    matrix = _collect_algo_matrix(all_metrics)
    if not matrix:
        render_banner(
            "Chưa có dữ liệu so sánh thuật toán. "
            "Chạy `python train_model.py` để cập nhật metrics đầy đủ.",
            "warning",
        )
        return

    df = pd.DataFrame(matrix)

    # ── 1. TỈ LỆ TRUNG BÌNH (Box đen, canh đều) ─────────
    rf_avg = df[df.algo == "Random Forest"]["accuracy"].mean()
    dt_avg = df[df.algo == "Decision Tree"]["accuracy"].mean()
    lr_min = df[df.algo == "Logistic Regression"]["accuracy"].min()
    lr_max = df[df.algo == "Logistic Regression"]["accuracy"].max()

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1b1e2c 0%, #11131f 100%); 
                    border-radius: 12px; padding: 24px 32px; 
                    display: flex; justify-content: space-between; 
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <div style="flex: 1;">
                <div style="font-family:'JetBrains Mono',monospace; font-size: 30px; font-weight: 700; color: #ffffff; line-height: 1.2;">{rf_avg:.1%}</div>
                <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px;">Random Forest — accuracy TB</div>
            </div>
            <div style="flex: 1;">
                <div style="font-family:'JetBrains Mono',monospace; font-size: 30px; font-weight: 700; color: #ffffff; line-height: 1.2;">{dt_avg:.1%}</div>
                <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px;">Decision Tree — accuracy TB</div>
            </div>
            <div style="flex: 1;">
                <div style="font-family:'JetBrains Mono',monospace; font-size: 30px; font-weight: 700; color: #ffffff; line-height: 1.2;">{lr_min:.0%}–{lr_max:.0%}</div>
                <div style="font-size: 12.5px; color: #94a3b8; margin-top: 4px;">Logistic Regression</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 2. BIỂU ĐỒ CỘT NHÓM — 3 tác vụ × 3 thuật toán ───────────────────
    render_section_label("So sánh 3 thuật toán ML trên 3 tác vụ quyết định")

    fig = go.Figure()
    tasks_vn = [TASK_META[t][0] for t, _, _ in
                [(t, None, None) for _, t, _ in MODEL_DEFS]]
    for algo in ["Random Forest", "Decision Tree", "Logistic Regression"]:
        sub = df[df.algo == algo]
        sub = sub.set_index("task_vn").reindex(tasks_vn).reset_index()
        fig.add_trace(go.Bar(
            name=algo,
            x=sub["task_vn"], y=sub["accuracy"],
            marker=dict(color=ALGO_COLORS[algo],
                        line=dict(width=0)),
            text=[f"{v:.1%}" for v in sub["accuracy"]],
            textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono"),
            hovertemplate="<b>%{x}</b><br>" + algo + ": %{y:.2%}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", bargap=0.32, bargroupgap=0.08,
        yaxis=dict(tickformat=".0%", range=[0, 1.12], title=None),
        xaxis=dict(title=None),
        legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center"),
    )
    st.plotly_chart(style_chart(fig, 380), width="stretch",
                    config={"displayModeBar": False})

    # Nhãn đối tượng dưới mỗi nhóm
    role_cols = st.columns(len(MODEL_DEFS))
    for col, (_, task, _) in zip(role_cols, MODEL_DEFS):
        vn, role, _q = TASK_META[task]
        with col:
            st.markdown(
                f"<p style='text-align:center;margin:0;font-size:12.5px;"
                f"color:var(--text-muted);'>{vn} → <b style='color:var(--text-secondary)'>"
                f"{role}</b></p>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    render_callout(
        "Cách đọc kết quả",
        "Cột càng cao = dự đoán càng đúng. Điều đáng chú ý không phải Random Forest "
        "thắng sát nút Decision Tree, mà là <b>khoảng cách lớn của Logistic "
        "Regression</b>: mô hình tuyến tính chỉ vẽ được ranh giới thẳng, trong khi "
        "ranh giới thật là cong (pin yếu chỉ nguy hiểm KHI gió mạnh). Con số thấp "
        "của LR vì vậy là <b>bằng chứng</b> dữ liệu phi tuyến, không phải lỗi.",
    )

    st.divider()

    # ── 3. BẢNG CHỈ SỐ ĐẦY ĐỦ — 9 hàng ──────────────────────────────────
    render_section_label("Bảng chỉ số đầy đủ (3 tác vụ × 3 thuật toán)")

    tbl = df[["task_vn", "algo", "accuracy", "precision", "recall", "f1"]].copy()
    tbl.columns = ["Tác vụ", "Thuật toán", "Accuracy", "Precision", "Recall", "F1"]

    def _hl_rf(row):
        is_rf = row["Thuật toán"] == "Random Forest"
        return ["background-color:#eaedfb;font-weight:600" if is_rf else "" for _ in row]

    st.dataframe(
        tbl.style.apply(_hl_rf, axis=1).format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}",
            "Recall": "{:.2%}", "F1": "{:.2%}",
        }),
        width="stretch", hide_index=True,
    )
    st.caption("Hàng tô nền là **Random Forest** — mô hình được chọn cho hệ thống.")

    st.divider()

    # ── 4. HAI THẺ INSIGHT ──────────────────────────────────────────────
    ins1, ins2 = st.columns(2, gap="large")

    noise = load_research_json("noise_robustness.json")
    with ins1:
        if noise and noise.get("levels"):
            lv0, lvN = noise["levels"][0], noise["levels"][-1]
            drops = {a: lv0["results"][a]["accuracy"] - lvN["results"][a]["accuracy"]
                     for a in ("Random Forest", "Decision Tree")}
            mx = max(drops.values()) or 1
            render_compare_bars(
                f"Độ bền khi {lvN['effective_train_noise_pct']:.1f}% nhãn train bị nhiễu",
                [("Random Forest", f"−{drops['Random Forest']*100:.2f} điểm %",
                  drops["Random Forest"] / mx * 100, "success"),
                 ("Decision Tree", f"−{drops['Decision Tree']*100:.2f} điểm %",
                  drops["Decision Tree"] / mx * 100, "danger")],
                note=f"Decision Tree tụt gần <b>{drops['Decision Tree']/max(drops['Random Forest'],1e-9):.1f}×</b> "
                     "so với Random Forest. Ensemble 100 cây trung bình hoá nên hấp thụ "
                     "nhiễu nhãn tốt hơn cây đơn — đây là tiêu chí chọn mô hình chính.",
            )
        else:
            render_banner("Chưa có noise_robustness.json.", "warning")

    with ins2:
        render_compare_bars(
            "Bằng chứng dữ liệu phi tuyến",
            [("Random Forest / Decision Tree (phi tuyến)", f"{rf_avg:.1%}", 100, "primary"),
             ("Logistic Regression (tuyến tính)",
              f"{lr_min:.0%}–{lr_max:.0%}", lr_max / max(rf_avg, 1e-9) * 100, "danger")],
            note="Logistic Regression chỉ vẽ được một ranh giới <b>thẳng</b> giữa các lớp. "
                 "Ranh giới thật trong dữ liệu là <b>cong</b> vì các biến tương tác với nhau "
                 "— ví dụ pin yếu chỉ thực sự nguy hiểm khi đồng thời gió mạnh. Khoảng cách "
                 "~20 điểm % này chính là thước đo mức độ phi tuyến.",
        )

    st.divider()

    # ── 5. BA MÔ HÌNH — BA QUYẾT ĐỊNH ───────────────────────────────────
    render_section_label("3 mô hình — 3 quyết định cho 3 đối tượng")
    st.caption(
        "Ba mô hình **không cạnh tranh nhau**: mỗi mô hình trả lời một câu hỏi "
        "quyết định khác nhau cho một đối tượng khác nhau. Hệ thống cần cả ba — "
        "vì vậy trang này **không xếp hạng “mô hình tốt nhất”**."
    )

    dcols = st.columns(len(MODEL_DEFS), gap="large")
    for col, (mname, task, _desc) in zip(dcols, MODEL_DEFS):
        m = all_metrics.get(mname)
        if m is None:
            continue
        vn, role, question = TASK_META[task]
        g = m.get("group_cv", {})
        with col:
            render_decision_card(
                role=role,
                question=question,
                target=task,
                accuracy=f"{m.get('accuracy', 0):.2%}",
                extra=f"{len(m.get('classes', []))} lớp · Group CV "
                      f"{g.get('mean', 0):.2%} (drone chưa từng thấy)",
            )

    st.divider()

    # ── 1c. Bảng chi tiết chỉ số RF (kèm đánh giá trên drone mới) ─────────
    render_section_label("Đánh giá tổng quát hoá — Random Forest (theo tác vụ)")

    gen_rows = []
    for mname, task, _ in MODEL_DEFS:
        m = all_metrics.get(mname)
        if m is None:
            continue
        g = m.get("group_cv", {})
        gen_rows.append({
            "Tác vụ":                TASK_META.get(task, (task,))[0],
            "Test Accuracy":         f"{m.get('accuracy', 0):.2%}",
            "Random CV":             f"{m.get('cv_mean', 0):.2%} ±{m.get('cv_std', 0):.2%}",
            "Group CV (drone mới)":  f"{g.get('mean', 0):.2%}" if g else "—",
            "Leakage gap":           f"{g.get('gap_vs_random', 0):+.4f}" if g else "—",
        })

    st.dataframe(pd.DataFrame(gen_rows), width="stretch", hide_index=True)


# ─── TAB 2 & 3: PILL SELECTOR DÙNG CHUNG ────────────────────────────────────

_PILL_KEYS = ("mi_pill_detail", "mi_pill_impact")


def _model_pill(all_metrics: dict, widget_key: str):
    """
    Dropdown Box chọn tác vụ. State DÙNG CHUNG giữa Tab 2 và Tab 3:
    on_change đồng bộ giá trị sang widget của tab kia.
    """
    by_label = {TASK_META[task][0]: mname
                for mname, task, _ in MODEL_DEFS if mname in all_metrics}
    labels = list(by_label.keys())
    if not labels:
        return None
    cur = st.session_state.get("mi_model_label", labels[0])
    if cur not in by_label:
        cur = labels[0]

    def _sync():
        val = st.session_state.get(widget_key)
        if val:
            st.session_state["mi_model_label"] = val
            for other in _PILL_KEYS:            # dong bo sang tab kia
                if other != widget_key and other in st.session_state:
                    st.session_state[other] = val

    # Lấy index mặc định
    current_index = labels.index(cur) if cur in labels else 0

    sel = st.selectbox(
        "Chọn tác vụ",
        options=labels,
        index=current_index,
        key=widget_key,
        label_visibility="collapsed",
        on_change=_sync
    )
    label = sel or cur
    st.session_state["mi_model_label"] = label
    return by_label[label]


# ─── TAB 2: PER-MODEL DETAIL ────────────────────────────────────────────────

def _render_detail_tab(all_metrics: dict):
    mname = _model_pill(all_metrics, "mi_pill_detail")
    if mname is None:
        render_banner("Chưa có metrics. Chạy `python train_model.py`.", "warning")
        return
    m = all_metrics[mname]
    task = next(t for mn, t, _ in MODEL_DEFS if mn == mname)
    task_vn, role, question = TASK_META[task]
    g = m.get("group_cv", {})

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Header model: tên + đối tượng + accuracy GroupKFold lớn ──────────
    st.markdown(
        f"""<div class="mi-head">
              <div>
                <p class="mh-title">{task_vn} — <span style="font-weight:500;
                   color:var(--text-secondary)">{question}</span></p>
                <span class="mh-role">{role}</span>
              </div>
              <div>
                <div class="mh-acc">{g.get('mean', m.get('accuracy', 0)):.2%}</div>
                <div class="mh-acc-l">accuracy trên drone chưa từng thấy (GroupKFold)</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 4 chỉ số cơ bản ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, label, val, delta in [
        (c1, "Accuracy",  f"{m.get('accuracy',  0):.2%}",
             f"CV {m.get('cv_mean', 0):.2%} ±{m.get('cv_std', 0):.2%}"),
        (c2, "Precision", f"{m.get('precision', 0):.2%}", "weighted avg"),
        (c3, "Recall",    f"{m.get('recall',    0):.2%}", "weighted avg"),
        (c4, "F1-Score",  f"{m.get('f1',        0):.2%}", "weighted avg"),
    ]:
        with col:
            with st.container(border=True):
                st.metric(label=label, value=val, delta=delta, delta_color="off")

    st.divider()

    # ── Per-class F1 + Confusion matrix ───────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        render_section_label("Per-class F1 Score")
        render_class_f1_table(
            m["classes"],
            m["per_class_f1"],
            m.get("per_class_precision"),
            m.get("per_class_recall"),
        )

    with right:
        render_section_label("Confusion Matrix")
        fig_cm = render_confusion_matrix_heatmap(m["confusion_matrix"], m["classes"])
        st.plotly_chart(
            style_chart(fig_cm, 300),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.divider()

    # ── Random CV vs GroupKFold (3 thẻ) ──────────────────────────────────
    render_section_label("Random CV vs GroupKFold — tổng quát hóa sang drone mới")
    v1, v2, v3 = st.columns(3, gap="large")
    with v1:
        st.markdown(
            f"""<div class="cv-card"><div class="cv-v">{m.get('cv_mean', 0):.2%}</div>
                <div class="cv-l">Random CV (baseline — có leakage tiềm ẩn)</div></div>""",
            unsafe_allow_html=True)
    with v2:
        st.markdown(
            f"""<div class="cv-card"><div class="cv-v" style="color:var(--accent)">
                {g.get('mean', 0):.2%}</div>
                <div class="cv-l">GroupKFold theo drone (drone chưa từng thấy)</div></div>""",
            unsafe_allow_html=True)
    with v3:
        st.markdown(
            f"""<div class="cv-card ok"><div class="cv-v">{g.get('gap_vs_random', 0):+.4f}</div>
                <div class="cv-l">Leakage gap ≈ 0 → không overfit theo thiết bị</div></div>""",
            unsafe_allow_html=True)

    st.divider()

    # ── Comparison models cho target này ─────────────────────────────────
    cmp_data = m.get("comparison")
    if cmp_data:
        render_section_label("So Sánh Với Các Thuật Toán Khác (cùng target)")
        cmp_rows = []
        cmp_rows.append({
            "Thuật Toán": "Random Forest ✅",
            "Accuracy":   m.get("accuracy",  0),
            "Precision":  m.get("precision", 0),
            "Recall":     m.get("recall",    0),
            "F1":         m.get("f1",        0),
        })
        for algo, vals in cmp_data.items():
            cmp_rows.append({
                "Thuật Toán": algo,
                "Accuracy":   vals.get("accuracy",  0),
                "Precision":  vals.get("precision", 0),
                "Recall":     vals.get("recall",    0),
                "F1":         vals.get("f1",        0),
            })

        cmp_tbl = pd.DataFrame(cmp_rows)
        st.dataframe(
            cmp_tbl.style.format({
                "Accuracy":  "{:.2%}",
                "Precision": "{:.2%}",
                "Recall":    "{:.2%}",
                "F1":        "{:.2%}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        f"📅 Trained: **{m.get('trained_at', '—')}** · "
        f"Sample: **{m.get('sample_size', m.get('train_samples', 0) + m.get('test_samples', 0)):,}** · "
        f"Test: **{m['test_samples']:,}**"
    )


# ─── TAB 3: FEATURE IMPACT (theo từng model — dữ liệu thật) ─────────────────

_FEATURE_VN = {
    "battery_level":   "Sức khỏe pin",
    "wind_speed":      "Tốc độ gió",
    "flight_time":     "Giờ bay tích lũy",
    "temperature":     "Nhiệt độ",
    "signal_strength": "Cường độ tín hiệu",
    "gps_accuracy":    "GPS accuracy",
    "humidity":        "Độ ẩm",
    "speed":           "Tốc độ bay",
    "altitude":        "Độ cao",
    "pressure":        "Áp suất",
}

_TASK_TO_MODEL_KEY = {
    "operation_risk": "risk", "maintenance_action": "maint",
    "recommendation": "rec",
}


def _model_importances(task: str):
    """Đọc feature_importances_ THẬT của model đã train (không dùng số mẫu)."""
    from core import load_models, FEATURES
    try:
        model, _ = load_models()[_TASK_TO_MODEL_KEY[task]]
        pairs = sorted(zip(FEATURES, model.feature_importances_),
                       key=lambda x: -x[1])
        return pairs
    except Exception:
        return None


def _render_impact_tab(all_metrics: dict):
    mname = _model_pill(all_metrics, "mi_pill_impact")
    if mname is None:
        render_banner("Chưa có metrics. Chạy `python train_model.py`.", "warning")
        return
    task = next(t for mn, t, _ in MODEL_DEFS if mn == mname)
    task_vn, role, _q = TASK_META[task]

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    pairs = _model_importances(task)
    if pairs is None:
        # Fallback: file trung bình 3 model
        fi_df = load_feature_importance()
        if fi_df is None:
            render_banner("Chưa có model/feature_importance. Chạy `python train_model.py`.",
                          "warning")
            return
        st.plotly_chart(style_chart(render_feature_importance_chart(fi_df)),
                        width="stretch", config={"displayModeBar": False})
        return

    render_section_label(f"Feature importance — {task_vn} (Random Forest)")

    labels = [f"{_FEATURE_VN.get(f, f)}  ·  {f}" for f, _ in pairs][::-1]
    vals   = [v for _, v in pairs][::-1]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker=dict(color="#4f63d2"),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, max(vals) * 1.22], title=None),
        yaxis=dict(title=None),
        margin=dict(l=10, r=10),
    )
    st.plotly_chart(style_chart(fig, 400), width="stretch",
                    config={"displayModeBar": False})

    top = pairs[:3]
    bot = pairs[-4:]
    top_txt = " · ".join(f"<b>{_FEATURE_VN.get(f, f)}</b> ({v:.1%})" for f, v in top)
    render_callout(
        "Nghĩa là gì cho người ra quyết định",
        f"Với tác vụ <b>{task_vn}</b> (phục vụ {role}), quyết định của mô hình "
        f"được dẫn dắt chủ yếu bởi: {top_txt}. Các biến này còn <b>tương tác</b> "
        "với nhau — ví dụ pin yếu chỉ thực sự nguy hiểm khi đồng thời gió mạnh "
        "— đó chính là cấu trúc phi tuyến khiến Logistic Regression thất bại.",
        icon="🎯",
    )
    st.caption(
        "• 4 biến ít ảnh hưởng nhất (`"
        + "`, `".join(f for f, _ in bot)
        + f"`) chỉ đóng góp ~{sum(v for _, v in bot):.1%} tổng importance — "
        "nhất quán với bộ luật sinh nhãn."
    )