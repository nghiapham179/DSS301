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
    render_top_nav, render_page_title, render_section_label, render_banner,
    render_class_f1_table, render_confusion_matrix_heatmap,
    render_feature_importance_chart,
)


MODEL_DEFS = [
    ("operation_risk_model",     "operation_risk",     "Dự đoán mức rủi ro tổng thể (High / Medium / Low)"),
    ("recommendation_model",     "recommendation",     "Khuyến nghị vận hành cuối cùng cho phi công"),
    ("maintenance_action_model", "maintenance_action", "Khuyến nghị hành động bảo trì (Monitor / Inspect / ...)"),
]

COMPARISON_MODELS = ["Random Forest", "Decision Tree", "Logistic Regression"]


def render():
    render_top_nav()
    startup_load_or_stop()

    render_page_title(
        "Model Info",
        "So sánh hiệu năng giữa 3 RF models — chọn model tốt nhất cho từng tác vụ.",
    )

    all_metrics = {}
    for mname, _, _ in MODEL_DEFS:
        m = load_metrics(mname)
        if m is not None:
            all_metrics[mname] = m

    if not all_metrics:
        render_banner(
            "Chưa có file _metrics.json. Chạy `python retrain_fast.py` trước.",
            "warning",
        )
        return

    tab_overview, tab_detail, tab_impact, tab_research = st.tabs([
        "📊  Tổng quan & So sánh",
        "🔍  Chi tiết theo model",
        "🎯  Feature Impact",
        "🧪  Nghiên cứu",
    ])

    with tab_overview:
        _render_overview_tab(all_metrics)

    with tab_detail:
        _render_detail_tab(all_metrics)

    with tab_impact:
        _render_impact_tab()

    with tab_research:
        _render_research_tab(all_metrics)


# ─── TAB 1: OVERVIEW & COMPARISON ───────────────────────────────────────────

def _render_overview_tab(all_metrics: dict):
    # ── 1a. RF model ranking cards ────────────────────────────────────────
    render_section_label("Xếp hạng 3 RF Models (Random Forest)")

    rows = []
    for mname, display, desc in MODEL_DEFS:
        m = all_metrics.get(mname)
        if m is None:
            continue
        f1_vals  = list(m["per_class_f1"].values())
        macro_f1 = round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0
        rows.append({
            "mname":        mname,
            "Model":        display,
            "Desc":         desc,
            "Test Acc":     m["accuracy"],
            "CV Mean":      m["cv_mean"],
            "CV Std":       m["cv_std"],
            "F1 Macro":     macro_f1,
            "Precision":    m.get("precision", 0),
            "Recall":       m.get("recall", 0),
            "RMSE":         m.get("rmse", 0),
            "MAE":          m.get("mae", 0),
            "Classes":      len(m["classes"]),
            "Train time":   m.get("train_time_s", 0),
        })

    if not rows:
        return

    cmp_df = pd.DataFrame(rows)
    cmp_df["Combined Score"] = (cmp_df["CV Mean"] * cmp_df["F1 Macro"]).round(4)
    ranked = cmp_df.sort_values("Combined Score", ascending=False).reset_index(drop=True)

    rank_cols = st.columns(len(ranked), gap="large")

    for i, (col, (_, row)) in enumerate(zip(rank_cols, ranked.iterrows())):
        if i == 0:
            medal, ac, bg, rl = "🥇", "#16a34a", "#dcfce7", "Tốt nhất"
        elif i == 1:
            medal, ac, bg, rl = "🥈", "#0284c7", "#e0f2fe", "Thứ 2"
        else:
            medal, ac, bg, rl = "🥉", "#d97706", "#fef3c7", "Thứ 3"

        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                        <span style="font-size:1.4rem">{medal}</span>
                        <span style="background:{bg};color:{ac};
                                     font-size:0.65rem;font-weight:700;
                                     padding:2px 8px;border-radius:999px;
                                     letter-spacing:.05em;">{rl.upper()}</span>
                    </div>
                    <p style="font-size:0.95rem;font-weight:700;margin:2px 0 0;
                              color:#1c1e2e;">{row['Model']}</p>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Combined Score",
                    value=f"{row['Combined Score']:.4f}",
                    delta=f"CV: {row['CV Mean']:.3f} · F1: {row['F1 Macro']:.3f}",
                    delta_color="off",
                )
                st.caption(f"Test accuracy: **{row['Test Acc']:.2%}**")

    st.divider()

    # ── 1b. Bảng so sánh RF ──────────────────────────────────────────────
    render_section_label("Chi tiết chỉ số — Random Forest")

    display_df = cmp_df[[
        "Model", "Test Acc", "Precision", "Recall",
        "F1 Macro", "RMSE", "MAE", "CV Mean", "CV Std", "Combined Score"
    ]].copy()

    for col in ["Test Acc", "Precision", "Recall", "F1 Macro", "CV Mean"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
    display_df["CV Std"]         = display_df["CV Std"].apply(lambda x: f"±{x:.2%}")
    display_df["RMSE"]           = display_df["RMSE"].apply(lambda x: f"{x:.4f}")
    display_df["MAE"]            = display_df["MAE"].apply(lambda x: f"{x:.4f}")
    display_df["Combined Score"] = display_df["Combined Score"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "💡 **Combined Score** = CV Mean × F1 Macro. "
        "**RMSE / MAE** tính trên nhãn encoded — càng gần 0 càng tốt."
    )

    st.divider()

    # ── 1c. So sánh 3 thuật toán (RF vs DT vs LR) ────────────────────────
    render_section_label("So Sánh Thuật Toán: RF vs Decision Tree vs Logistic Regression")

    has_comparison = any(
        "comparison" in all_metrics.get(mname, {})
        for mname, _, _ in MODEL_DEFS
    )

    if not has_comparison:
        render_banner(
            "Chưa có dữ liệu so sánh thuật toán. "
            "Chạy `python retrain_fast.py` để cập nhật metrics đầy đủ.",
            "warning",
        )
    else:
        algo_rows = []
        for algo in ["Random Forest", "Decision Tree", "Logistic Regression"]:
            acc_list, f1_list, rmse_list, mae_list = [], [], [], []
            for mname, _, _ in MODEL_DEFS:
                m = all_metrics.get(mname, {})
                if algo == "Random Forest":
                    acc_list.append(m.get("accuracy", 0))
                    f1_list.append(m.get("f1", 0))
                    rmse_list.append(m.get("rmse", 0))
                    mae_list.append(m.get("mae", 0))
                else:
                    cmp = m.get("comparison", {}).get(algo, {})
                    acc_list.append(cmp.get("accuracy", 0))
                    f1_list.append(cmp.get("f1", 0))
                    rmse_list.append(cmp.get("rmse", 0))
                    mae_list.append(cmp.get("mae", 0))

            algo_rows.append({
                "Thuật Toán":   algo,
                "Avg Accuracy": round(sum(acc_list)/len(acc_list),  4),
                "Avg F1":       round(sum(f1_list)/len(f1_list),    4),
                "Avg RMSE":     round(sum(rmse_list)/len(rmse_list), 4),
                "Avg MAE":      round(sum(mae_list)/len(mae_list),   4),
                "Lý Do":        _algo_reason(algo),
            })

        algo_df = pd.DataFrame(algo_rows)
        best_acc = algo_df["Avg Accuracy"].max()

        def _highlight(row):
            color = "#dcfce7" if row["Avg Accuracy"] == best_acc else ""
            return [f"background-color:{color}" if color else "" for _ in row]

        styled = algo_df.style.apply(_highlight, axis=1).format({
            "Avg Accuracy": "{:.2%}",
            "Avg F1":       "{:.2%}",
            "Avg RMSE":     "{:.4f}",
            "Avg MAE":      "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        fig = go.Figure()
        algos  = algo_df["Thuật Toán"].tolist()
        for metric, color in zip(["Avg Accuracy", "Avg F1"], ["#4f63d2", "#16a34a"]):
            fig.add_trace(go.Bar(
                name=metric.replace("Avg ", ""),
                x=algos,
                y=algo_df[metric].tolist(),
                marker=dict(color=color),
                text=[f"{v:.1%}" for v in algo_df[metric].tolist()],
                textposition="outside",
                textfont=dict(size=11, family="JetBrains Mono"),
            ))

        fig.update_layout(
            barmode="group",
            yaxis=dict(tickformat=".0%", title="Score", range=[0, 1.05]),
            xaxis=dict(title=None),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(
            style_chart(fig, 360),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption(
            "📌 **Random Forest** được chọn làm model chính: "
            "accuracy cao nhất, ổn định hơn Decision Tree (ensemble), "
            "và vượt trội Logistic Regression (data phi tuyến)."
        )


def _algo_reason(algo: str) -> str:
    return {
        "Random Forest":       "✅ Model chính — ensemble 100 cây, kháng overfit tốt",
        "Decision Tree":       "🔶 So sánh #1 — cây đơn, phương sai cao hơn RF",
        "Logistic Regression": "🔴 So sánh #2 — tuyến tính, không phù hợp data phi tuyến",
    }.get(algo, "")


# ─── TAB 2: PER-MODEL DETAIL ────────────────────────────────────────────────

def _render_detail_tab(all_metrics: dict):
    render_banner(
        "Chọn model để xem chi tiết: tất cả 6 metrics, F1 từng class, confusion matrix.",
        "info",
    )

    options = {
        f"{display} — {desc}": mname
        for mname, display, desc in MODEL_DEFS
        if mname in all_metrics
    }

    selected_label = st.selectbox(
        "Chọn model", list(options.keys()), key="model_detail_selector",
    )
    mname = options[selected_label]
    m     = all_metrics[mname]

    st.divider()

    # ── 6 metrics cards ───────────────────────────────────────────────────
    render_section_label("6 Metrics Đánh Giá")

    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    metrics_map = [
        (c1, "Accuracy",  f"{m.get('accuracy',  0):.2%}", f"CV {m.get('cv_mean', 0):.2%} ±{m.get('cv_std', 0):.2%}"),
        (c2, "Precision", f"{m.get('precision', 0):.2%}", "weighted avg"),
        (c3, "Recall",    f"{m.get('recall',    0):.2%}", "weighted avg"),
        (c4, "F1-Score",  f"{m.get('f1',        0):.2%}", "weighted avg"),
        (c5, "RMSE",      f"{m.get('rmse',      0):.4f}", "nhãn encoded"),
        (c6, "MAE",       f"{m.get('mae',       0):.4f}", "nhãn encoded"),
    ]
    for col, label, val, delta in metrics_map:
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
            use_container_width=True,
            config={"displayModeBar": False},
        )

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
            "RMSE":       m.get("rmse",      0),
            "MAE":        m.get("mae",       0),
        })
        for algo, vals in cmp_data.items():
            cmp_rows.append({
                "Thuật Toán": algo,
                "Accuracy":   vals.get("accuracy",  0),
                "Precision":  vals.get("precision", 0),
                "Recall":     vals.get("recall",    0),
                "F1":         vals.get("f1",        0),
                "RMSE":       vals.get("rmse",      0),
                "MAE":        vals.get("mae",       0),
            })

        cmp_tbl = pd.DataFrame(cmp_rows)
        st.dataframe(
            cmp_tbl.style.format({
                "Accuracy":  "{:.2%}",
                "Precision": "{:.2%}",
                "Recall":    "{:.2%}",
                "F1":        "{:.2%}",
                "RMSE":      "{:.4f}",
                "MAE":       "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"📅 Trained: **{m.get('trained_at', '—')}** · "
        f"Sample: **{m.get('sample_size', m.get('train_samples', 0) + m.get('test_samples', 0)):,}** · "
        f"Test: **{m['test_samples']:,}** · "
        f"Protocol: **{m.get('evaluation_protocol', m.get('relabeling', '—'))}**"
    )


# ─── TAB 3: FEATURE IMPACT ──────────────────────────────────────────────────

def _render_impact_tab():
    render_banner(
        "Feature importance trung bình của 3 models — feature nào ảnh hưởng "
        "lớn nhất đến quyết định của hệ thống.",
        "info",
    )

    fi_df = load_feature_importance()
    if fi_df is None:
        render_banner(
            "Chưa có file feature_importance.csv. Chạy `python retrain_fast.py`.",
            "warning",
        )
        return

    st.plotly_chart(
        style_chart(render_feature_importance_chart(fi_df)),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    top3 = fi_df.head(3)
    bot4 = fi_df.tail(4)

    st.caption("💡 **Nhận xét:**")
    st.caption(
        "• **Top 3** feature quan trọng nhất: "
        + ", ".join(
            f"`{r.feature}` ({r.importance:.1%})" for r in top3.itertuples()
        )
        + " — chiếm phần lớn quyết định của model."
    )
    st.caption(
        "• **4 feature ít quan trọng nhất** (`"
        + "`, `".join(bot4["feature"].tolist())
        + "`) chỉ đóng góp ~"
        + f"{bot4['importance'].sum():.1%} tổng importance — có thể loại bỏ "
        + "nếu cần tối ưu thời gian train mà không ảnh hưởng accuracy đáng kể."
    )


# ─── TAB 4: NGHIÊN CỨU (3 thí nghiệm) ───────────────────────────────────────

def _render_research_tab(all_metrics: dict):
    render_banner(
        "3 thí nghiệm đánh giá nâng cao: **tổng quát hóa sang drone mới** "
        "(GroupKFold), **độ bền nhiễu nhãn** (train-only noise), và "
        "**quyết định nhạy chi phí** (Elkan 2001).",
        "info",
    )

    _render_generalization_section(all_metrics)
    st.divider()
    _render_noise_section()
    st.divider()
    _render_cost_section()


# ── [1] Generalization: Random CV vs GroupKFold theo drone ──────────────────

def _render_generalization_section(all_metrics: dict):
    render_section_label("Thí nghiệm 1 — Tổng quát hóa sang drone chưa từng thấy")

    st.markdown(
        "**Câu hỏi nghiên cứu:** mô hình có dùng được cho drone *mới* không, "
        "hay chỉ 'thuộc bài' các drone trong tập train? "
        "**Phương pháp:** so sánh *Random Stratified CV* (bản ghi của cùng "
        "một drone xuất hiện ở cả train lẫn validation) với *GroupKFold theo "
        "`drone_id`* (mỗi fold validate trên 2 drone bị loại hoàn toàn khỏi "
        "train). Chênh lệch giữa hai giá trị chính là **leakage gap** "
        "(Kapoor & Narayanan 2023, *Patterns*; Roberts et al. 2021, "
        "*Nature Machine Intelligence*)."
    )

    rows = []
    for mname, display, _ in MODEL_DEFS:
        m = all_metrics.get(mname)
        if m is None or "group_cv" not in m:
            continue
        g = m["group_cv"]
        rows.append({
            "target":      display,
            "random_mean": m["cv_mean"],
            "random_std":  m["cv_std"],
            "group_mean":  g["mean"],
            "group_std":   g["std"],
            "gap":         g["gap_vs_random"],
        })

    if not rows:
        render_banner(
            "Chưa có kết quả GroupKFold. Chạy lại `python train_model.py` "
            "(phiên bản v3) để sinh dữ liệu.",
            "warning",
        )
        return

    fig = go.Figure()
    targets = [r["target"] for r in rows]
    fig.add_trace(go.Bar(
        name="Random CV (có leakage tiềm ẩn)",
        x=targets, y=[r["random_mean"] for r in rows],
        error_y=dict(type="data", array=[r["random_std"] for r in rows]),
        marker=dict(color="#4f63d2"),
        text=[f"{r['random_mean']:.2%}" for r in rows],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="GroupKFold theo drone (unseen-drone)",
        x=targets, y=[r["group_mean"] for r in rows],
        error_y=dict(type="data", array=[r["group_std"] for r in rows]),
        marker=dict(color="#16a34a"),
        text=[f"{r['group_mean']:.2%}" for r in rows],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(tickformat=".0%", title="Accuracy (CV mean ± std)",
                   range=[0, 1.12]),
        legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
    )
    st.plotly_chart(style_chart(fig, 380), use_container_width=True,
                    config={"displayModeBar": False})

    gap_df = pd.DataFrame([{
        "Target":            r["target"],
        "Random CV":         f"{r['random_mean']:.2%} ±{r['random_std']:.2%}",
        "Group CV (drone mới)": f"{r['group_mean']:.2%} ±{r['group_std']:.2%}",
        "Leakage gap":       f"{r['gap']:+.4f}",
    } for r in rows])
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    max_gap = max(abs(r["gap"]) for r in rows)
    if max_gap < 0.01:
        st.caption(
            "💡 **Diễn giải:** gap < 1 điểm % — mô hình tổng quát tốt sang "
            "drone chưa thấy. Điều này *nhất quán* với bản chất dữ liệu: nhãn "
            "synthetic được sinh từ một bộ luật chung cho mọi drone, không có "
            "đặc thù riêng theo thiết bị. Với dữ liệu bay thật (mỗi drone hao "
            "mòn khác nhau), gap này dự kiến sẽ lớn hơn — đây là hạn chế của "
            "dữ liệu synthetic cần nêu trong báo cáo."
        )
    else:
        st.caption(
            f"⚠️ **Diễn giải:** gap lớn nhất {max_gap:.2%} — random split đã "
            "ước lượng **quá lạc quan**; con số GroupKFold mới là ước lượng "
            "trung thực cho kịch bản triển khai trên drone mới."
        )


# ── [2] Noise robustness: nhiễu train-only, test sạch ────────────────────────

def _render_noise_section():
    render_section_label("Thí nghiệm 2 — Độ bền nhiễu nhãn (Noise Robustness)")

    data = load_research_json("noise_robustness.json")
    if data is None:
        render_banner(
            "Chưa có noise_robustness.json. Chạy `python train_model.py` (v3).",
            "warning",
        )
        return

    st.markdown(
        "**Câu hỏi nghiên cứu:** khi nhãn huấn luyện bị gán sai (điều tất yếu "
        "với dữ liệu vận hành thật), thuật toán nào giữ được chất lượng tốt "
        "nhất? **Phương pháp:** tiêm nhiễu nhãn *phụ thuộc đặc trưng* (NNAR — "
        "bản ghi ở vùng biên giữa các lớp bị flip nhiều hơn vùng rõ ràng, sát "
        "thực tế hơn flip đều; Frénay & Verleysen 2014, *IEEE TNNLS*) vào "
        "**tập train duy nhất** ở 5 mức, rồi đo trên **tập test giữ sạch**. "
        "Khác với cách cũ (v2) tiêm nhiễu vào cả test làm sai lệch thước đo."
    )

    levels   = data["levels"]
    x_vals   = [lv["boundary_flip_rate"] * 100 for lv in levels]
    eff_vals = [lv["effective_train_noise_pct"] for lv in levels]
    algos    = list(levels[0]["results"].keys())
    algo_colors = {
        "Random Forest":       "#4f63d2",
        "Decision Tree":       "#d97706",
        "Logistic Regression": "#dc2626",
    }

    col_acc, col_rec = st.columns(2, gap="large")

    with col_acc:
        fig = go.Figure()
        for algo in algos:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=[lv["results"][algo]["accuracy"] for lv in levels],
                mode="lines+markers", name=algo,
                line=dict(color=algo_colors.get(algo, "#888"), width=2.5),
                marker=dict(size=7),
            ))
        fig.update_layout(
            xaxis=dict(title="Mức nhiễu vùng biên trong TRAIN (%)"),
            yaxis=dict(title="Accuracy trên test SẠCH", tickformat=".0%"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(style_chart(fig, 360), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Accuracy trên test sạch theo mức nhiễu train.")

    with col_rec:
        fig = go.Figure()
        for algo in algos:
            rec_vals = [lv["results"][algo].get("recall_high") for lv in levels]
            if any(v is None for v in rec_vals):
                continue
            fig.add_trace(go.Scatter(
                x=x_vals, y=rec_vals,
                mode="lines+markers", name=algo,
                line=dict(color=algo_colors.get(algo, "#888"), width=2.5,
                          dash="dot"),
                marker=dict(size=7),
            ))
        fig.update_layout(
            xaxis=dict(title="Mức nhiễu vùng biên trong TRAIN (%)"),
            yaxis=dict(title="Recall lớp High (an toàn)", tickformat=".0%"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(style_chart(fig, 360), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Recall lớp High — tỷ lệ tình huống nguy hiểm KHÔNG bị bỏ sót.")

    tbl = pd.DataFrame([{
        "Nhiễu biên (%)":       f"{lv['boundary_flip_rate']:.0%}",
        "Flip thực tế (%)":     f"{lv['effective_train_noise_pct']:.1f}%",
        **{f"{algo}": f"{lv['results'][algo]['accuracy']:.2%}" for algo in algos},
    } for lv in levels])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Nhận xét tự động: thuật toán bền nhiễu nhất = giảm accuracy ít nhất,
    # nhưng CHỈ so giữa các mô hình có baseline mạnh (>= 90% ở mức nhiễu 0) —
    # mô hình vốn đã kém (vd LR ~76%) "phẳng" theo nhiễu không phải vì bền,
    # mà vì nó chưa bao giờ học được ranh giới lớp để mà bị nhiễu phá.
    drops = {
        algo: levels[0]["results"][algo]["accuracy"]
              - levels[-1]["results"][algo]["accuracy"]
        for algo in algos
    }
    strong = {a: d for a, d in drops.items()
              if levels[0]["results"][a]["accuracy"] >= 0.90}
    pool        = strong if strong else drops
    most_robust = min(pool, key=pool.get)
    st.caption(
        f"💡 **Kết luận:** từ 0% → {levels[-1]['boundary_flip_rate']:.0%} nhiễu "
        f"(flip thực tế {eff_vals[-1]:.1f}%), trong các mô hình có baseline mạnh, "
        f"**{most_robust}** giảm accuracy ít nhất ({pool[most_robust]*100:.2f} điểm %) — "
        + " · ".join(f"{a}: −{d*100:.2f}đ%" for a, d in
                     sorted(drops.items(), key=lambda kv: kv[1]))
        + ". Ensemble trung bình hóa nhiều cây giúp RF hấp thụ nhiễu nhãn tốt hơn "
          "cây đơn (Frénay & Verleysen 2014, §V). Lưu ý: LR trông 'phẳng' theo "
          "nhiễu chỉ vì baseline của nó đã thấp (mô hình tuyến tính không biểu "
          "diễn được ranh giới phi tuyến), không phải vì bền nhiễu."
    )


# ── [3] Cost-sensitive decision (Elkan 2001) ─────────────────────────────────

def _render_cost_section():
    render_section_label("Thí nghiệm 3 — Quyết định nhạy chi phí (Cost-Sensitive)")

    data = load_research_json("cost_sensitive.json")
    if data is None:
        render_banner(
            "Chưa có cost_sensitive.json. Chạy `python train_model.py` (v3).",
            "warning",
        )
        return

    st.markdown(
        "**Vấn đề:** với DSS an toàn bay, *bỏ sót* một tình huống High-risk "
        "(false negative) nguy hiểm hơn nhiều một *báo động nhầm* (false "
        "positive), nhưng argmax mặc định coi hai lỗi này ngang nhau. "
        "**Phương pháp:** định nghĩa ma trận chi phí bất đối xứng rồi thay "
        "argmax bằng **quy tắc Bayes tối thiểu chi phí kỳ vọng** "
        "*j\\* = argmin<sub>j</sub> Σ<sub>i</sub> P(i|x)·C(i,j)* — không cần "
        "huấn luyện lại, không có siêu tham số cần tinh chỉnh (Elkan 2001, "
        "*IJCAI*). **Quy tắc này đang được áp dụng thật trong trang Dự đoán.**",
        unsafe_allow_html=True,
    )

    classes = data["classes"]
    order   = [c for c in ["High", "Medium", "Low"] if c in classes]
    C       = pd.DataFrame(data["cost_matrix"], index=classes, columns=classes)
    C       = C.loc[order, order]

    col_matrix, col_compare = st.columns([2, 3], gap="large")

    with col_matrix:
        st.markdown("**Ma trận chi phí C(thực tế → dự đoán)**")
        cmax = float(C.values.max()) or 1.0

        def _shade(v):
            # đỏ đậm dần theo chi phí (không cần matplotlib)
            alpha = 0.08 + 0.72 * (float(v) / cmax)
            fg    = "#7f1d1d" if v < cmax * 0.6 else "#ffffff"
            return f"background-color: rgba(220,38,38,{alpha:.2f}); color:{fg};"

        st.dataframe(
            C.style.format("{:.0f}").map(_shade),
            use_container_width=True,
        )
        st.caption(
            "Bỏ sót hoàn toàn High (dự đoán Low) = 10 điểm — đắt gấp 10 lần "
            "một báo động nhầm nhẹ (1 điểm)."
        )

    base  = data["baseline_argmax"]
    bayes = data["bayes_cost_minimizing"]

    with col_compare:
        st.markdown("**Argmax (mặc định) vs Bayes cost-minimizing**")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric(
                    "Recall lớp High",
                    f"{bayes['recall_high']:.2%}",
                    delta=f"{(bayes['recall_high']-base['recall_high'])*100:+.2f} điểm % so với argmax",
                )
        with c2:
            with st.container(border=True):
                st.metric(
                    "Chi phí kỳ vọng / chuyến bay",
                    f"{bayes['mean_cost']:.4f}",
                    delta=f"{(bayes['mean_cost']/base['mean_cost']-1)*100:+.1f}% so với argmax",
                    delta_color="inverse",
                )
        with c3:
            with st.container(border=True):
                st.metric(
                    "Accuracy tổng",
                    f"{bayes['accuracy']:.2%}",
                    delta=f"{(bayes['accuracy']-base['accuracy'])*100:+.2f} điểm % so với argmax",
                )
        st.caption(
            f"Baseline argmax: recall(High) {base['recall_high']:.2%} · "
            f"chi phí {base['mean_cost']:.4f} · accuracy {base['accuracy']:.2%} "
            f"(test n={data['test_samples']:,})."
        )

    if bayes["mean_cost"] <= base["mean_cost"]:
        st.caption(
            "💡 **Diễn giải:** quy tắc Bayes giảm chi phí kỳ vọng đúng như "
            "lý thuyết (Elkan 2001) — đánh đổi một phần accuracy để không "
            "bỏ sót High-risk."
        )
    else:
        st.caption(
            "💡 **Diễn giải (phát hiện trung thực):** trên nhãn sạch, mô hình "
            "đã gần hoàn hảo nên hầu như không còn ca High bị bỏ sót để 'cứu'; "
            "quy tắc Bayes đẩy các ca biên sang High → đạt **recall High = "
            f"{bayes['recall_high']:.0%}** (an toàn tuyệt đối) nhưng chi phí "
            "thực tế *tăng* do báo động nhầm. Điều này minh họa hai điểm lý "
            "thuyết: (1) lợi ích của cost-sensitive chỉ rõ khi mô hình còn "
            "**bất định** (dữ liệu thật, nhiễu — xem Thí nghiệm 2); (2) quy tắc "
            "Elkan giả định xác suất **đã hiệu chỉnh** — xác suất thô của RF "
            "vốn miscalibrated (Niculescu-Mizil & Caruana 2005), nên bước "
            "calibration là hướng phát triển tiếp theo tự nhiên."
        )

    # Threshold sweep — đường cong mô tả trade-off
    sweep = data.get("threshold_sweep", [])
    if sweep:
        fig = go.Figure()
        taus = [s["tau"] for s in sweep]
        fig.add_trace(go.Scatter(
            x=taus, y=[s["recall_high"] for s in sweep],
            mode="lines+markers", name="Recall (High)",
            line=dict(color="#dc2626", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=taus, y=[s["accuracy"] for s in sweep],
            mode="lines+markers", name="Accuracy",
            line=dict(color="#4f63d2", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=taus, y=[s["mean_cost"] for s in sweep],
            mode="lines+markers", name="Chi phí kỳ vọng (trục phải)",
            line=dict(color="#d97706", width=2.5, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            xaxis=dict(title="Ngưỡng τ — dự đoán High nếu P(High) ≥ τ"),
            yaxis=dict(title="Recall / Accuracy", tickformat=".0%"),
            yaxis2=dict(title="Chi phí kỳ vọng", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(style_chart(fig, 380), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "💡 Đường cong độ nhạy theo ngưỡng τ (mô tả trade-off, tính trên "
            "test). Hạ τ → bắt được nhiều High hơn (recall ↑) nhưng accuracy "
            "giảm nhẹ do báo động nhầm tăng. Quy tắc Bayes không cần chọn τ "
            "thủ công — ngưỡng tối ưu được suy trực tiếp từ ma trận chi phí "
            "(Elkan 2001, Theorem 1)."
        )