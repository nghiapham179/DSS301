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
    load_metrics, load_feature_importance, style_chart,
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
        _render_impact_tab()


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
        f"Relabeling: **{m.get('relabeling', 'hard labels')}**"
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