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


# Đã cập nhật tên hiển thị thành Alpha, Beta, Delta tại đây
MODEL_DEFS = [
    ("operation_risk_model",     "Model Alpha", "Dự đoán mức rủi ro tổng thể (High / Medium / Low)"),
    ("recommendation_model",     "Model Beta",  "Khuyến nghị vận hành cuối cùng cho phi công"),
    ("maintenance_action_model", "Model Delta", "Khuyến nghị hành động bảo trì (Monitor / Inspect / ...)"),
]


def render():
    render_top_nav()
    startup_load_or_stop()

    render_page_title(
        "Model Info",
        "So sánh hiệu năng giữa 3 RF models — chọn model tốt nhất cho từng tác vụ.",
    )

    # Load all metrics upfront
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
        _render_impact_tab()


# ─── TAB 1: OVERVIEW & COMPARISON ───────────────────────────────────────────

def _render_overview_tab(all_metrics: dict):
    render_banner(
        "Bảng so sánh 3 model side-by-side. Model có **CV Mean cao + CV Std thấp + F1 macro cao** "
        "là model ổn định và đáng tin cậy nhất.",
        "info",
    )

    # ── Comparison summary table ──────────────────────────
    render_section_label("Bảng so sánh hiệu năng")

    rows = []
    for mname, display, desc in MODEL_DEFS:
        m = all_metrics.get(mname)
        if m is None:
            continue
        # Compute macro F1
        f1_vals = list(m["per_class_f1"].values())
        macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0
        rows.append({
            "Model":       display,
            "Test Acc":    m["accuracy"],
            "CV Mean":     m["cv_mean"],
            "CV Std":      m["cv_std"],
            "F1 Macro":    round(macro_f1, 4),
            "Classes":     len(m["classes"]),
            "Train time":  f"{m.get('train_time_s', 0):.1f}s",
        })

    if not rows:
        return

    cmp_df = pd.DataFrame(rows)

    # ── Best-model ranking ────────────────────────────────
    # Score = CV_mean × F1_macro (penalises both inaccuracy and class imbalance)
    cmp_df["Combined Score"] = (cmp_df["CV Mean"] * cmp_df["F1 Macro"]).round(4)
    ranked = cmp_df.sort_values("Combined Score", ascending=False).reset_index(drop=True)

    # Highlight cards
    rank_cols = st.columns(len(ranked), gap="large")
    for i, (col, row) in enumerate(zip(rank_cols, ranked.itertuples())):
        # Medal styling
        if i == 0:
            medal, accent_color, bg_color = "🥇", "#16a34a", "#dcfce7"
            rank_label = "Tốt nhất"
        elif i == 1:
            medal, accent_color, bg_color = "🥈", "#0284c7", "#e0f2fe"
            rank_label = "Thứ 2"
        else:
            medal, accent_color, bg_color = "🥉", "#d97706", "#fef3c7"
            rank_label = "Thứ 3"

        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                        <span style="font-size:1.4rem">{medal}</span>
                        <span style="background:{bg_color};color:{accent_color};
                                     font-size:0.65rem;font-weight:700;
                                     padding:2px 8px;border-radius:999px;
                                     letter-spacing:.05em;">{rank_label.upper()}</span>
                    </div>
                    <p style="font-size:0.95rem;font-weight:700;margin:2px 0 0;
                              color:#1c1e2e;">{row.Model}</p>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Combined Score",
                    value=f"{row._8:.4f}",
                    delta=f"CV: {row._3:.3f} · F1: {row._5:.3f}",
                    delta_color="off",
                )
                st.caption(f"Test accuracy: **{row._2:.2%}**")

    st.divider()

    # ── Full comparison table ─────────────────────────────
    render_section_label("Chi tiết các chỉ số")

    display_df = cmp_df.copy()
    display_df["Test Acc"]       = display_df["Test Acc"].apply(lambda x: f"{x:.2%}")
    display_df["CV Mean"]        = display_df["CV Mean"].apply(lambda x: f"{x:.2%}")
    display_df["CV Std"]         = display_df["CV Std"].apply(lambda x: f"±{x:.2%}")
    display_df["F1 Macro"]       = display_df["F1 Macro"].apply(lambda x: f"{x:.4f}")
    display_df["Combined Score"] = display_df["Combined Score"].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "💡 **Combined Score** = CV Mean × F1 Macro — kết hợp độ chính xác và "
        "khả năng cân bằng giữa các class. Model có Combined Score cao nhất "
        "đáng tin cậy nhất cho việc deploy thực tế."
    )

    # ── Multi-model accuracy comparison chart ─────────────
    st.divider()
    render_section_label("Biểu đồ so sánh trực quan")

    fig = go.Figure()
    metrics_to_plot = ["Test Acc", "CV Mean", "F1 Macro"]
    raw_values = {
        "Test Acc": cmp_df["Test Acc"].astype(float).tolist(),
        "CV Mean":  cmp_df["CV Mean"].astype(float).tolist(),
        "F1 Macro": cmp_df["F1 Macro"].astype(float).tolist(),
    }
    model_names = cmp_df["Model"].tolist()
    colors = ["#4f63d2", "#16a34a", "#d97706"]

    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            name=metric,
            x=model_names,
            y=raw_values[metric],
            marker=dict(color=colors[i], line=dict(width=0)),
            text=[f"{v:.3f}" for v in raw_values[metric]],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0.95, 1.005], tickformat=".2%", title="Score"),
        xaxis=dict(title=None),
        bargap=0.25,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )

    st.plotly_chart(
        style_chart(fig, 380),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        "📊 So sánh 3 chỉ số chính giữa các model. Trục Y zoom từ 95% để thấy "
        "sự khác biệt rõ hơn (tất cả đều > 99%)."
    )


# ─── TAB 2: PER-MODEL DETAIL ────────────────────────────────────────────────

def _render_detail_tab(all_metrics: dict):
    render_banner(
        "Chọn model để xem chi tiết: F1 từng class, confusion matrix, thời gian train.",
        "info",
    )

    # Model selector dropdown
    options = {f"{display} — {desc}": mname
               for mname, display, desc in MODEL_DEFS if mname in all_metrics}

    selected_label = st.selectbox(
        "Chọn model",
        list(options.keys()),
        key="model_detail_selector",
    )
    mname = options[selected_label]
    m     = all_metrics[mname]

    st.divider()

    # Metrics row
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        with st.container(border=True):
            st.metric("Test Accuracy",  f"{m['accuracy']:.2%}",
                      delta=f"{(m['accuracy'] - m['cv_mean']):.2%} vs CV", delta_color="off")
    with k2:
        with st.container(border=True):
            st.metric("CV Mean", f"{m['cv_mean']:.2%}",
                      delta=f"±{m['cv_std']:.2%}", delta_color="off")
    with k3:
        with st.container(border=True):
            st.metric("Số class", str(len(m["classes"])),
                      delta=f"Train: {m['train_samples']:,}", delta_color="off")
    with k4:
        with st.container(border=True):
            st.metric("Thời gian train", f"{m.get('train_time_s', 0):.1f}s",
                      delta=f"{m['n_estimators']} cây", delta_color="off")

    st.divider()

    # Per-class F1 + Confusion matrix side by side
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
            style_chart(fig_cm, 270),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.caption(
        f"📅 Trained: **{m.get('trained_at', '—')}** ·  "
        f"Test samples: **{m['test_samples']:,}** ·  "
        f"CV folds: **{m['cv_folds']}**"
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
            "Chưa có file feature_importance.csv. Chạy `python train_model.py`.",
            "warning",
        )
        return

    st.plotly_chart(
        style_chart(render_feature_importance_chart(fi_df)),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Insights
    top3 = fi_df.head(3)
    bot4 = fi_df.tail(4)

    st.caption("💡 **Nhận xét:**")
    st.caption(
        f"• **Top 3** feature quan trọng nhất: "
        + ", ".join(f"`{r.feature}` ({r.importance:.1%})" for r in top3.itertuples())
        + " — chiếm phần lớn quyết định của model."
    )
    st.caption(
        f"• **4 feature ít quan trọng nhất** (`"
        + "`, `".join(bot4["feature"].tolist())
        + "`) chỉ đóng góp ~"
        + f"{bot4['importance'].sum():.1%} tổng importance — có thể loại bỏ "
        + "nếu cần tối ưu thời gian train mà không ảnh hưởng accuracy đáng kể."
    )