"""
SmartMeter AI – Model Performance Page.
Comprehensive machine learning diagnostics:
- Forecasting models benchmark (MAE, RMSE, R2 comparison)
- Residual distributions & parity plots
- Random Forest feature importances
- Isolation Forest evaluation against ground-truth (Precision, Recall, F1, Confusion Matrix)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from src.dashboard_utils import (
    apply_custom_styling,
    render_metric_card,
    load_cached_metrics,
    load_cached_feature_importance,
    load_cached_metadata,
    load_cached_data,
    load_cached_forecasting_model
)
from src.visualization import (
    plot_feature_importance,
    plot_confusion_matrix_heatmap,
    THEME
)

st.set_page_config(page_title="Model Performance | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-cyan">MODEL AUDITING & BENCHMARKS</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Machine Learning Performance Evaluation</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Rigorous evaluation of time-series forecasting models and unsupervised anomaly isolation against held-out ground truth.
        </p>
    </div>
""", unsafe_allow_html=True)

metrics_df = load_cached_metrics()
feat_imp_df = load_cached_feature_importance()
metadata = load_cached_metadata()

# Check availability
if metrics_df.empty or not metadata:
    st.info("ℹ️ Pipeline performance artifacts are currently being generated. Run 'python train.py' to populate.")
    st.stop()

# -------------------------------------------------------------
# SECTION 1: Forecasting Models Benchmark
# -------------------------------------------------------------
st.markdown("### **1. Electricity Demand Forecasting Comparison**")

# Top KPI comparison cards for Champion Model
rf_row = metrics_df[metrics_df["Model"].str.contains("Random Forest")].iloc[0] if any(metrics_df["Model"].str.contains("Random Forest")) else metrics_df.iloc[-1]
naive_row = metrics_df[metrics_df["Model"].str.contains("Naive")].iloc[0] if any(metrics_df["Model"].str.contains("Naive")) else metrics_df.iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Champion Model", "Random Forest", "Supervised Ensemble")
with c2:
    mae_diff = ((naive_row["MAE"] - rf_row["MAE"]) / naive_row["MAE"]) * 100
    render_metric_card("Test MAE", f"{rf_row['MAE']:.4f} kWh", f"{mae_diff:.1f}% reduction vs Naive", status="success")
with c3:
    rmse_diff = ((naive_row["RMSE"] - rf_row["RMSE"]) / naive_row["RMSE"]) * 100
    render_metric_card("Test RMSE", f"{rf_row['RMSE']:.4f} kWh", f"{rmse_diff:.1f}% reduction vs Naive", status="success")
with c4:
    render_metric_card("Test R² Score", f"{rf_row['R2']:.4f}", f"Naive R²: {naive_row['R2']:.4f}", status="success")

st.markdown("<br>", unsafe_allow_html=True)

# Comparative Bar Charts: MAE, RMSE, R2
bar1, bar2, bar3 = st.columns(3)

with bar1:
    fig_mae = go.Figure(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["MAE"],
        marker_color=["#94a3b8", "#38bdf8", "#00d2be"],
        text=[f"{v:.4f}" for v in metrics_df["MAE"]],
        textposition="auto"
    ))
    fig_mae.update_layout(
        title="<b>Mean Absolute Error (MAE) - Lower is Better</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=320,
        margin=dict(l=30, r=20, t=50, b=40),
        yaxis=dict(gridcolor=THEME["grid_color"], title="MAE (kWh)")
    )
    st.plotly_chart(fig_mae, use_container_width=True)

with bar2:
    fig_rmse = go.Figure(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["RMSE"],
        marker_color=["#94a3b8", "#38bdf8", "#00d2be"],
        text=[f"{v:.4f}" for v in metrics_df["RMSE"]],
        textposition="auto"
    ))
    fig_rmse.update_layout(
        title="<b>Root Mean Squared Error (RMSE) - Lower is Better</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=320,
        margin=dict(l=30, r=20, t=50, b=40),
        yaxis=dict(gridcolor=THEME["grid_color"], title="RMSE (kWh)")
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with bar3:
    fig_r2 = go.Figure(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["R2"],
        marker_color=["#94a3b8", "#38bdf8", "#00d2be"],
        text=[f"{v:.4f}" for v in metrics_df["R2"]],
        textposition="auto"
    ))
    fig_r2.update_layout(
        title="<b>R² Explained Variance - Higher is Better</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=320,
        margin=dict(l=30, r=20, t=50, b=40),
        yaxis=dict(gridcolor=THEME["grid_color"], title="R² Score")
    )
    st.plotly_chart(fig_r2, use_container_width=True)

# Comparison Data Table
st.dataframe(
    metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# -------------------------------------------------------------
# SECTION 2: Feature Importance (Random Forest)
# -------------------------------------------------------------
st.markdown("### **2. Random Forest Feature Importance**")

col_feat1, col_feat2 = st.columns([7, 5])

with col_feat1:
    if not feat_imp_df.empty:
        fig_imp = plot_feature_importance(feat_imp_df, top_n=15)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance data not found.")

with col_feat2:
    st.markdown("""
        <div style="background-color: #161b26; border: 1px solid #232936; border-radius: 8px; padding: 18px; margin-top: 10px;">
            <h4 style="color: #00d2be; margin-top: 0;">Predictive Dynamics Analysis</h4>
            <ul style="color: #94a3b8; font-size: 0.88rem; line-height: 1.6; padding-left: 20px;">
                <li><b>Immediate Autoregressive Signals (lag_1, lag_2):</b> Exhibit the highest predictive power, capturing instantaneous household inertia and active appliance states.</li>
                <li><b>Diurnal Baseline (rolling_mean_24 & lag_24):</b> Anchor the predictions to the household's cyclical 24-hour routine and seasonal baseload.</li>
                <li><b>Temporal Encodings (hour, day_of_week):</b> Accurately partition peak evening demand surges from nighttime standby baseloads.</li>
                <li><b>Exogenous Weather (temperature):</b> Provides elasticity adjustments during winter heating and summer cooling spikes.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------------
# SECTION 3: Anomaly Detection Evaluation (Isolation Forest)
# -------------------------------------------------------------
st.markdown("### **3. Isolation Forest Evaluation vs Ground Truth**")

iso_meta = metadata.get("anomaly_detection", {})
test_eval = iso_meta.get("test_evaluation", {})

if test_eval:
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_metric_card("Precision", f"{test_eval.get('precision', 0):.4f}", "True Anom / Pred Anom")
    with p2:
        render_metric_card("Recall (Sensitivity)", f"{test_eval.get('recall', 0):.4f}", "True Anom / Actual Anom")
    with p3:
        render_metric_card("F1 Score", f"{test_eval.get('f1_score', 0):.4f}", "Harmonic Mean", status="success" if test_eval.get('f1_score', 0) > 0.6 else "normal")
    with p4:
        tp = test_eval.get("true_positives", 0)
        fp = test_eval.get("false_positives", 0)
        render_metric_card("True Positives", f"{tp:,}", f"False Positives: {fp:,}")

    col_cm, col_comment = st.columns([6, 6])

    with col_cm:
        cm_data = test_eval.get("confusion_matrix", [[0, 0], [0, 0]])
        fig_cm = plot_confusion_matrix_heatmap(cm_data)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_comment:
        st.markdown(f"""
            <div style="background-color: #161b26; border: 1px solid #232936; border-radius: 8px; padding: 18px; margin-top: 10px;">
                <h4 style="color: #38bdf8; margin-top: 0;">Unsupervised Anomaly Auditing Insights</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.6;">
                    The Isolation Forest was fitted completely <b>unsupervised</b> (without access to ground-truth anomaly flags).
                    The contamination threshold was set to <code>contamination = {iso_meta.get('contamination', 0.025)}</code>.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>True Negatives:</b> {test_eval.get('true_negatives', 0):,} routine readings correctly flagged as normal.</li>
                    <li><b>True Positives:</b> {tp:,} genuine irregularities captured solely via multidimensional path isolation.</li>
                    <li><b>False Negatives:</b> {test_eval.get('false_negatives', 0):,} subtle deviations masked by routine behavioral noise.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("Anomaly evaluation metrics not available. Run train.py to refresh.")
