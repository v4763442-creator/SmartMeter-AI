"""
SmartMeter AI – Anomaly Detection Page.
Provides unsupervised anomaly detection analytics using Isolation Forest:
behavioral classification, severity distributions, temporal timelines,
and exportable anomaly incident audits.
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
    load_cached_data,
    load_cached_anomalies,
    load_cached_metadata
)
from src.visualization import (
    plot_anomaly_timeline,
    plot_anomaly_types,
    THEME
)

st.set_page_config(page_title="Anomaly Detection | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-amber">INTELLIGENT ANOMALY DETECTION</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Consumption Anomaly Detection</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Unsupervised Isolation Forest detection of consumption surges, drops, unexpected nighttime draws, and erratic household load shifts.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
df = load_cached_data()
anomalies_df = load_cached_anomalies()
metadata = load_cached_metadata()

if anomalies_df.empty:
    st.info("ℹ️ Anomaly results are being computed by the central pipeline. Run 'python train.py' to generate.")
    st.stop()

# Ensure timestamp format
anomalies_df["timestamp"] = pd.to_datetime(anomalies_df["timestamp"])

# Sidebar Filters
st.sidebar.markdown("### **Anomaly Filters**")

# Meter Filter
meters_list = ["All Meters"] + sorted(anomalies_df["meter_id"].unique().tolist())
selected_meter = st.sidebar.selectbox("Filter by Meter ID", options=meters_list, index=0)

# Date Filter
min_date = anomalies_df["timestamp"].min().date()
max_date = anomalies_df["timestamp"].max().date()
date_range = st.sidebar.date_input("Filter Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Anomaly Type Filter
available_types = sorted(anomalies_df["anomaly_type"].unique().tolist())
selected_types = st.sidebar.multiselect("Anomaly Classification", options=available_types, default=available_types)

# Severity Filter
available_severities = ["High", "Medium", "Low"]
selected_severities = st.sidebar.multiselect("Severity Level", options=available_severities, default=available_severities)

# Minimum Anomaly Score Filter
min_score = st.sidebar.slider("Min Anomaly Score Threshold", min_value=0.50, max_value=0.99, value=0.60, step=0.01)

# Apply filtering
filtered_anom = anomalies_df.copy()

if selected_meter != "All Meters":
    filtered_anom = filtered_anom[filtered_anom["meter_id"] == selected_meter]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_anom = filtered_anom[
        (filtered_anom["timestamp"].dt.date >= start_d) &
        (filtered_anom["timestamp"].dt.date <= end_d)
    ]

if selected_types:
    filtered_anom = filtered_anom[filtered_anom["anomaly_type"].isin(selected_types)]

if selected_severities:
    filtered_anom = filtered_anom[filtered_anom["severity"].isin(selected_severities)]

filtered_anom = filtered_anom[filtered_anom["anomaly_score"] >= min_score]

# Top KPI Metric Cards
total_dataset_rows = metadata.get("total_records", len(df))
tot_flagged = len(filtered_anom)
high_sev_count = (filtered_anom["severity"] == "High").sum()
impacted_meters = filtered_anom["meter_id"].nunique()
overall_rate = (len(anomalies_df) / total_dataset_rows) * 100

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    render_metric_card("Total Flagged", f"{tot_flagged:,}", f"Matching filters", status="danger" if tot_flagged > 0 else "normal")
with kpi2:
    render_metric_card("Overall Anomaly Rate", f"{overall_rate:.2f}%", "Unsupervised contamination: 2.5%")
with kpi3:
    render_metric_card("High-Severity Events", f"{high_sev_count:,}", "Score ≥ 0.85", status="danger" if high_sev_count > 0 else "normal")
with kpi4:
    render_metric_card("Impacted Meters", f"{impacted_meters}", f"Out of {df['meter_id'].nunique()} monitored")
with kpi5:
    avg_score = filtered_anom["anomaly_score"].mean() if not filtered_anom.empty else 0.0
    render_metric_card("Average Score", f"{avg_score:.3f}", "Mean anomaly index")

st.markdown("---")

# Visualizations Row 1: Timeline & Classification Breakdown
col_chart1, col_chart2 = st.columns([7, 5])

with col_chart1:
    # Full or sample context timeline
    if not filtered_anom.empty:
        # Construct timeline dataframe
        sample_context = df[df["meter_id"].isin(filtered_anom["meter_id"].unique()[:10])].copy()
        sample_context = sample_context.sample(n=min(10000, len(sample_context)), random_state=42)
        sample_context["anomaly_flag"] = 0
        sample_context["anomaly_type"] = "Normal"
        sample_context["anomaly_score"] = 0.0

        plot_anom = filtered_anom.copy()
        plot_anom["anomaly_flag"] = 1

        combined_plot = pd.concat([
            sample_context[["timestamp", "consumption_kwh", "meter_id", "anomaly_flag", "anomaly_type", "anomaly_score"]],
            plot_anom[["timestamp", "consumption_kwh", "meter_id", "anomaly_flag", "anomaly_type", "anomaly_score"]]
        ]).sort_values("timestamp")

        fig_tl = plot_anomaly_timeline(combined_plot, sample_n=15000)
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("No anomalies match the selected filters.")

with col_chart2:
    if not filtered_anom.empty:
        fig_types = plot_anomaly_types(filtered_anom)
        st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.info("No classification breakdown available for current filters.")

# Visualizations Row 2: Top Anomalous Meters & Severity Breakdown
col_chart3, col_chart4 = st.columns([6, 6])

with col_chart3:
    if not filtered_anom.empty:
        meter_counts = filtered_anom["meter_id"].value_counts().reset_index()
        meter_counts.columns = ["meter_id", "count"]
        top_anom_meters = meter_counts.head(10).sort_values("count", ascending=True)

        fig_m_bar = go.Figure(go.Bar(
            x=top_anom_meters["count"],
            y=top_anom_meters["meter_id"],
            orientation="h",
            marker=dict(color="#ef4444"),
            hovertemplate="Meter %{y}: %{x} anomalies<extra></extra>"
        ))
        fig_m_bar.update_layout(
            title="<b>Top 10 Meters with Most Detected Anomalies</b>",
            paper_bgcolor=THEME["card_bg"],
            plot_bgcolor=THEME["card_bg"],
            font=dict(color=THEME["muted_text"]),
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(gridcolor=THEME["grid_color"], title="Anomaly Count"),
            yaxis=dict(gridcolor=THEME["grid_color"], title="Smart Meter")
        )
        st.plotly_chart(fig_m_bar, use_container_width=True)

with col_chart4:
    if not filtered_anom.empty:
        sev_counts = filtered_anom["severity"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        sev_counts.columns = ["severity", "count"]
        fig_sev = go.Figure(go.Bar(
            x=sev_counts["severity"],
            y=sev_counts["count"],
            marker=dict(color=["#ef4444", "#f59e0b", "#38bdf8"]),
            hovertemplate="Severity %{x}: %{y} incidents<extra></extra>"
        ))
        fig_sev.update_layout(
            title="<b>Anomaly Incidents by Severity Level</b>",
            paper_bgcolor=THEME["card_bg"],
            plot_bgcolor=THEME["card_bg"],
            font=dict(color=THEME["muted_text"]),
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(gridcolor=THEME["grid_color"], title="Severity"),
            yaxis=dict(gridcolor=THEME["grid_color"], title="Count")
        )
        st.plotly_chart(fig_sev, use_container_width=True)

# Incident Audit Table & CSV Export
st.markdown("---")
st.markdown("### **Anomaly Incident Audit Table**")

if not filtered_anom.empty:
    display_table = filtered_anom.copy()
    display_table["timestamp_str"] = display_table["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    
    cols_order = [
        "meter_id", "timestamp_str", "consumption_kwh", "rolling_mean_24",
        "anomaly_score", "anomaly_type", "severity"
    ]
    if "is_anomaly" in display_table.columns:
        cols_order.append("is_anomaly")

    st.dataframe(
        display_table[cols_order].rename(columns={
            "meter_id": "Meter ID",
            "timestamp_str": "Timestamp",
            "consumption_kwh": "Consumption (kWh)",
            "rolling_mean_24": "Rolling 24h Mean (kWh)",
            "anomaly_score": "Anomaly Score",
            "anomaly_type": "Classification",
            "severity": "Severity",
            "is_anomaly": "Ground Truth"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv_bytes = filtered_anom.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Anomaly Incidents (CSV)",
        data=csv_bytes,
        file_name=f"smartmeter_anomalies_{selected_meter}.csv",
        mime="text/csv"
    )
else:
    st.info("No records to display.")

# Important Domain Note
st.markdown("""
    <div style="background-color: #161b26; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 14px 18px; margin-top: 24px;">
        <b style="color: #38bdf8;">Operational Context: Statistical Anomaly vs Confirmed Irregularity</b><br>
        <span style="color: #94a3b8; font-size: 0.88rem;">
            Unsupervised Isolation Forest isolates points that deviate from multidimensional multivariate distributions.
            Not all statistical anomalies represent illicit activity or meter tampering; typical legitimate causes include
            ev charging sessions, heat pump or air conditioning turn-ons during temperature extremes, temporary household vacancies,
            or brief grid-side power quality events. Ground-truth cross-validation evaluates detection fidelity against known benchmarks.
        </span>
    </div>
""", unsafe_allow_html=True)
