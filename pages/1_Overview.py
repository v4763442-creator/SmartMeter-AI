"""
SmartMeter AI – Executive Overview Page.
Displays macro-level telemetry KPIs, temporal distributions, seasonal trends,
normal vs anomalous readings, and top consumer meters with interactive filters.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Local imports
from src.dashboard_utils import (
    apply_custom_styling,
    render_metric_card,
    load_cached_data,
    load_cached_anomalies
)
from src.visualization import (
    plot_system_consumption_trend,
    plot_hourly_profile,
    plot_monthly_trend,
    plot_top_meters,
    plot_anomaly_timeline
)

# Page configuration
st.set_page_config(page_title="Executive Overview | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-cyan">EXECUTIVE OVERVIEW</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Macro Telemetry & Grid Consumption</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Real-time aggregate consumption trends, diurnal load cycles, and anomaly distributions across all connected smart meters.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
df = load_cached_data()
anomalies_df = load_cached_anomalies()

# Sidebar Filters
st.sidebar.markdown("### **Overview Filters**")

# 1. Date Range Filter
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 2. Meter Selection Filter
all_meters = ["All Meters"] + sorted(df["meter_id"].unique().tolist())
selected_meter = st.sidebar.selectbox("Smart Meter ID", options=all_meters, index=0)

# 3. Consumption Period Filter
period_options = ["All Hours", "Peak Hours (17:00 - 22:00)", "Business Hours (07:00 - 16:00)", "Off-Peak Night (23:00 - 06:00)"]
selected_period = st.sidebar.selectbox("Consumption Period", options=period_options, index=0)

# Apply Filters
filtered_df = df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_d) &
        (filtered_df["timestamp"].dt.date <= end_d)
    ]

if selected_meter != "All Meters":
    filtered_df = filtered_df[filtered_df["meter_id"] == selected_meter]

if selected_period == "Peak Hours (17:00 - 22:00)":
    filtered_df = filtered_df[filtered_df["hour"].between(17, 22)]
elif selected_period == "Business Hours (07:00 - 16:00)":
    filtered_df = filtered_df[filtered_df["hour"].between(7, 16)]
elif selected_period == "Off-Peak Night (23:00 - 06:00)":
    filtered_df = filtered_df[(filtered_df["hour"] >= 23) | (filtered_df["hour"] <= 6)]

if filtered_df.empty:
    st.warning("⚠️ No data available matching the selected filter criteria. Please broaden your selection.")
    st.stop()

# Key Metric Cards
col1, col2, col3, col4, col5, col6 = st.columns(6)

tot_kwh = filtered_df["consumption_kwh"].sum()
tot_mwh = tot_kwh / 1000.0
tot_meters = filtered_df["meter_id"].nunique()
tot_readings = len(filtered_df)
avg_kwh = filtered_df["consumption_kwh"].mean()

# Detected anomalies in filtered set
if "anomaly_flag" in filtered_df.columns:
    tot_anomalies = int(filtered_df["anomaly_flag"].sum())
elif not anomalies_df.empty:
    merged_anom = pd.merge(filtered_df[["meter_id", "timestamp"]], anomalies_df[["meter_id", "timestamp"]], on=["meter_id", "timestamp"], how="inner")
    tot_anomalies = len(merged_anom)
else:
    tot_anomalies = int(filtered_df["is_anomaly"].sum())

anom_pct = (tot_anomalies / tot_readings * 100) if tot_readings > 0 else 0.0

with col1:
    render_metric_card("Total Demand", f"{tot_mwh:,.1f} MWh", f"Across {tot_meters} meter(s)")
with col2:
    render_metric_card("Smart Meters", f"{tot_meters:,}", "Active units")
with col3:
    render_metric_card("Readings", f"{tot_readings:,}", "Filtered telemetry")
with col4:
    render_metric_card("Avg Hourly Demand", f"{avg_kwh:.3f} kWh", "Per hour per meter")
with col5:
    render_metric_card("Detected Anomalies", f"{tot_anomalies:,}", "Model flagged", status="danger" if tot_anomalies > 0 else "normal")
with col6:
    render_metric_card("Anomaly Rate", f"{anom_pct:.2f}%", "Of total readings")

st.markdown("---")

# Visualizations Row 1: System Trend & Hourly Distribution
row1_left, row1_right = st.columns([7, 5])

with row1_left:
    fig_trend = plot_system_consumption_trend(filtered_df)
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_right:
    fig_hourly = plot_hourly_profile(filtered_df)
    st.plotly_chart(fig_hourly, use_container_width=True)

# Visualizations Row 2: Monthly Trends & Top 10 Meters
row2_left, row2_right = st.columns([5, 7])

with row2_left:
    fig_monthly = plot_monthly_trend(filtered_df)
    st.plotly_chart(fig_monthly, use_container_width=True)

with row2_right:
    fig_top = plot_top_meters(filtered_df, top_n=10)
    st.plotly_chart(fig_top, use_container_width=True)

# Visualizations Row 3: Normal vs Anomalous Readings Scatter Timeline
st.markdown("---")
st.markdown("### **Normal vs Anomalous Consumption Timeline**")
if "anomaly_flag" not in filtered_df.columns:
    # Use ground truth or anomalies_df
    if not anomalies_df.empty:
        anom_set = set(zip(anomalies_df["meter_id"], anomalies_df["timestamp"]))
        filtered_df["anomaly_flag"] = filtered_df.apply(lambda r: 1 if (r["meter_id"], r["timestamp"]) in anom_set else 0, axis=1)
        filtered_df["anomaly_type"] = "Flagged Anomaly"
        filtered_df["anomaly_score"] = 0.95
    else:
        filtered_df["anomaly_flag"] = filtered_df["is_anomaly"]
        filtered_df["anomaly_type"] = "Flagged Anomaly"
        filtered_df["anomaly_score"] = 0.95

fig_timeline = plot_anomaly_timeline(filtered_df, sample_n=12000)
st.plotly_chart(fig_timeline, use_container_width=True)
