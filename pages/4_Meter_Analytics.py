"""
SmartMeter AI – Meter Analytics Page.
Provides household-level deep-dive analytics: consumption history,
hourly diurnal cycles, weekly load heatmaps, monthly volumes,
and cohort benchmark comparisons.
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
    load_cached_anomalies
)
from src.visualization import (
    plot_meter_weekly_heatmap,
    THEME
)

st.set_page_config(page_title="Meter Analytics | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-cyan">HOUSEHOLD INTELLIGENCE</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Smart Meter Analytics & Benchmarking</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Granular consumption profiling, appliance duty signatures, weekly usage heatmaps, and peer cohort comparisons.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
df = load_cached_data()
anomalies_df = load_cached_anomalies()

# Sidebar: Meter Selector
st.sidebar.markdown("### **Meter Selection**")
all_meters = sorted(df["meter_id"].unique().tolist())
selected_meter = st.sidebar.selectbox("Choose Smart Meter", options=all_meters, index=0)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input("Filter Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter for selected meter
meter_df = df[df["meter_id"] == selected_meter].sort_values("timestamp").copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    meter_df = meter_df[
        (meter_df["timestamp"].dt.date >= start_d) &
        (meter_df["timestamp"].dt.date <= end_d)
    ]

if meter_df.empty:
    st.warning("No data found for the selected date range.")
    st.stop()

# Compute Cohort vs Meter Statistics
meter_total_kwh = meter_df["consumption_kwh"].sum()
meter_avg_kwh = meter_df["consumption_kwh"].mean()
meter_peak_kwh = meter_df["consumption_kwh"].max()
meter_min_kwh = meter_df["consumption_kwh"].min()
meter_peak_row = meter_df.loc[meter_df["consumption_kwh"].idxmax()]
load_factor = (meter_avg_kwh / meter_peak_kwh) if meter_peak_kwh > 0 else 0.0

# Overall Cohort benchmark across all meters for the same date window
cohort_window = df[
    (df["timestamp"].dt.date >= meter_df["timestamp"].dt.date.min()) &
    (df["timestamp"].dt.date <= meter_df["timestamp"].dt.date.max())
]
cohort_avg_kwh = cohort_window["consumption_kwh"].mean()
pct_diff_cohort = ((meter_avg_kwh - cohort_avg_kwh) / cohort_avg_kwh) * 100

# Anomalies for this meter
if not anomalies_df.empty:
    meter_anom_df = anomalies_df[anomalies_df["meter_id"] == selected_meter].copy()
else:
    meter_anom_df = meter_df[meter_df["is_anomaly"] == 1].copy()
tot_meter_anomalies = len(meter_anom_df)
anom_rate_meter = (tot_meter_anomalies / len(meter_df)) * 100 if len(meter_df) > 0 else 0.0

# KPI Metric Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    render_metric_card("Total Consumption", f"{meter_total_kwh:,.1f} kWh", f"{meter_total_kwh/1000.0:.2f} MWh")
with kpi2:
    diff_sign = "+" if pct_diff_cohort >= 0 else ""
    render_metric_card(
        "Avg Hourly Demand",
        f"{meter_avg_kwh:.3f} kWh",
        f"{diff_sign}{pct_diff_cohort:.1f}% vs cohort ({cohort_avg_kwh:.3f})",
        status="danger" if pct_diff_cohort > 20 else ("success" if pct_diff_cohort < -10 else "normal")
    )
with kpi3:
    render_metric_card("Peak Recorded Load", f"{meter_peak_kwh:.2f} kWh", f"At {pd.to_datetime(meter_peak_row['timestamp']).strftime('%b %d %H:%M')}")
with kpi4:
    render_metric_card("Load Factor", f"{load_factor:.2f}", "Avg load / Peak load (Grid efficiency)")
with kpi5:
    render_metric_card("Detected Anomalies", f"{tot_meter_anomalies}", f"{anom_rate_meter:.2f}% anomaly rate", status="danger" if tot_meter_anomalies > 10 else "normal")

st.markdown("---")

# Visualizations Row 1: Time Series History with Rolling Average
st.markdown("### **Consumption History & Baseline Trend**")

# Rolling 24h average for smoothed trend visualization
meter_df["rolling_view_24"] = meter_df["consumption_kwh"].rolling(24, min_periods=1).mean()

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=meter_df["timestamp"],
    y=meter_df["consumption_kwh"],
    mode="lines",
    name="Hourly Consumption",
    line=dict(color="#334155", width=1.0),
    opacity=0.6
))
fig_ts.add_trace(go.Scatter(
    x=meter_df["timestamp"],
    y=meter_df["rolling_view_24"],
    mode="lines",
    name="24h Rolling Trend",
    line=dict(color=THEME["primary"], width=2.5)
))

# Overlay detected anomalies if any
if not meter_anom_df.empty:
    fig_ts.add_trace(go.Scatter(
        x=meter_anom_df["timestamp"],
        y=meter_anom_df["consumption_kwh"],
        mode="markers",
        name="Detected Anomaly",
        marker=dict(color=THEME["danger"], size=7, symbol="diamond")
    ))

fig_ts.update_layout(
    title=f"<b>Electricity Consumption History: {selected_meter}</b>",
    paper_bgcolor=THEME["card_bg"],
    plot_bgcolor=THEME["card_bg"],
    font=dict(color=THEME["muted_text"]),
    height=400,
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=THEME["grid_color"], title="Timestamp"),
    yaxis=dict(gridcolor=THEME["grid_color"], title="Consumption (kWh)")
)
st.plotly_chart(fig_ts, use_container_width=True)

# Visualizations Row 2: Weekly Heatmap & Hourly Benchmark
row2_left, row2_right = st.columns([6, 6])

with row2_left:
    fig_heat = plot_meter_weekly_heatmap(meter_df)
    st.plotly_chart(fig_heat, use_container_width=True)

with row2_right:
    # Hourly Profile: Selected Meter vs Cohort Average
    meter_hourly = meter_df.groupby("hour")["consumption_kwh"].mean().reset_index()
    cohort_hourly = cohort_window.groupby("hour")["consumption_kwh"].mean().reset_index()

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=meter_hourly["hour"],
        y=meter_hourly["consumption_kwh"],
        mode="lines+markers",
        name=f"{selected_meter}",
        line=dict(color=THEME["primary"], width=2.8),
        marker=dict(size=6)
    ))
    fig_comp.add_trace(go.Scatter(
        x=cohort_hourly["hour"],
        y=cohort_hourly["consumption_kwh"],
        mode="lines",
        name="Cohort Benchmark (All Meters)",
        line=dict(color="#94a3b8", width=2.0, dash="dash")
    ))

    fig_comp.update_layout(
        title=f"<b>Diurnal Profile vs Cohort Average</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=THEME["grid_color"], title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(gridcolor=THEME["grid_color"], title="Average Consumption (kWh)")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# Visualizations Row 3: Day of Week & Monthly Volumes
row3_left, row3_right = st.columns([6, 6])

with row3_left:
    days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_df = meter_df.groupby("day_of_week")["consumption_kwh"].mean().reset_index()
    dow_df["day_name"] = dow_df["day_of_week"].apply(lambda d: days_labels[d])

    fig_dow = go.Figure(go.Bar(
        x=dow_df["day_name"],
        y=dow_df["consumption_kwh"],
        marker_color=THEME["accent"],
        hovertemplate="%{x}: %{y:.3f} kWh<extra></extra>"
    ))
    fig_dow.update_layout(
        title="<b>Average Consumption by Day of Week</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=THEME["grid_color"], title="Day of Week"),
        yaxis=dict(gridcolor=THEME["grid_color"], title="Avg Consumption (kWh)")
    )
    st.plotly_chart(fig_dow, use_container_width=True)

with row3_right:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_meter = meter_df.groupby("month")["consumption_kwh"].sum().reset_index()
    monthly_meter["month_name"] = monthly_meter["month"].apply(lambda m: month_names[int(m) - 1] if 1 <= m <= 12 else str(m))

    fig_mon = go.Figure(go.Bar(
        x=monthly_meter["month_name"],
        y=monthly_meter["consumption_kwh"],
        marker_color=THEME["secondary"],
        hovertemplate="%{x}: %{y:.1f} kWh<extra></extra>"
    ))
    fig_mon.update_layout(
        title="<b>Total Monthly Consumption (kWh)</b>",
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font=dict(color=THEME["muted_text"]),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=THEME["grid_color"], title="Month"),
        yaxis=dict(gridcolor=THEME["grid_color"], title="Total Consumption (kWh)")
    )
    st.plotly_chart(fig_mon, use_container_width=True)

# Incident Audit for this Meter
if not meter_anom_df.empty:
    with st.expander(f"⚠️ View Detected Anomaly Incidents for {selected_meter} ({tot_meter_anomalies} Events)"):
        st.dataframe(
            meter_anom_df[[
                "timestamp", "consumption_kwh", "rolling_mean_24",
                "anomaly_score", "anomaly_type", "severity"
            ]].rename(columns={
                "timestamp": "Timestamp",
                "consumption_kwh": "Consumption (kWh)",
                "rolling_mean_24": "Rolling 24h Baseline",
                "anomaly_score": "Score",
                "anomaly_type": "Anomaly Type",
                "severity": "Severity"
            }),
            use_container_width=True,
            hide_index=True
        )
