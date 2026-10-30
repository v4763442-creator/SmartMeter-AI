"""
SmartMeter AI – Electricity Demand Forecasting & Intelligent Anomaly Detection
Main Application Entrypoint.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Local imports
from src.dashboard_utils import (
    apply_custom_styling,
    render_metric_card,
    load_cached_data,
    load_cached_anomalies,
    load_cached_metadata
)
from src.visualization import plot_system_consumption_trend, plot_hourly_profile

# Configure page settings
st.set_page_config(
    page_title="SmartMeter AI | Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek dark SaaS styling
apply_custom_styling()

# Sidebar branding & system health
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0; border-bottom: 1px solid #1f293d; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">⚡</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #ffffff;">SMARTMETER AI</h2>
                    <span style="font-size: 0.75rem; color: #00d2be; letter-spacing: 0.08em; font-weight: 600;">ENTERPRISE ANALYTICS</span>
                </div>
            </div>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 8px 0 0 0;">
                Electricity Demand Forecasting & Intelligent Anomaly Detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### **System Status**")
    st.markdown("""
        <div style="background-color: #161b26; border: 1px solid #232936; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; color: #94a3b8;">Data Pipeline</span>
                <span class="badge-cyan">Operational</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; color: #94a3b8;">RandomForest</span>
                <span class="badge-cyan">Trained</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.8rem; color: #94a3b8;">IsolationForest</span>
                <span class="badge-cyan">Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>Dataset:</b> 876,000 Hourly Readings<br>
            <b>Coverage:</b> 100 Smart Meters (2025)<br>
            <b>Frequency:</b> 1 Hour Resolution
        </div>
    """, unsafe_allow_html=True)


# Main View Header
st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="badge-cyan">ENTERPRISE GRID MONITORING</span>
        <h1 style="font-size: 2.3rem; margin-top: 8px; margin-bottom: 4px;">SmartMeter AI Dashboard</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Real-time electricity consumption analysis, machine learning demand forecasting, and unsupervised anomaly detection.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load cached data & metadata
with st.spinner("Connecting to telemetry stream and loading models..."):
    df = load_cached_data()
    anomalies_df = load_cached_anomalies()
    metadata = load_cached_metadata()

# Top KPI metrics row
col1, col2, col3, col4, col5 = st.columns(5)

total_kwh = df["consumption_kwh"].sum()
total_mwh = total_kwh / 1000.0
total_meters = df["meter_id"].nunique()
avg_kwh = df["consumption_kwh"].mean()
total_anomalies = len(anomalies_df) if not anomalies_df.empty else int(df["is_anomaly"].sum())
anom_pct = (total_anomalies / len(df)) * 100

with col1:
    render_metric_card("Total Consumption", f"{total_mwh:,.1f} MWh", "Grid total (2025)")

with col2:
    render_metric_card("Smart Meters", f"{total_meters}", "Connected households")

with col3:
    render_metric_card("Avg Hourly Demand", f"{avg_kwh:.3f} kWh", "Per household mean")

with col4:
    render_metric_card("Telemetry Records", f"{len(df):,}", "Validated readings")

with col5:
    render_metric_card("Anomalies Flagged", f"{total_anomalies:,}", f"{anom_pct:.2f}% anomaly rate", status="danger" if anom_pct > 3 else "normal")


# System-wide preview charts
st.markdown("---")
st.markdown("### **Grid Telemetry & Load Dynamics**")

chart_col1, chart_col2 = st.columns([7, 5])

with chart_col1:
    fig_trend = plot_system_consumption_trend(df)
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    fig_hourly = plot_hourly_profile(df)
    st.plotly_chart(fig_hourly, use_container_width=True)


# Navigation & Feature Highlights
st.markdown("---")
st.markdown("### **Platform Modules**")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">1. Executive Overview</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Macro-level telemetry KPIs, temporal distributions, seasonal patterns, and top consumer rankings.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">2. Demand Forecasting</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Compare Naive, Moving Average, and Random Forest models with 24-hour recursive multi-step forecasting.
            </p>
        </div>
    """, unsafe_allow_html=True)

with nav2:
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">3. Anomaly Detection</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Unsupervised Isolation Forest detection of consumption surges, drops, nighttime spikes, and deviations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">4. Meter Analytics</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Household-level deep-dive: weekly load heatmaps, peak load factors, and cohort benchmark comparisons.
            </p>
        </div>
    """, unsafe_allow_html=True)

with nav3:
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">5. Model Performance</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Comprehensive evaluation metrics (MAE, RMSE, R²), error distributions, feature importance, and confusion matrix.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d2be; margin-top: 0;">6. Data Explorer</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Interactive data table filtering, raw vs cleaned audit statistics, and CSV export capabilities.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.info("👈 Use the navigation sidebar to explore individual analytical modules.")
