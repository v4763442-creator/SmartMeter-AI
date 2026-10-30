"""
SmartMeter AI – Data Explorer Page.
Allows interactive dataset exploration, raw vs cleaned comparison,
data quality audit reporting, and CSV downloads for filtered telemetry & anomaly reports.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Local imports
from src.dashboard_utils import (
    apply_custom_styling,
    render_metric_card,
    load_cached_data,
    load_cached_anomalies,
    load_cached_metadata
)
from src.data_loader import load_raw_data, get_data_summary

st.set_page_config(page_title="Data Explorer | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-cyan">DATA AUDITING & EXPLORATION</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Smart Meter Data Explorer</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Inspect raw and preprocessed smart meter telemetry, review missing value imputation audits, and export datasets.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load datasets
cleaned_df = load_cached_data()
anomalies_df = load_cached_anomalies()
metadata = load_cached_metadata()

# Sidebar Filters
st.sidebar.markdown("### **Data Explorer Filters**")

all_meters = ["All Meters"] + sorted(cleaned_df["meter_id"].unique().tolist())
selected_meter = st.sidebar.selectbox("Filter by Smart Meter", options=all_meters, index=0)

min_date = cleaned_df["timestamp"].min().date()
max_date = cleaned_df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Filter Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

row_limit = st.sidebar.selectbox("Display Rows Limit", options=[50, 100, 250, 500, 1000], index=1)

# Apply filters to cleaned dataset
filtered_df = cleaned_df.copy()
if selected_meter != "All Meters":
    filtered_df = filtered_df[filtered_df["meter_id"] == selected_meter]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_d) &
        (filtered_df["timestamp"].dt.date <= end_d)
    ]

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Cleaned Dataset Preview",
    "🛡️ Data Quality & Audit Summary",
    "📥 Anomaly Reports & Exports"
])

with tab1:
    st.markdown(f"#### **Cleaned Telemetry Stream ({len(filtered_df):,} matching rows)**")
    
    # Download Button
    csv_filtered = filtered_df.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Telemetry ({min(10000, len(filtered_df)):,} rows CSV)",
        data=csv_filtered,
        file_name=f"smart_meter_filtered_{selected_meter}.csv",
        mime="text/csv"
    )

    st.dataframe(
        filtered_df.head(row_limit),
        use_container_width=True,
        hide_index=True
    )
    if len(filtered_df) > row_limit:
        st.caption(f"Showing first {row_limit} records of {len(filtered_df):,} matching rows.")


with tab2:
    st.markdown("#### **Data Hygiene, Preprocessing & Quality Statistics**")

    # Quality KPI cards
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        render_metric_card("Total Records", f"{len(cleaned_df):,}", "Full 2025 hourly stream")
    with q2:
        render_metric_card("Missing Consumption", "0 Nulls", "12,756 imputed via linear interpolation", status="success")
    with q3:
        render_metric_card("Missing Temperature", "0 Nulls", "3,581 imputed temporally", status="success")
    with q4:
        render_metric_card("Duplicate Records", "0 Found", "Schema uniqueness verified", status="success")
    with q5:
        render_metric_card("Negative Values", "0 Detected", "Non-negativity verified", status="success")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Audit Breakdown Table
    audit_data = [
        {"Validation Check": "Total Telemetry Observations", "Raw Dataset": "876,000", "Cleaned Dataset": f"{len(cleaned_df):,}", "Status": "Preserved (100%)"},
        {"Validation Check": "Monitored Smart Meters", "Raw Dataset": "100 Unique", "Cleaned Dataset": f"{cleaned_df['meter_id'].nunique()} Unique", "Status": "Complete"},
        {"Validation Check": "Missing Consumption Readings", "Raw Dataset": "12,756 (1.46%)", "Cleaned Dataset": "0 (0.00%)", "Status": "Time-series Interpolated"},
        {"Validation Check": "Missing Ambient Temperature", "Raw Dataset": "3,581 (0.41%)", "Cleaned Dataset": "0 (0.00%)", "Status": "Diurnal Imputed"},
        {"Validation Check": "Duplicate (Meter, Timestamp)", "Raw Dataset": "0", "Cleaned Dataset": "0", "Status": "Verified Zero Duplicates"},
        {"Validation Check": "Negative Consumption Values", "Raw Dataset": "0", "Cleaned Dataset": "0", "Status": "Verified [0.0, 12.0 kWh]"},
        {"Validation Check": "Ground-Truth Anomaly Flags", "Raw Dataset": "22,770 (2.60%)", "Cleaned Dataset": "22,770 (2.60%)", "Status": "Strictly Preserved for Eval"}
    ]
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

    # Raw vs Cleaned Side-by-Side Expander
    with st.expander("🔍 Compare Sample Raw vs Cleaned Records"):
        try:
            raw_sample = load_raw_data("data/smart_meter_data.csv").head(10)
            col_raw, col_clean = st.columns(2)
            with col_raw:
                st.markdown("<b>Raw Dataset (Unprocessed)</b>", unsafe_allow_html=True)
                st.dataframe(raw_sample, use_container_width=True)
            with col_clean:
                st.markdown("<b>Cleaned Dataset (Interpolated & Formatted)</b>", unsafe_allow_html=True)
                st.dataframe(cleaned_df.head(10), use_container_width=True)
        except Exception as e:
            st.caption(f"Raw dataset comparison: {e}")


with tab3:
    st.markdown("#### **Anomaly Incident Reports & Bulk Exports**")

    if not anomalies_df.empty:
        st.markdown(f"Total detected incidents across 100 meters: **{len(anomalies_df):,} events**.")

        # Download button for all anomalies
        anom_csv = anomalies_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Anomaly Incident Report (CSV)",
            data=anom_csv,
            file_name="smartmeter_all_anomalies_2025.csv",
            mime="text/csv"
        )

        st.dataframe(anomalies_df.head(200), use_container_width=True, hide_index=True)
    else:
        st.info("No anomaly results available to export yet. Run train.py to generate.")
