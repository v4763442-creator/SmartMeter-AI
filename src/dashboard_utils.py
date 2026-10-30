"""
Dashboard Utilities and Styling Module for SmartMeter AI Streamlit App.
Provides cached loaders, custom CSS injection, KPI cards, and UI helpers.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any, Tuple, Optional


CUSTOM_CSS = """
<style>
    /* Dark SaaS Aesthetic */
    .stApp {
        background-color: #0c1017;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header & Title Adjustments */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #151b26 0%, #1a2232 100%);
        border: 1px solid #243046;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00d2be;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 6px;
        color: #38bdf8;
    }
    .metric-sub.danger {
        color: #f87171;
    }
    .metric-sub.success {
        color: #34d399;
    }

    /* Badges & Pills */
    .badge-cyan {
        background-color: rgba(0, 210, 190, 0.15);
        color: #00d2be;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(0, 210, 190, 0.3);
        display: inline-block;
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.3);
        display: inline-block;
    }

    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #0f141d !important;
        border-right: 1px solid #1f2937 !important;
    }

    /* Tables & DataFrames */
    .stDataFrame {
        border: 1px solid #1f293d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b26;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: #94a3b8;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f293d !important;
        color: #00d2be !important;
        border-bottom: 2px solid #00d2be !important;
    }
</style>
"""


def apply_custom_styling():
    """Injects high-end dark SaaS CSS styling into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    status: str = "normal"
):
    """Renders a styled dark-theme metric card."""
    status_class = f" {status}" if status != "normal" else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub{status_class}">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data(show_spinner=False)
def load_cached_data() -> pd.DataFrame:
    """Loads and caches cleaned smart meter data."""
    path = "data/smart_meter_cleaned.csv"
    if not os.path.exists(path):
        alt = "data/smart_meter_data.csv"
        if os.path.exists(alt):
            path = alt
        else:
            path = "smart_meter_data.csv"

    dtypes = {
        "meter_id": "category",
        "consumption_kwh": "float32",
        "temperature": "float32",
        "hour": "int8",
        "day_of_week": "int8",
        "month": "int8",
        "is_weekend": "int8",
        "is_holiday": "int8",
        "is_anomaly": "int8"
    }
    df = pd.read_csv(path, dtype=dtypes, parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_cached_anomalies() -> pd.DataFrame:
    """Loads detected anomalies summary table."""
    path = "outputs/anomaly_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cached_metrics() -> pd.DataFrame:
    """Loads forecasting model metrics comparison."""
    path = "outputs/model_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cached_feature_importance() -> pd.DataFrame:
    """Loads feature importance table."""
    path = "outputs/feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cached_metadata() -> Dict[str, Any]:
    """Loads training pipeline metadata and scores."""
    path = "models/model_metadata.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def load_cached_forecasting_model():
    """Loads persisted Random Forest forecasting model."""
    path = "models/random_forest_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource(show_spinner=False)
def load_cached_isolation_forest():
    """Loads persisted Isolation Forest anomaly model."""
    path = "models/isolation_forest_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None
