"""
SmartMeter AI – Demand Forecasting Page.
Provides interactive electricity demand forecasting, comparing Naive, Moving Average,
and Random Forest Regressor models with recursive 24-hour ahead predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Local imports
from src.dashboard_utils import (
    apply_custom_styling,
    render_metric_card,
    load_cached_data,
    load_cached_metrics,
    load_cached_forecasting_model,
    load_cached_metadata
)
from src.forecasting import forecast_next_24h_recursive
from src.feature_engineering import (
    engineer_meter_features,
    get_forecasting_feature_names
)
from src.evaluation import calculate_forecasting_metrics
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_recursive_forecast
)

st.set_page_config(page_title="Demand Forecasting | SmartMeter AI", page_icon="⚡", layout="wide")
apply_custom_styling()

st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-cyan">MACHINE LEARNING FORECASTING</span>
        <h1 style="font-size: 2.1rem; margin-top: 6px; margin-bottom: 4px;">Electricity Demand Forecasting</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Evaluate model accuracy on held-out test periods and generate recursive 24-hour multi-step forecasts for any household.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load cached assets
df = load_cached_data()
rf_model = load_cached_forecasting_model()
global_metrics_df = load_cached_metrics()
metadata = load_cached_metadata()

# Sidebar Selectors
st.sidebar.markdown("### **Forecasting Controls**")
all_meters = sorted(df["meter_id"].unique().tolist())
selected_meter = st.sidebar.selectbox("Select Smart Meter", options=all_meters, index=0)

model_options = [
    "Random Forest Regressor (Trained ML)",
    "Naive Baseline (Previous Day Lag-24)",
    "Moving Average Baseline (Rolling 24h Mean)"
]
selected_model_name = st.sidebar.selectbox("Forecasting Algorithm", options=model_options, index=0)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Hours)", min_value=12, max_value=48, value=24, step=6)

# Feature engineering for the selected meter
meter_df = df[df["meter_id"] == selected_meter].sort_values("timestamp").copy()
feature_cols = get_forecasting_feature_names()
meter_feats = engineer_meter_features(meter_df, drop_na_lags=True)

# Determine train / val / test split timestamp
val_end_time = metadata.get("chronological_split", {}).get("val_end", "2025-11-08 07:00:00")
test_meter_df = meter_feats[meter_feats["timestamp"] > val_end_time].copy()

# Compute predictions on test set for selected meter
y_test_meter = test_meter_df["consumption_kwh"].values

if "Random Forest" in selected_model_name and rf_model is not None:
    y_pred_meter = rf_model.predict(test_meter_df[feature_cols].values)
    algo_key = "Random Forest Regressor"
elif "Naive" in selected_model_name:
    y_pred_meter = test_meter_df["lag_24"].values
    algo_key = "Naive Baseline (lag_24)"
else:
    y_pred_meter = test_meter_df["rolling_mean_24"].values
    algo_key = "Moving Average Baseline (24h)"

# Metrics for the selected meter
meter_metrics = calculate_forecasting_metrics(y_test_meter, y_pred_meter)

# Global test set comparison for context
st.markdown(f"### **Performance for {selected_meter} on Held-Out Test Set**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card("Selected Model", algo_key.split("(")[0].strip(), f"Meter: {selected_meter}")
with col2:
    render_metric_card("Test MAE", f"{meter_metrics['MAE']:.4f} kWh", "Mean Absolute Error")
with col3:
    render_metric_card("Test RMSE", f"{meter_metrics['RMSE']:.4f} kWh", "Root Mean Squared Error")
with col4:
    render_metric_card("Test R² Score", f"{meter_metrics['R2']:.4f}", "Variance explained", status="success" if meter_metrics['R2'] > 0.5 else "normal")

st.markdown("---")

# Main Forecasting Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Recursive 24-Hour Forecast",
    "🔍 Actual vs Predicted (Test Period)",
    "📊 Algorithm Benchmark Comparison"
])

with tab1:
    st.markdown("#### **Recursive Multi-Step Ahead Demand Forecast**")
    
    # Recursive Forecasting Explanation
    st.markdown("""
        <div style="background-color: #161b26; border-left: 4px solid #00d2be; border-radius: 6px; padding: 12px 18px; margin-bottom: 20px;">
            <b style="color: #00d2be;">Methodology: Strict Recursive Forecasting</b><br>
            <span style="color: #94a3b8; font-size: 0.88rem;">
                Unlike single-step evaluations that reuse known future lags, this future forecast predicts hour <i>t+1</i>,
                dynamically updates lag features (<i>lag_1, lag_2</i>) and rolling windows using generated predictions,
                and recursively advances to hour <i>t+24</i>. This guarantees <b>zero future data leakage</b>.
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Generate recursive forecast
    model_for_rec = rf_model if ("Random Forest" in selected_model_name and rf_model is not None) else ("naive" if "Naive" in selected_model_name else "moving_average")
    
    with st.spinner("Computing recursive multi-step forecast..."):
        forecast_df = forecast_next_24h_recursive(
            model=model_for_rec,
            history_df=meter_feats,
            meter_id=selected_meter,
            feature_cols=feature_cols,
            hours_ahead=forecast_horizon
        )

    # Forecast summary KPIs
    tot_fc_kwh = forecast_df["predicted_kwh"].sum()
    peak_fc_row = forecast_df.loc[forecast_df["predicted_kwh"].idxmax()]
    min_fc_row = forecast_df.loc[forecast_df["predicted_kwh"].idxmin()]

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        render_metric_card(f"Total {forecast_horizon}h Forecast", f"{tot_fc_kwh:.2f} kWh", "Projected energy volume")
    with f_col2:
        render_metric_card("Peak Hourly Demand", f"{peak_fc_row['predicted_kwh']:.3f} kWh", f"At {pd.to_datetime(peak_fc_row['timestamp']).strftime('%H:%M')}")
    with f_col3:
        render_metric_card("Min Baseload Demand", f"{min_fc_row['predicted_kwh']:.3f} kWh", f"At {pd.to_datetime(min_fc_row['timestamp']).strftime('%H:%M')}")
    with f_col4:
        render_metric_card("Forecast Horizon", f"+{forecast_horizon} Hours", f"Ending {forecast_df['timestamp'].max().strftime('%b %d, %H:%M')}")

    # Forecast Plot with Confidence Intervals
    fig_fc = plot_recursive_forecast(
        forecast_df=forecast_df,
        recent_history_df=meter_feats,
        meter_id=selected_meter
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Detailed Forecast Table
    with st.expander("📋 View Hourly Forecast Projection Table"):
        display_fc = forecast_df.copy()
        display_fc["timestamp"] = display_fc["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_fc.rename(columns={
            "forecast_step": "Step (+h)",
            "predicted_kwh": "Forecast (kWh)",
            "lower_bound": "95% Lower Bound",
            "upper_bound": "95% Upper Bound",
            "temperature_est": "Est. Temp (°C)"
        }, inplace=True)
        st.dataframe(display_fc, use_container_width=True, hide_index=True)


with tab2:
    st.markdown(f"#### **Test Set Performance: Actual vs Predicted ({selected_meter})**")
    st.markdown(f"<span style='color: #94a3b8;'>Evaluating {len(test_meter_df):,} chronological test observations ({val_end_time} to 2025-12-31).</span>", unsafe_allow_html=True)

    # Zoom window slider
    sample_view_n = st.slider("Display Window (Recent Test Hours)", min_value=48, max_value=min(720, len(test_meter_df)), value=168, step=24)
    slice_test = test_meter_df.tail(sample_view_n)
    slice_actual = y_test_meter[-sample_view_n:]
    slice_pred = y_pred_meter[-sample_view_n:]

    fig_act_pred = plot_actual_vs_predicted(
        timestamps=slice_test["timestamp"],
        y_true=slice_actual,
        y_pred=slice_pred,
        title=f"Actual vs Predicted Demand: {selected_meter} (Last {sample_view_n} Test Hours)"
    )
    st.plotly_chart(fig_act_pred, use_container_width=True)

    # Error distribution
    fig_resid = plot_residual_distribution(y_test_meter, y_pred_meter)
    st.plotly_chart(fig_resid, use_container_width=True)


with tab3:
    st.markdown("#### **Forecasting Algorithm Benchmark Comparison**")
    st.markdown("Comparison across the full test cohort (128,800 out-of-sample observations):")

    if not global_metrics_df.empty:
        st.dataframe(
            global_metrics_df.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 10px;">
                <b>Takeaways:</b>
                <ul>
                    <li><b>Random Forest Regressor</b> significantly outperforms baselines by capturing non-linear diurnal cycles, day-of-week interactions, and temperature elasticity.</li>
                    <li><b>Naive Baseline (lag_24)</b> serves as a strong 24h persistence benchmark, beating simple rolling moving averages during volatile peak hours.</li>
                    <li><b>Moving Average Baseline</b> dampens peak demands, yielding higher errors during rapid spikes.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Pipeline metrics currently being generated. Run python train.py to refresh.")
