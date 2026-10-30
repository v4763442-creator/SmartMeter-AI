"""
Electricity Demand Forecasting Module for SmartMeter AI.
Implements Naive Baseline, Moving Average Baseline, Random Forest Regressor,
and Recursive Multi-Step 24-Hour Ahead Forecasting.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import calculate_forecasting_metrics


def split_time_series_chronological(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits the dataset chronologically across all smart meters without shuffling.
    - First 70%: Training
    - Next 15%: Validation
    - Final 15%: Testing
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_timestamps = np.sort(df["timestamp"].unique())
    n_timestamps = len(unique_timestamps)

    train_end_idx = int(n_timestamps * train_pct)
    val_end_idx = int(n_timestamps * (train_pct + val_pct))

    train_end_time = unique_timestamps[train_end_idx]
    val_end_time = unique_timestamps[val_end_idx]

    train_df = df[df["timestamp"] <= train_end_time].copy()
    val_df = df[(df["timestamp"] > train_end_time) & (df["timestamp"] <= val_end_time)].copy()
    test_df = df[df["timestamp"] > val_end_time].copy()

    split_info = {
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "train_rows": len(train_df),
        "val_start": str(val_df["timestamp"].min()),
        "val_end": str(val_df["timestamp"].max()),
        "val_rows": len(val_df),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "test_rows": len(test_df),
    }

    return train_df, val_df, test_df, split_info


def predict_naive_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Model 1: Naive Baseline
    Predicts current consumption using the previous day's corresponding hourly consumption (lag_24).
    """
    return df["lag_24"].to_numpy(dtype=np.float32)


def predict_moving_average_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Model 2: Moving Average Baseline
    Predicts current consumption using historical rolling 24h average.
    """
    return df["rolling_mean_24"].to_numpy(dtype=np.float32)


def tune_and_train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "consumption_kwh",
    candidate_depths: Optional[List[int]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    subsample_for_tuning: int = 150000
) -> Tuple[RandomForestRegressor, Dict[str, Any], int]:
    """
    Validates max_depth on validation set and trains the final RandomForestRegressor.
    Uses candidate depths and picks the depth with lowest validation RMSE.
    """
    if candidate_depths is None:
        candidate_depths = [12, 16, 20]

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    # If training set is very large, select best depth using stratified chronological sample
    if len(train_df) > subsample_for_tuning:
        step = max(1, len(train_df) // subsample_for_tuning)
        X_train_tune = X_train_full[::step]
        y_train_tune = y_train_full[::step]
    else:
        X_train_tune = X_train_full
        y_train_tune = y_train_full

    tuning_results = {}
    best_depth = candidate_depths[0]
    best_val_rmse = float("inf")

    for depth in candidate_depths:
        rf_tune = RandomForestRegressor(
            n_estimators=50,
            max_depth=depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf_tune.fit(X_train_tune, y_train_tune)
        preds_val = rf_tune.predict(X_val)
        val_metrics = calculate_forecasting_metrics(y_val, preds_val)
        tuning_results[f"depth_{depth}"] = val_metrics
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_depth = depth

    # Train final model on full training set with best depth and n_estimators=100
    final_rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=best_depth,
        random_state=random_state,
        n_jobs=-1
    )
    final_rf.fit(X_train_full, y_train_full)

    val_preds = final_rf.predict(X_val)
    val_metrics = calculate_forecasting_metrics(y_val, val_preds)

    return final_rf, {
        "best_max_depth": best_depth,
        "tuning_results": tuning_results,
        "final_val_metrics": val_metrics
    }, best_depth


def forecast_next_24h_recursive(
    model: Any,
    history_df: pd.DataFrame,
    meter_id: str,
    feature_cols: List[str],
    hours_ahead: int = 24
) -> pd.DataFrame:
    """
    Generates a 24-hour electricity demand forecast using recursive multi-step forecasting.
    
    METHODOLOGY:
    At each step t + h (h = 1..24):
    1. Lag features and rolling statistics are dynamically updated using predictions from prior steps
       (e.g., lag_1 at step 2 uses the forecast from step 1).
    2. Calendar features are computed from the forecast timestamp.
    3. Temperature is projected using the diurnal profile of the preceding day.
    4. The model predicts consumption for step h.
    5. The forecast is appended to the series to compute subsequent rolling windows, strictly avoiding future target leakage.
    """
    meter_data = history_df[history_df["meter_id"] == meter_id].sort_values("timestamp").copy()
    if len(meter_data) < 168:
        raise ValueError(f"Insufficient historical data for meter {meter_id} (requires at least 168 hours).")

    last_row = meter_data.iloc[-1]
    last_timestamp = pd.to_datetime(last_row["timestamp"])

    # Extract historical consumption series for the meter
    consumptions = meter_data["consumption_kwh"].tolist()
    temperatures = meter_data["temperature"].tolist()
    
    meter_encoded = int(last_row.get("meter_encoded", 1))

    forecast_records = []

    for h in range(1, hours_ahead + 1):
        target_timestamp = last_timestamp + pd.Timedelta(hours=h)
        hour = int(target_timestamp.hour)
        day_of_week = int(target_timestamp.dayofweek)
        month = int(target_timestamp.month)
        day_of_month = int(target_timestamp.day)
        is_weekend = int(day_of_week >= 5)
        is_holiday = int(last_row.get("is_holiday", 0))

        # Dynamic lag values from recent history + previous recursive predictions
        lag_1 = float(consumptions[-1])
        lag_2 = float(consumptions[-2])
        lag_24 = float(consumptions[-24])
        lag_48 = float(consumptions[-48])
        lag_168 = float(consumptions[-168])

        # Dynamic rolling stats from recent history
        last_24 = consumptions[-24:]
        last_168 = consumptions[-168:]

        rolling_mean_24 = float(np.mean(last_24))
        rolling_mean_168 = float(np.mean(last_168))
        rolling_std_24 = float(np.std(last_24))
        rolling_max_24 = float(np.max(last_24))
        rolling_min_24 = float(np.min(last_24))

        # Estimated temperature for forecast hour (seasonal diurnal continuity)
        # Uses temperature 24 hours prior
        temperature = float(temperatures[-24]) if len(temperatures) >= 24 else float(last_row["temperature"])

        # Construct single feature row matching training feature_cols
        feature_map = {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_24": lag_24,
            "lag_48": lag_48,
            "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24,
            "rolling_mean_168": rolling_mean_168,
            "rolling_std_24": rolling_std_24,
            "rolling_max_24": rolling_max_24,
            "rolling_min_24": rolling_min_24,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_month": day_of_month,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "temperature": temperature,
            "meter_encoded": meter_encoded
        }

        feature_vector = np.array([[feature_map[col] for col in feature_cols]], dtype=np.float32)

        # Predict
        if hasattr(model, "predict"):
            pred_kwh = float(model.predict(feature_vector)[0])
        elif isinstance(model, str) and model == "naive":
            pred_kwh = lag_24
        elif isinstance(model, str) and model == "moving_average":
            pred_kwh = rolling_mean_24
        else:
            pred_kwh = lag_1

        # Clip non-negative
        pred_kwh = max(0.0, pred_kwh)

        # Append predicted consumption to series for subsequent steps
        consumptions.append(pred_kwh)
        temperatures.append(temperature)

        # Estimate confidence interval (approximate 95% interval based on rolling std)
        ci_half_width = 1.96 * max(rolling_std_24, 0.05)
        lower_bound = max(0.0, pred_kwh - ci_half_width)
        upper_bound = pred_kwh + ci_half_width

        forecast_records.append({
            "timestamp": target_timestamp,
            "meter_id": meter_id,
            "forecast_step": h,
            "predicted_kwh": round(pred_kwh, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "temperature_est": round(temperature, 1)
        })

    return pd.DataFrame(forecast_records)
