"""
Feature Engineering Module for SmartMeter AI.
Generates lag features, rolling statistics, calendar indicators, and deviation metrics
while strictly avoiding future target leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def extract_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date and time features from the timestamp column.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["month"] = df["timestamp"].dt.month.astype("int8")
    df["day_of_month"] = df["timestamp"].dt.day.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # If is_holiday is not present, add 0
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    else:
        df["is_holiday"] = df["is_holiday"].astype("int8")

    return df


def engineer_meter_features(
    df: pd.DataFrame,
    drop_na_lags: bool = True
) -> pd.DataFrame:
    """
    Generates time-series lag and rolling statistics separately per smart meter.
    
    IMPORTANT:
    - shift(1) is applied prior to rolling window calculations to ensure NO CURRENT-TARGET LEAKAGE.
    - Chronological sorting per meter is guaranteed.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure chronological sort per meter
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Calendar features
    df = extract_calendar_features(df)

    # Meter ID integer encoding (e.g., METER_001 -> 1)
    if "meter_encoded" not in df.columns:
        df["meter_encoded"] = df["meter_id"].str.extract(r"(\d+)").astype("int16")

    # Groupby object for meter consumption
    grouped = df.groupby("meter_id", observed=False)["consumption_kwh"]

    # 1. Lag Features (values at t-1, t-2, t-24, t-48, t-168)
    df["lag_1"] = grouped.shift(1).astype("float32")
    df["lag_2"] = grouped.shift(2).astype("float32")
    df["lag_24"] = grouped.shift(24).astype("float32")
    df["lag_48"] = grouped.shift(48).astype("float32")
    df["lag_168"] = grouped.shift(168).astype("float32")

    # 2. Rolling Features computed strictly on shift(1) (values strictly before t)
    # Using transform with shift(1) guarantees zero leakage of consumption at time t.
    df["rolling_mean_24"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
        .astype("float32")
    )
    df["rolling_mean_168"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(168, min_periods=1).mean())
        .astype("float32")
    )
    df["rolling_std_24"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=1).std())
        .fillna(0.0)
        .astype("float32")
    )
    df["rolling_max_24"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=1).max())
        .astype("float32")
    )
    df["rolling_min_24"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=1).min())
        .astype("float32")
    )

    # 3. Anomaly-related features
    # Household deviation from historical rolling mean
    df["consumption_deviation"] = (df["consumption_kwh"] - df["rolling_mean_24"]).fillna(0.0).astype("float32")

    # Historical household average
    meter_hist_mean = df.groupby("meter_id", observed=False)["consumption_kwh"].transform("mean")
    df["household_hist_mean"] = meter_hist_mean.astype("float32")
    df["household_deviation_hist"] = (df["consumption_kwh"] - meter_hist_mean).astype("float32")

    if drop_na_lags:
        # Drop rows where lag_168 is null (first week of the year)
        df = df.dropna(subset=["lag_168"]).reset_index(drop=True)

    return df


def get_forecasting_feature_names() -> List[str]:
    """
    Returns list of feature column names used for training forecasting models.
    """
    return [
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_48",
        "lag_168",
        "rolling_mean_24",
        "rolling_mean_168",
        "rolling_std_24",
        "rolling_max_24",
        "rolling_min_24",
        "hour",
        "day_of_week",
        "month",
        "day_of_month",
        "is_weekend",
        "is_holiday",
        "temperature",
        "meter_encoded"
    ]


def get_anomaly_feature_names() -> List[str]:
    """
    Returns list of feature column names used for Isolation Forest anomaly detection.
    IMPORTANT: is_anomaly is strictly excluded!
    """
    return [
        "consumption_kwh",
        "rolling_mean_24",
        "rolling_std_24",
        "hour",
        "temperature",
        "consumption_deviation",
        "household_deviation_hist"
    ]
