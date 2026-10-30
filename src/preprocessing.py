"""
Preprocessing Module for SmartMeter AI.
Handles data cleaning, missing value imputation, anomaly label preservation,
and data quality auditing.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def clean_smart_meter_data(
    df: pd.DataFrame,
    save_path: Optional[str] = "data/smart_meter_cleaned.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw smart meter data:
    1. Validates and enforces schema.
    2. Sorts chronologically per meter.
    3. Handles missing consumption and temperature values intelligently.
    4. Eliminates duplicate records.
    5. Validates non-negative consumption.
    6. Preserves is_anomaly ground-truth labels.
    7. Calculates audit data quality statistics.
    8. Persists cleaned dataset separately.
    """
    stats: Dict[str, Any] = {}

    # Initial stats
    stats["initial_rows"] = len(df)
    stats["initial_null_consumption"] = int(df["consumption_kwh"].isnull().sum())
    stats["initial_null_temperature"] = int(df["temperature"].isnull().sum())
    stats["initial_negative_consumption"] = int((df["consumption_kwh"] < 0).sum())

    # Ensure timestamp is datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicates if any
    duplicates_count = int(df.duplicated(subset=["meter_id", "timestamp"]).sum())
    stats["duplicates_removed"] = duplicates_count
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=["meter_id", "timestamp"]).copy()
    else:
        df = df.copy()

    # Sort strictly chronologically per meter
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Intelligent Imputation for consumption_kwh:
    # 1. Time-series linear interpolation within each meter
    # 2. bfill and ffill for any remaining edge boundaries
    df["consumption_kwh"] = (
        df.groupby("meter_id", observed=False)["consumption_kwh"]
        .transform(lambda s: s.interpolate(method="linear").bfill().ffill())
    )

    # Intelligent Imputation for temperature:
    # Temperature varies regionally and temporally; interpolate by time
    df["temperature"] = df["temperature"].interpolate(method="linear").bfill().ffill()

    # Check and rectify any negative consumption values
    neg_mask = df["consumption_kwh"] < 0
    if neg_mask.sum() > 0:
        df.loc[neg_mask, "consumption_kwh"] = 0.0

    # Ensure proper data types
    df["consumption_kwh"] = df["consumption_kwh"].astype("float32")
    df["temperature"] = df["temperature"].astype("float32")
    for col in ["hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_anomaly"]:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    # Post-cleaning stats
    stats["final_rows"] = len(df)
    stats["final_null_consumption"] = int(df["consumption_kwh"].isnull().sum())
    stats["final_null_temperature"] = int(df["temperature"].isnull().sum())
    stats["final_negative_consumption"] = int((df["consumption_kwh"] < 0).sum())
    stats["unique_meters"] = int(df["meter_id"].nunique())
    stats["ground_truth_anomalies"] = int(df["is_anomaly"].sum())
    stats["ground_truth_anomaly_pct"] = round(float(df["is_anomaly"].mean() * 100), 3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Avoid overwriting original dataset
        if os.path.abspath(save_path) == os.path.abspath("data/smart_meter_data.csv"):
            raise ValueError("Safety error: save_path cannot be the raw data file!")
        df.to_csv(save_path, index=False)
        stats["saved_path"] = save_path

    return df, stats
