"""
Anomaly Detection Module for SmartMeter AI.
Trains an Isolation Forest on multidimensional consumption features (strictly without ground-truth labels),
detects anomalous readings, computes anomaly scores, and categorizes anomaly behavioral types.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from src.evaluation import calculate_anomaly_metrics


def train_isolation_forest(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.025,
    random_state: int = 42
) -> IsolationForest:
    """
    Trains an Isolation Forest model for unsupervised anomaly detection.
    
    IMPORTANT:
    is_anomaly is strictly excluded from feature_cols.
    Only consumption, rolling stats, calendar, and deviation features are used.
    """
    X_train = train_df[feature_cols].values
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    return iso_forest


def classify_anomaly_type(row: pd.Series) -> str:
    """
    Assigns an operational category to detected anomalies based on consumption patterns.
    Note: Statistical anomalies are not automatically electricity theft; they represent
    equipment behavior, outages, surges, or atypical household activity.
    """
    if row.get("anomaly_flag", 0) == 0:
        return "Normal"

    kwh = row["consumption_kwh"]
    rolling_mean = row.get("rolling_mean_24", kwh)
    rolling_std = row.get("rolling_std_24", 0.1)
    hour = row.get("hour", 12)

    # Nighttime surge: Hours 0-5 with consumption substantially higher than average
    if hour in [0, 1, 2, 3, 4, 5] and kwh > (rolling_mean + 1.5 * max(rolling_std, 0.1)):
        return "Nighttime Surge"

    # Severe Spike
    if kwh > (rolling_mean + 3.0 * max(rolling_std, 0.1)):
        return "Consumption Spike"

    # Unexpected Drop or Near-Zero Outage
    if kwh < 0.05 and rolling_mean > 0.3:
        return "Unexpected Drop / Outage"
    if kwh < (rolling_mean - 2.0 * max(rolling_std, 0.1)):
        return "Consumption Drop"

    # Sustained abnormal high usage
    hist_mean = row.get("household_hist_mean", rolling_mean)
    if kwh > 2.5 * hist_mean:
        return "Abnormal High Usage"

    return "Pattern Deviation"


def detect_anomalies(
    model: IsolationForest,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Applies the trained Isolation Forest to compute anomaly flags, continuous anomaly scores,
    and classifies anomaly behavioral signatures.
    """
    df = df.copy()
    X = df[feature_cols].values

    # Isolation Forest: raw decision_function gives negative score for anomalies
    raw_scores = model.decision_function(X)
    raw_preds = model.predict(X)

    # Convert preds: -1 -> 1 (anomaly), 1 -> 0 (normal)
    df["anomaly_flag"] = (raw_preds == -1).astype("int8")

    # Invert and normalize score so that 0 = perfectly normal, 1 = maximum anomaly
    # decision_function typically ranges from -0.3 to +0.2
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        normalized_score = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
    else:
        normalized_score = np.zeros_like(raw_scores)
    df["anomaly_score"] = np.round(normalized_score, 4).astype("float32")

    # Categorize anomaly severity
    conditions = [
        (df["anomaly_flag"] == 1) & (df["anomaly_score"] >= 0.85),
        (df["anomaly_flag"] == 1) & (df["anomaly_score"] >= 0.70),
        (df["anomaly_flag"] == 1)
    ]
    severities = ["High", "Medium", "Low"]
    df["severity"] = np.select(conditions, severities, default="Normal")

    # Assign operational classification
    # Vectorized classification check
    kwh = df["consumption_kwh"]
    r_mean = df["rolling_mean_24"] if "rolling_mean_24" in df.columns else kwh
    r_std = df["rolling_std_24"] if "rolling_std_24" in df.columns else 0.1
    hour = df["hour"] if "hour" in df.columns else 12

    types = np.full(len(df), "Normal", dtype=object)
    anom_mask = df["anomaly_flag"] == 1

    night_mask = anom_mask & (hour.isin([0, 1, 2, 3, 4, 5])) & (kwh > (r_mean + 1.5 * np.maximum(r_std, 0.1)))
    spike_mask = anom_mask & (~night_mask) & (kwh > (r_mean + 2.5 * np.maximum(r_std, 0.1)))
    drop_mask = anom_mask & (~night_mask) & (~spike_mask) & ((kwh < 0.05) | (kwh < (r_mean - 1.8 * np.maximum(r_std, 0.1))))
    pattern_mask = anom_mask & (~night_mask) & (~spike_mask) & (~drop_mask)

    types[night_mask] = "Nighttime Surge"
    types[spike_mask] = "Consumption Spike"
    types[drop_mask] = "Unexpected Drop"
    types[pattern_mask] = "Pattern Deviation"

    df["anomaly_type"] = types

    return df


def evaluate_isolation_forest(
    detected_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates detected anomalies against the synthetic ground-truth 'is_anomaly' column.
    """
    if "is_anomaly" not in detected_df.columns:
        raise ValueError("Ground-truth column 'is_anomaly' missing from dataframe.")

    y_true = detected_df["is_anomaly"].values
    y_pred = detected_df["anomaly_flag"].values

    return calculate_anomaly_metrics(y_true, y_pred)
