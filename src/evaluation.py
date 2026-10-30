"""
Evaluation Module for SmartMeter AI.
Computes forecasting performance metrics (MAE, RMSE, R2) and
anomaly detection metrics (Precision, Recall, F1, Confusion Matrix).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_forecasting_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, R2.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Handle any NaN if present
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t = y_t[mask]
    y_p = y_p[mask]

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def calculate_anomaly_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, Any]:
    """
    Computes classification evaluation metrics for anomaly detection:
    Precision, Recall, F1 Score, and Confusion Matrix.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    precision = float(precision_score(y_t, y_p, zero_division=0))
    recall = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    cm = confusion_matrix(y_t, y_p)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def format_metrics_table(models_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Converts a nested dictionary of model metrics into a polished comparison DataFrame.
    """
    df = pd.DataFrame.from_dict(models_metrics, orient="index")
    df.index.name = "Model"
    return df.reset_index()
