"""
Data Loader Module for SmartMeter AI.
Handles loading raw and processed smart meter time-series datasets.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional


def load_raw_data(data_path: str = "data/smart_meter_data.csv") -> pd.DataFrame:
    """
    Loads raw electricity consumption dataset.
    Never modifies or overwrites the original raw file.
    """
    if not os.path.exists(data_path):
        # Fallback to current directory if not in data/
        alt_path = "smart_meter_data.csv"
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Raw data file not found at {data_path} or {alt_path}")

    # Optimize column dtypes on read
    dtypes = {
        'meter_id': 'str',
        'consumption_kwh': 'float32',
        'temperature': 'float32',
        'hour': 'int8',
        'day_of_week': 'int8',
        'month': 'int8',
        'is_weekend': 'int8',
        'is_holiday': 'int8',
        'is_anomaly': 'int8'
    }

    df = pd.read_csv(data_path, dtype=dtypes, parse_dates=['timestamp'])
    return df


def load_cleaned_data(data_path: str = "data/smart_meter_cleaned.csv") -> pd.DataFrame:
    """
    Loads preprocessed and cleaned smart meter dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run preprocessing first.")

    dtypes = {
        'meter_id': 'category',
        'consumption_kwh': 'float32',
        'temperature': 'float32',
        'hour': 'int8',
        'day_of_week': 'int8',
        'month': 'int8',
        'is_weekend': 'int8',
        'is_holiday': 'int8',
        'is_anomaly': 'int8'
    }
    df = pd.read_csv(data_path, dtype=dtypes, parse_dates=['timestamp'])
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates high-level metadata and data quality metrics for the dataset.
    """
    return {
        "total_records": len(df),
        "unique_meters": df['meter_id'].nunique(),
        "min_timestamp": str(df['timestamp'].min()),
        "max_timestamp": str(df['timestamp'].max()),
        "columns": list(df.columns),
        "missing_consumption": int(df['consumption_kwh'].isnull().sum()),
        "missing_temperature": int(df['temperature'].isnull().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "anomaly_count": int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
        "anomaly_percentage": round(float(df['is_anomaly'].mean() * 100), 2) if 'is_anomaly' in df.columns else 0.0
    }
