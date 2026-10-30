"""
Central Machine Learning Training Pipeline for SmartMeter AI.
Orchestrates:
1. Raw dataset loading and validation.
2. Missing value imputation and data quality auditing.
3. Feature engineering (lags, rolling stats, calendar, deviation features).
4. Strictly chronological train/val/test splitting (70% / 15% / 15%).
5. Baseline forecasting models (Naive lag-24 & Moving Average).
6. Random Forest Regressor hyperparameter validation and training.
7. Unsupervised Isolation Forest anomaly detection training (without ground-truth labels).
8. Model evaluation on held-out test data and artifact persistence.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Local application imports
from src.data_loader import load_raw_data, get_data_summary
from src.preprocessing import clean_smart_meter_data
from src.feature_engineering import (
    engineer_meter_features,
    get_forecasting_feature_names,
    get_anomaly_feature_names
)
from src.forecasting import (
    split_time_series_chronological,
    predict_naive_baseline,
    predict_moving_average_baseline
)
from src.anomaly_detection import (
    train_isolation_forest,
    detect_anomalies,
    evaluate_isolation_forest
)
from src.evaluation import (
    calculate_forecasting_metrics,
    format_metrics_table
)
from sklearn.ensemble import RandomForestRegressor


def run_training_pipeline():
    start_total_time = time.time()
    print("=" * 70, flush=True)
    print("SMARTMETER AI: CENTRAL MACHINE LEARNING TRAINING PIPELINE", flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    # Create destination directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: Load Raw Dataset
    # ---------------------------------------------------------
    print("\n[STAGE 1/7] Loading Raw Dataset...", flush=True)
    t0 = time.time()
    raw_df = load_raw_data("data/smart_meter_data.csv")
    raw_summary = get_data_summary(raw_df)
    print(f"Loaded {raw_summary['total_records']:,} records across {raw_summary['unique_meters']} meters in {time.time()-t0:.2f}s", flush=True)
    print(f"Initial missing consumption: {raw_summary['missing_consumption']:,} | missing temperature: {raw_summary['missing_temperature']:,}", flush=True)
    print(f"Ground-truth anomalies present: {raw_summary['anomaly_count']:,} ({raw_summary['anomaly_percentage']}%)", flush=True)

    # ---------------------------------------------------------
    # STAGE 2: Preprocessing and Quality Auditing
    # ---------------------------------------------------------
    print("\n[STAGE 2/7] Preprocessing & Data Quality Auditing...", flush=True)
    t0 = time.time()
    cleaned_df, clean_stats = clean_smart_meter_data(
        raw_df,
        save_path="data/smart_meter_cleaned.csv"
    )
    print(f"Data cleaned and saved to {clean_stats['saved_path']} in {time.time()-t0:.2f}s", flush=True)
    print(f"Post-cleaning missing consumption: {clean_stats['final_null_consumption']} | missing temperature: {clean_stats['final_null_temperature']}", flush=True)
    print(f"Duplicates removed: {clean_stats['duplicates_removed']} | Negative values: {clean_stats['final_negative_consumption']}", flush=True)

    # ---------------------------------------------------------
    # STAGE 3: Feature Engineering
    # ---------------------------------------------------------
    print("\n[STAGE 3/7] Generating Time-Series Features...", flush=True)
    t0 = time.time()
    # engineer_meter_features computes lags (1, 2, 24, 48, 168) and rolling statistics (24, 168)
    # strictly using shift(1) to avoid target leakage
    features_df = engineer_meter_features(cleaned_df, drop_na_lags=True)
    forecasting_features = get_forecasting_feature_names()
    anomaly_features = get_anomaly_feature_names()
    print(f"Features engineered: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns in {time.time()-t0:.2f}s", flush=True)
    print(f"Forecasting feature set ({len(forecasting_features)}): {forecasting_features}", flush=True)
    print(f"Anomaly feature set ({len(anomaly_features)}): {anomaly_features}", flush=True)

    # ---------------------------------------------------------
    # STAGE 4: Chronological Train / Val / Test Split
    # ---------------------------------------------------------
    print("\n[STAGE 4/7] Chronological Train/Val/Test Splitting...", flush=True)
    train_df, val_df, test_df, split_info = split_time_series_chronological(
        features_df,
        train_pct=0.70,
        val_pct=0.15,
        test_pct=0.15
    )
    print(f"Train split: {split_info['train_rows']:,} rows ({split_info['train_start']} -> {split_info['train_end']})", flush=True)
    print(f"Val split:   {split_info['val_rows']:,} rows ({split_info['val_start']} -> {split_info['val_end']})", flush=True)
    print(f"Test split:  {split_info['test_rows']:,} rows ({split_info['test_start']} -> {split_info['test_end']})", flush=True)

    y_test = test_df["consumption_kwh"].values
    y_val = val_df["consumption_kwh"].values

    # ---------------------------------------------------------
    # STAGE 5: Electricity Demand Forecasting Models
    # ---------------------------------------------------------
    print("\n[STAGE 5/7] Evaluating & Training Demand Forecasting Models...", flush=True)
    forecasting_metrics = {}

    # Model 1: Naive Baseline (lag_24)
    t0 = time.time()
    naive_test_preds = predict_naive_baseline(test_df)
    naive_metrics = calculate_forecasting_metrics(y_test, naive_test_preds)
    forecasting_metrics["Naive Baseline (lag_24)"] = naive_metrics
    print(f"Model 1 (Naive Baseline) -> Test MAE: {naive_metrics['MAE']:.4f}, RMSE: {naive_metrics['RMSE']:.4f}, R2: {naive_metrics['R2']:.4f} ({time.time()-t0:.2f}s)", flush=True)

    # Model 2: Moving Average Baseline (rolling_mean_24)
    t0 = time.time()
    ma_test_preds = predict_moving_average_baseline(test_df)
    ma_metrics = calculate_forecasting_metrics(y_test, ma_test_preds)
    forecasting_metrics["Moving Average Baseline (24h)"] = ma_metrics
    print(f"Model 2 (Moving Average) -> Test MAE: {ma_metrics['MAE']:.4f}, RMSE: {ma_metrics['RMSE']:.4f}, R2: {ma_metrics['R2']:.4f} ({time.time()-t0:.2f}s)", flush=True)

    # Model 3: Random Forest Regressor
    print("Model 3: Tuning & Training Random Forest Regressor...", flush=True)
    t0 = time.time()
    
    # Validate candidate max_depth on validation set
    candidate_depths = [8, 10, 12]
    best_depth = 12
    best_val_rmse = float("inf")
    depth_tuning_scores = {}

    X_train_tune = train_df[forecasting_features].values[::2]  # fast validation subset
    y_train_tune = train_df["consumption_kwh"].values[::2]
    X_val = val_df[forecasting_features].values

    for depth in candidate_depths:
        rf_tune = RandomForestRegressor(
            n_estimators=40,
            max_depth=depth,
            max_samples=0.5,
            random_state=42,
            n_jobs=-1
        )
        rf_tune.fit(X_train_tune, y_train_tune)
        val_pred = rf_tune.predict(X_val)
        val_m = calculate_forecasting_metrics(y_val, val_pred)
        depth_tuning_scores[depth] = val_m
        print(f"  - Validation with max_depth={depth}: MAE={val_m['MAE']:.4f}, RMSE={val_m['RMSE']:.4f}, R2={val_m['R2']:.4f}", flush=True)
        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_depth = depth

    print(f"Selected best max_depth = {best_depth} (Validation RMSE: {best_val_rmse:.4f})", flush=True)

    # Train final Random Forest on full training data
    print(f"Training final Random Forest (n_estimators=100, max_depth={best_depth}, random_state=42)...", flush=True)
    t_train = time.time()
    rf_final = RandomForestRegressor(
        n_estimators=100,
        max_depth=best_depth,
        max_samples=0.5,
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(train_df[forecasting_features].values, train_df["consumption_kwh"].values)
    print(f"Final Random Forest trained in {time.time()-t_train:.2f}s", flush=True)

    # Evaluate on held-out test set
    rf_test_preds = rf_final.predict(test_df[forecasting_features].values)
    rf_metrics = calculate_forecasting_metrics(y_test, rf_test_preds)
    forecasting_metrics["Random Forest Regressor"] = rf_metrics
    print(f"Model 3 (Random Forest)  -> Test MAE: {rf_metrics['MAE']:.4f}, RMSE: {rf_metrics['RMSE']:.4f}, R2: {rf_metrics['R2']:.4f}", flush=True)

    # Save RF Model
    rf_model_path = "models/random_forest_model.pkl"
    joblib.dump(rf_final, rf_model_path, compress=3)
    print(f"Saved Random Forest model to {rf_model_path}", flush=True)

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": forecasting_features,
        "importance": rf_final.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv("outputs/feature_importance.csv", index=False)
    print("Saved feature importance to outputs/feature_importance.csv", flush=True)

    # Save Forecasting Comparison Table
    metrics_df = format_metrics_table(forecasting_metrics)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)
    print("Saved forecasting metrics comparison to outputs/model_metrics.csv", flush=True)

    # ---------------------------------------------------------
    # STAGE 6: Anomaly Detection (Isolation Forest)
    # ---------------------------------------------------------
    print("\n[STAGE 6/7] Training & Evaluating Isolation Forest...", flush=True)
    t0 = time.time()
    # Isolation Forest trained WITHOUT is_anomaly
    iso_model = train_isolation_forest(
        train_df=train_df,
        feature_cols=anomaly_features,
        contamination=0.025,
        random_state=42
    )
    print(f"Isolation Forest trained in {time.time()-t0:.2f}s", flush=True)

    # Save Isolation Forest Model
    iso_model_path = "models/isolation_forest_model.pkl"
    joblib.dump(iso_model, iso_model_path, compress=3)
    print(f"Saved Isolation Forest model to {iso_model_path}", flush=True)

    # Run inference across dataset
    print("Detecting anomalies across the dataset...", flush=True)
    t0 = time.time()
    scored_df = detect_anomalies(iso_model, features_df, anomaly_features)
    print(f"Anomaly scoring completed in {time.time()-t0:.2f}s", flush=True)

    # Evaluate on held-out test set
    test_mask = scored_df["timestamp"] > split_info["val_end"]
    test_scored = scored_df[test_mask]
    iso_eval_test = evaluate_isolation_forest(test_scored)
    
    # Evaluate across full dataset
    iso_eval_full = evaluate_isolation_forest(scored_df)

    print("\n--- Isolation Forest Evaluation vs Ground Truth (Test Set) ---", flush=True)
    print(f"Precision:         {iso_eval_test['precision']:.4f}", flush=True)
    print(f"Recall:            {iso_eval_test['recall']:.4f}", flush=True)
    print(f"F1 Score:          {iso_eval_test['f1_score']:.4f}", flush=True)
    print(f"Confusion Matrix:  {iso_eval_test['confusion_matrix']}", flush=True)
    print(f"True Positives:    {iso_eval_test['true_positives']:,} | False Positives: {iso_eval_test['false_positives']:,}", flush=True)
    print(f"True Negatives:    {iso_eval_test['true_negatives']:,} | False Negatives: {iso_eval_test['false_negatives']:,}", flush=True)

    # Save detected anomalies to outputs/anomaly_results.csv
    anomalies_only = scored_df[scored_df["anomaly_flag"] == 1][[
        "meter_id",
        "timestamp",
        "consumption_kwh",
        "rolling_mean_24",
        "anomaly_score",
        "anomaly_type",
        "severity",
        "is_anomaly"
    ]].copy()
    anomalies_only.to_csv("outputs/anomaly_results.csv", index=False)
    print(f"Saved {len(anomalies_only):,} detected anomalies to outputs/anomaly_results.csv", flush=True)

    # ---------------------------------------------------------
    # STAGE 7: Save Comprehensive Metadata
    # ---------------------------------------------------------
    print("\n[STAGE 7/7] Persisting Metadata & Summaries...", flush=True)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(cleaned_df),
        "unique_meters": int(cleaned_df["meter_id"].nunique()),
        "time_range": {
            "start": str(cleaned_df["timestamp"].min()),
            "end": str(cleaned_df["timestamp"].max())
        },
        "chronological_split": split_info,
        "forecasting": {
            "target": "consumption_kwh",
            "features": forecasting_features,
            "best_max_depth": best_depth,
            "depth_tuning_scores": {str(k): v for k, v in depth_tuning_scores.items()},
            "test_metrics": forecasting_metrics
        },
        "anomaly_detection": {
            "features": anomaly_features,
            "contamination": 0.025,
            "test_evaluation": iso_eval_test,
            "full_evaluation": iso_eval_full,
            "total_anomalies_detected": int(scored_df["anomaly_flag"].sum()),
            "anomaly_rate_pct": round(float(scored_df["anomaly_flag"].mean() * 100), 3)
        },
        "execution_time_seconds": round(time.time() - start_total_time, 2)
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Saved pipeline metadata to models/model_metadata.json", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("TRAINING PIPELINE COMPLETE!", flush=True)
    print(f"Total pipeline execution time: {time.time()-start_total_time:.2f} seconds", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    run_training_pipeline()
