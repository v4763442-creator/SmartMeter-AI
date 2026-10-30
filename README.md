# SmartMeter AI – Electricity Demand Forecasting & Intelligent Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**SmartMeter AI** is a production-grade enterprise energy analytics platform built to monitor, forecast, and audit smart electricity meters at scale. Combining supervised machine learning for time-series demand forecasting with unsupervised isolation trees for anomaly detection, the platform enables grid operators, energy retailers, and utility distributors to forecast consumption and identify abnormal behavioral patterns.

---

## 1. Project Overview

Modern electrical grids face unprecedented volatility driven by electrification (EV charging, heat pumps), renewable intermittency, and demand-side surges. SmartMeter AI processes high-frequency smart meter telemetry to deliver:
- **Macro Grid Telemetry Auditing:** Real-time visibility into system-wide load curves, diurnal cycles, and consumer distributions across 100 smart meters (876,000 readings).
- **Time-Series Demand Forecasting:** Multi-model comparative forecasting (Naive Baseline, Moving Average Baseline, and Random Forest Regressor) with true recursive 24-hour ahead multi-step predictions.
- **Intelligent Anomaly Detection:** Completely unsupervised multidimensional Isolation Forest for identifying consumption spikes, daylight outages, nighttime surges, and household signature drifts.
- **Enterprise SaaS Dashboard:** Responsive dark-theme interactive UI built with Streamlit and Plotly, featuring metric cards, drill-down filters, and audit report downloads.

---

## 2. Business Use Cases

1. **Short-Term Load Forecasting & Grid Dispatch:** Utilities can forecast day-ahead load per feeder and individual household, reducing costly peak peaker-plant dispatches.
2. **Infrastructure Protection & Transformer Sizing:** Detect sustained overloading or high load factors before local distribution transformers overheat.
3. **Revenue Protection & Tampering Triage:** Isolate sudden drops or abnormal usage patterns for targeted field inspections while distinguishing operational spikes from genuine illicit activity.
4. **Demand-Side Management (DSM):** Provide automated insights into household routines (e.g., peak vs off-peak consumption) for dynamic time-of-use (TOU) tariff optimization.

---

## 3. Dataset Specifications

The platform utilizes a comprehensive smart meter dataset spanning the entire year of 2025:
- **Total Records:** 876,000 hourly readings
- **Coverage:** 100 Unique Smart Meters (`METER_001` through `METER_100`)
- **Temporal Span:** January 1, 2025 00:00:00 to December 31, 2025 23:00:00 (8,760 consecutive hours per meter)
- **Telemetry Attributes:**
  - `timestamp`: Chronological observation time (hourly resolution)
  - `meter_id`: Unique identifier for each household meter
  - `consumption_kwh`: Active electricity consumption (kWh)
  - `temperature`: Regional ambient temperature (°C)
  - `hour`, `day_of_week`, `month`: Temporal components
  - `is_weekend`, `is_holiday`: Calendar indicators
  - `is_anomaly`: Ground-truth benchmark label (strictly preserved for held-out evaluation only)

---

## 4. Data Preprocessing & Quality Auditing

The raw telemetry data undergoes strict automated validation and preprocessing via `src/preprocessing.py`:
1. **Schema Validation & Typing:** Enforces float32 numerical precision and int8 calendar encodings to optimize memory efficiency.
2. **Missing Value Imputation:**
   - **Consumption Nulls (12,756 records / 1.46%):** Handled via intra-meter time-series linear interpolation, capturing continuous diurnal continuity, with backward/forward fills for boundary edges.
   - **Temperature Nulls (3,581 records / 0.41%):** Handled via chronological linear interpolation.
3. **Duplicate & Invariant Verification:** Verified zero duplicate `(meter_id, timestamp)` tuples.
4. **Range Enforcement:** Verified non-negativity across all readings (`consumption_kwh >= 0.0`).
5. **Separation of Raw & Processed Data:** Raw data (`data/smart_meter_data.csv`) is treated as read-only; preprocessed data is persisted separately to `data/smart_meter_cleaned.csv`.

---

## 5. Feature Engineering

Feature engineering (`src/feature_engineering.py`) generates a comprehensive 18-feature space for forecasting and a 7-feature space for anomaly detection while **strictly avoiding target leakage**:

### Lag Features (Per Meter)
- `lag_1`: Consumption at $t-1$ (instantaneous autoregressive inertia)
- `lag_2`: Consumption at $t-2$
- `lag_24`: Consumption at $t-24$ (previous day's identical hour)
- `lag_48`: Consumption at $t-48$ (two days prior)
- `lag_168`: Consumption at $t-168$ (same hour on the preceding week)

### Rolling Statistics (Strictly Computed on `shift(1)`)
To ensure zero current-target leakage, all rolling statistics are computed on the `shift(1)` series:
- `rolling_mean_24`: 24-hour historical rolling mean
- `rolling_mean_168`: 7-day (168-hour) historical rolling mean
- `rolling_std_24`: 24-hour historical rolling standard deviation (volatility indicator)
- `rolling_max_24`: 24-hour historical peak demand
- `rolling_min_24`: 24-hour historical minimum baseload

### Calendar & Exogenous Features
- `hour` (0–23), `day_of_week` (0–6), `month` (1–12), `day_of_month` (1–31)
- `is_weekend`, `is_holiday`
- `temperature`: Regional weather impact on heating/cooling demand
- `meter_encoded`: Integer household identifier

### Deviation Features for Anomaly Detection
- `consumption_deviation`: Difference between current consumption and rolling 24h mean ($y_t - \mu_{24}$)
- `household_deviation_hist`: Difference between current consumption and household historical baseline

---

## 6. Electricity Demand Forecasting Methodology

### Chronological Train / Validation / Test Splitting
Time-series data is split strictly chronologically across all 100 meters without shuffling:
- **Training Set (70%):** Jan 8, 2025 to Sep 15, 2025 (601,500 observations)
- **Validation Set (15%):** Sep 15, 2025 to Nov 8, 2025 (128,900 observations)
- **Testing Set (15%):** Nov 8, 2025 to Dec 31, 2025 (128,800 out-of-sample observations)

### Models Compared
1. **Model 1: Naive Baseline (`lag_24`)**
   - Assumes electricity demand at hour $h$ today equals consumption at hour $h$ yesterday.
2. **Model 2: Moving Average Baseline (`rolling_mean_24`)**
   - Predicts consumption using the recent 24-hour rolling average.
3. **Model 3: Random Forest Regressor**
   - Ensemble of 100 decision trees trained with sub-sampling (`max_samples=0.5`).
   - Hyperparameter validation selected `max_depth = 12` based on lowest validation RMSE.

### Recursive Multi-Step 24-Hour Forecasting
For forward-looking projections:
- Rather than reusing actual future lags (which would be data leakage), the system implements **strict recursive forecasting**:
  $$\hat{y}_{t+1} = f(\mathbf{x}_t), \quad \hat{y}_{t+2} = f(\hat{\mathbf{x}}_{t+1}), \quad \dots \quad \hat{y}_{t+24} = f(\hat{\mathbf{x}}_{t+23})$$
- Each predicted consumption step dynamically updates the lag array and rolling windows for subsequent steps.
- Projects 95% confidence intervals derived from recent local residual variance.

---

## 7. Anomaly Detection Methodology

### Isolation Forest Configuration
- **Algorithm:** Scikit-Learn `IsolationForest`
- **Contamination:** `contamination = 0.025` (~2.5% expected outlier fraction)
- **Random State:** 42
- **Features Used:** `consumption_kwh`, `rolling_mean_24`, `rolling_std_24`, `hour`, `temperature`, `consumption_deviation`, `household_deviation_hist`
- **CRITICAL INTEGRITY RULE:** Ground-truth `is_anomaly` is **strictly excluded** from model fitting. The model operates 100% unsupervised.

### Operational Anomaly Categorization
Detected anomalies are mathematically categorized into four operational signatures:
1. **Consumption Spike:** $y_t > \mu_{24} + 2.5 \times \sigma_{24}$ (severe demand spikes)
2. **Unexpected Drop / Outage:** $y_t < 0.05 \text{ kWh}$ while baseline is active, or $y_t < \mu_{24} - 1.8 \times \sigma_{24}$
3. **Nighttime Surge:** High consumption ($> \mu_{24} + 1.5 \times \sigma_{24}$) occurring between 00:00 and 05:00
4. **Pattern Deviation:** Multidimensional tree isolation not captured by single-axis thresholding

> **Operational Distinction:** Statistical anomalies indicate significant behavioral deviations from normal routine. They should not be conflated automatically with meter theft; common legitimate causes include EV fast charging, space heaters, heat pump defrost cycles, temporary vacancies, and grid voltage events.

---

## 8. Model Evaluation & Benchmark Results

### Demand Forecasting Benchmark (128,800 Held-Out Test Records)

| Model | MAE (kWh) | RMSE (kWh) | $R^2$ Score | Performance Summary |
|---|:---:|:---:|:---:|---|
| **Naive Baseline (`lag_24`)** | 0.1362 | 0.2941 | 0.2086 | Captures 24h diurnal rhythm but susceptible to day-to-day shifts |
| **Moving Average Baseline (24h)** | 0.1851 | 0.2884 | 0.2388 | Smooths baseline demand; lags significantly during rapid peak spikes |
| **Random Forest Regressor (Champion)** | **0.0892** | **0.1929** | **0.6595** | **68% error reduction vs MA; 3.2x higher explained variance ($R^2 = 0.66$)** |

### Top Predictive Features (Random Forest MDI Importance)
1. `lag_1` (78.56%): Autoregressive immediate inertia
2. `lag_24` (8.60%): 24-hour diurnal recurrence
3. `lag_168` (2.64%): Weekly routine recurrence
4. `temperature` (2.15%): Weather-driven heating/cooling response
5. `lag_2` (1.10%) & `lag_48` (1.09%): Short-term smoothing
6. `hour` (0.99%): Diurnal schedule alignment

### Isolation Forest Evaluation vs Ground Truth (Test Period)
- **Precision:** 0.5646
- **Recall (Sensitivity):** 0.2730
- **F1 Score:** 0.3681
- **Confusion Matrix:**
  - True Negatives (TN): 124,597
  - False Positives (FP): 731
  - False Negatives (FN): 2,524
  - True Positives (TP): 948
- **Total Flagged Anomalies Across All Data:** 17,501 incidents

---

## 9. Modular Architecture

```
POWER_ML/
│
├── app.py                            # Streamlit main entrypoint & executive landing
│
├── pages/
│   ├── 1_Overview.py                 # Executive Overview: macro load, diurnal cycles, top meters
│   ├── 2_Forecasting.py              # Demand Forecasting: 24h recursive forecast & model comparisons
│   ├── 3_Anomaly_Detection.py        # Anomaly Detection: timeline, severity distributions, audit table
│   ├── 4_Meter_Analytics.py          # Meter Analytics: weekly heatmaps, load factor, peer benchmarking
│   ├── 5_Model_Performance.py        # Model Auditing: MAE/RMSE/R2, residuals, feature importance, CM
│   └── 6_Data_Explorer.py            # Data Explorer: raw vs clean audit, interactive filters, CSV export
│
├── src/
│   ├── __init__.py                   # Package initializers
│   ├── data_loader.py                # Raw & cleaned data ingestion, type optimization, metadata summary
│   ├── preprocessing.py              # Imputation, range checks, deduplication, audit logging
│   ├── feature_engineering.py        # Shift(1) rolling windows, lag generation, calendar extraction
│   ├── forecasting.py                # Chronological split, Naive, Moving Avg, RF & Recursive Forecasting
│   ├── anomaly_detection.py          # Unsupervised Isolation Forest, scoring & operational classification
│   ├── evaluation.py                 # MAE, RMSE, R2, Precision, Recall, F1, Confusion Matrix
│   ├── visualization.py              # Consistent dark SaaS Plotly styling, heatmaps, time series
│   └── dashboard_utils.py            # Streamlit caching (@st.cache_data), custom CSS injection, KPI cards
│
├── data/
│   ├── smart_meter_data.csv          # Raw benchmark dataset (876,000 rows, 100 meters)
│   └── smart_meter_cleaned.csv       # Preprocessed and imputed dataset
│
├── models/
│   ├── random_forest_model.pkl       # Serialized Random Forest model (joblib compressed, ~14MB)
│   ├── isolation_forest_model.pkl    # Serialized Isolation Forest model (joblib compressed, ~300KB)
│   └── model_metadata.json           # Complete pipeline run metadata, hyperparameters & metrics
│
├── outputs/
│   ├── model_metrics.csv             # Forecasting benchmark comparison table
│   ├── anomaly_results.csv           # Detected anomalous incidents with scores & classifications
│   └── feature_importance.csv        # Random Forest MDI importance ranking
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # Comprehensive EDA, feature engineering, and modeling notebook
│
├── train.py                          # Central training & pipeline orchestration script
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Standard exclusions
```

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- 2 GB available RAM

### Step 1: Clone or Open Workspace
```bash
cd POWER_ML
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. Execution Instructions

### Step 1: Run the Central Training Pipeline
Executes data validation, feature generation, model training, evaluation, and persists all artifacts:
```bash
python train.py
```
*Expected execution time: ~2 minutes.*

### Step 2: Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser to:
`http://localhost:8501`

---

## 12. Dashboard Navigation & Feature Walkthrough

### Page 1: Executive Overview (`1_Overview.py`)
- Real-time KPI metric cards (Total MWh, Smart Meters, Avg Demand, Telemetry count, Anomaly rate).
- System-wide electricity consumption trend chart.
- Diurnal load profile (24-hour cycle mean vs median).
- Monthly consumption volumes.
- Top 10 smart meters by annual energy volume.
- Interactive filters for date range, specific meter, and consumption period (Peak, Business, Off-Peak).

### Page 2: Demand Forecasting (`2_Forecasting.py`)
- Meter selector dropdown (`METER_001` – `METER_100`).
- Model selector: Random Forest Regressor, Naive Baseline, Moving Average.
- **24-Hour Recursive Ahead Forecast:** Forecast plot with 95% confidence intervals and preceding 48 hours of context.
- Actual vs Predicted overlay on the test period with zoom slider.
- Prediction error (residual) distribution chart with zero-error reference line.

### Page 3: Anomaly Detection (`3_Anomaly_Detection.py`)
- Total anomalies detected, anomaly percentage, high-severity event counts.
- Timeline scatter chart mapping normal readings vs detected anomalies.
- Anomaly categorization breakdown (Spikes, Drops, Nighttime Surges, Pattern Deviations).
- Top 10 anomalous meters ranking.
- Incident audit table with search and single-click CSV export.

### Page 4: Meter Analytics (`4_Meter_Analytics.py`)
- Household consumption history with 24h rolling trend.
- Day of Week vs Hour of Day **weekly heatmap** illustrating household routines.
- Household diurnal profile benchmarked against overall cohort average.
- Peak demand, baseload, and load factor calculation.

### Page 5: Model Performance (`5_Model_Performance.py`)
- Comparative MAE, RMSE, and $R^2$ bar charts across all models.
- Random Forest feature importance chart.
- Isolation Forest Confusion Matrix heatmap against ground truth.
- Precision, Recall, and F1 Score summary cards.

### Page 6: Data Explorer (`6_Data_Explorer.py`)
- Telemetry table with meter and date range filters.
- Data hygiene audit summary (missing values before vs after, duplicates, negative values).
- Side-by-side raw vs cleaned dataset preview.
- CSV export for filtered datasets and anomaly reports.

---

## 13. Limitations & Future Roadmap

1. **Weather Exogenous Depth:** Ambient temperature is currently included; incorporating humidity, solar irradiance, and wind chill will further improve seasonal transition forecasting.
2. **Deep Learning Sequence Models:** Expanding the benchmark suite to include Temporal Fusion Transformers (TFT) or LSTM networks for multi-horizon probabilistic forecasts.
3. **Automated Retraining Pipelines:** Integrating Apache Airflow or Prefect to trigger scheduled monthly retraining as new meter telemetry arrives.
4. **Demand Response Simulation:** Adding a simulation module allowing utilities to model consumer load-shedding incentives during critical peak hours.

---

## 14. License
Distributed under the MIT License. Developed for enterprise energy analytics and smart grid modernization.
