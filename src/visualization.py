"""
Visualization Module for SmartMeter AI.
Provides modern, interactive Plotly charts with a dark SaaS theme
(navy/charcoal background with cyan and emerald accents).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any

# Modern Dark SaaS Palette
THEME = {
    "bg_color": "#0e1117",
    "card_bg": "#161b26",
    "text_color": "#f1f5f9",
    "muted_text": "#94a3b8",
    "grid_color": "#232936",
    "primary": "#00d2be",      # Electric Cyan/Teal
    "secondary": "#10b981",    # Emerald
    "accent": "#38bdf8",       # Sky Blue
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Rose/Red
    "purple": "#a855f7"        # Electric Violet
}


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Applies a consistent, sleek SaaS dark theme to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "y": 0.95,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": THEME["text_color"], "family": "Inter, Segoe UI, sans-serif"}
        },
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
        font={"color": THEME["muted_text"], "family": "Inter, Segoe UI, sans-serif"},
        height=height,
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        xaxis=dict(
            gridcolor=THEME["grid_color"],
            linecolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            tickfont={"color": THEME["muted_text"]}
        ),
        yaxis=dict(
            gridcolor=THEME["grid_color"],
            linecolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            tickfont={"color": THEME["muted_text"]}
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Inter, sans-serif"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={"color": THEME["text_color"]}
        )
    )
    return fig


def plot_system_consumption_trend(df: pd.DataFrame) -> go.Figure:
    """Aggregated hourly system electricity consumption trend."""
    # Compute system-wide aggregate by timestamp
    agg_df = df.groupby("timestamp", as_index=False)["consumption_kwh"].agg(["sum", "mean"]).reset_index()
    agg_df.rename(columns={"sum": "total_kwh", "mean": "avg_kwh"}, inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg_df["timestamp"],
        y=agg_df["avg_kwh"],
        mode="lines",
        name="Average Consumption (kWh)",
        line=dict(color=THEME["primary"], width=1.8),
        hovertemplate="<b>%{x|%b %d, %Y %H:%M}</b><br>Avg: %{y:.3f} kWh<extra></extra>"
    ))

    _apply_dark_layout(fig, title="System-Wide Electricity Consumption Trend (2025)", height=380)
    fig.update_xaxes(title="Timestamp")
    fig.update_yaxes(title="Average Consumption (kWh)")
    return fig


def plot_hourly_profile(df: pd.DataFrame) -> go.Figure:
    """Average electricity consumption by hour of the day (diurnal curve)."""
    hourly_stats = df.groupby("hour")["consumption_kwh"].agg(["mean", "median", "std"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_stats["hour"],
        y=hourly_stats["mean"],
        name="Mean Consumption",
        marker=dict(
            color=hourly_stats["mean"],
            colorscale=[[0, "#0369a1"], [0.5, "#00d2be"], [1, "#10b981"]],
            showscale=False
        ),
        hovertemplate="Hour %{x}:00<br>Mean: %{y:.3f} kWh<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["median"],
        mode="lines+markers",
        name="Median Consumption",
        line=dict(color=THEME["accent"], width=2.5, dash="dash"),
        marker=dict(size=6)
    ))

    _apply_dark_layout(fig, title="Diurnal Load Profile (24-Hour Cycle)", height=380)
    fig.update_xaxes(title="Hour of Day", tickmode="linear", tick0=0, dtick=2)
    fig.update_yaxes(title="Consumption (kWh)")
    return fig


def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly electricity demand volume and averages."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = df.groupby("month")["consumption_kwh"].agg(["sum", "mean"]).reset_index()
    monthly["month_label"] = monthly["month"].apply(lambda m: month_names[int(m) - 1] if 1 <= m <= 12 else str(m))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month_label"],
        y=monthly["sum"] / 1000.0,
        name="Total MWh",
        marker_color=THEME["accent"],
        hovertemplate="%{x}<br>Total: %{y:.1f} MWh<extra></extra>"
    ))

    _apply_dark_layout(fig, title="Monthly Electricity Consumption (MWh)", height=380)
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Total Demand (MWh)")
    return fig


def plot_top_meters(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top N meters ranked by total electricity consumption."""
    top_df = (
        df.groupby("meter_id")["consumption_kwh"]
        .sum()
        .reset_index()
        .sort_values("consumption_kwh", ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=top_df["consumption_kwh"] / 1000.0,
        y=top_df["meter_id"],
        orientation="h",
        marker=dict(
            color=top_df["consumption_kwh"],
            colorscale=[[0, "#38bdf8"], [1, "#00d2be"]],
            showscale=False
        ),
        hovertemplate="%{y}: %{x:.2f} MWh<extra></extra>"
    ))

    _apply_dark_layout(fig, title=f"Top {top_n} Smart Meters by Annual Consumption", height=380)
    fig.update_xaxes(title="Total Annual Consumption (MWh)")
    fig.update_yaxes(title="Meter ID")
    return fig


def plot_actual_vs_predicted(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Consumption"
) -> go.Figure:
    """Compares actual vs predicted time-series consumption."""
    plot_df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "actual": y_true,
        "predicted": y_pred
    }).sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["actual"],
        mode="lines",
        name="Actual Consumption",
        line=dict(color="#94a3b8", width=1.5),
        opacity=0.75
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["predicted"],
        mode="lines",
        name="Predicted Demand",
        line=dict(color=THEME["primary"], width=2.2)
    ))

    _apply_dark_layout(fig, title=title, height=420)
    fig.update_xaxes(title="Timestamp")
    fig.update_yaxes(title="Consumption (kWh)")
    return fig


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Forecasting error (residuals) distribution."""
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=60,
        marker_color=THEME["secondary"],
        opacity=0.8,
        name="Residuals",
        hovertemplate="Error: %{x:.3f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=THEME["danger"], line_width=1.5)

    _apply_dark_layout(fig, title="Forecasting Error Distribution (Residuals = Actual - Pred)", height=380)
    fig.update_xaxes(title="Residual (kWh)")
    fig.update_yaxes(title="Frequency")
    return fig


def plot_recursive_forecast(
    forecast_df: pd.DataFrame,
    recent_history_df: Optional[pd.DataFrame] = None,
    meter_id: str = ""
) -> go.Figure:
    """
    Renders 24-hour recursive forecast with 95% confidence interval band
    and preceding 48 hours of context.
    """
    fig = go.Figure()

    # Preceding actuals if provided
    if recent_history_df is not None and len(recent_history_df) > 0:
        hist_tail = recent_history_df.tail(48).copy()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(hist_tail["timestamp"]),
            y=hist_tail["consumption_kwh"],
            mode="lines+markers",
            name="Recent Actuals (48h)",
            line=dict(color="#94a3b8", width=1.8),
            marker=dict(size=4)
        ))

    # Confidence interval band (upper bound then lower bound reversed)
    ts = pd.to_datetime(forecast_df["timestamp"])
    fig.add_trace(go.Scatter(
        x=pd.concat([ts, ts[::-1]]),
        y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(0, 210, 190, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Band"
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=ts,
        y=forecast_df["predicted_kwh"],
        mode="lines+markers",
        name="24h Forecast (Recursive)",
        line=dict(color=THEME["primary"], width=2.8),
        marker=dict(size=6, color=THEME["primary"]),
        hovertemplate="<b>+%{customdata}h (%{x|%a %H:%M})</b><br>Forecast: %{y:.3f} kWh<br>CI: [%{text:.3f} - %{hovertext:.3f}]<extra></extra>",
        customdata=forecast_df["forecast_step"],
        text=forecast_df["lower_bound"],
        hovertext=forecast_df["upper_bound"]
    ))

    title_text = f"24-Hour Ahead Electricity Demand Forecast: {meter_id}" if meter_id else "24-Hour Ahead Electricity Demand Forecast"
    _apply_dark_layout(fig, title=title_text, height=440)
    fig.update_xaxes(title="Timestamp")
    fig.update_yaxes(title="Demand (kWh)")
    return fig


def plot_anomaly_timeline(df: pd.DataFrame, sample_n: int = 15000) -> go.Figure:
    """Interactive timeline showing normal vs anomalous consumption readings."""
    if len(df) > sample_n:
        anom_rows = df[df["anomaly_flag"] == 1]
        norm_pool = df[df["anomaly_flag"] == 0]

        # Allocate budget between anomalies and normal background readings
        max_anom = min(len(anom_rows), max(1000, sample_n // 4))
        if len(anom_rows) > max_anom and max_anom > 0:
            anom_sample = anom_rows.sample(n=max_anom, random_state=42)
        else:
            anom_sample = anom_rows

        budget_remaining = max(0, sample_n - len(anom_sample))
        n_norm = min(budget_remaining, len(norm_pool))

        if n_norm > 0 and len(norm_pool) > 0:
            norm_sample = norm_pool.sample(n=n_norm, random_state=42)
        else:
            norm_sample = norm_pool.head(0)

        plot_df = pd.concat([anom_sample, norm_sample]).sort_values("timestamp")
    else:
        plot_df = df.sort_values("timestamp")

    normal_df = plot_df[plot_df["anomaly_flag"] == 0]
    anomaly_df = plot_df[plot_df["anomaly_flag"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_df["timestamp"],
        y=normal_df["consumption_kwh"],
        mode="markers",
        name="Normal Readings",
        marker=dict(color="#334155", size=3, opacity=0.4),
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=anomaly_df["timestamp"],
        y=anomaly_df["consumption_kwh"],
        mode="markers",
        name="Detected Anomalies",
        marker=dict(
            color=THEME["danger"],
            size=7,
            symbol="diamond",
            line=dict(width=1, color="#ffffff")
        ),
        hovertemplate="<b>Anomaly Detected</b><br>Meter: %{customdata[0]}<br>Time: %{x|%Y-%m-%d %H:%M}<br>Consumption: %{y:.2f} kWh<br>Type: %{customdata[1]}<br>Score: %{customdata[2]:.2f}<extra></extra>",
        customdata=anomaly_df[["meter_id", "anomaly_type", "anomaly_score"]].values
    ))

    _apply_dark_layout(fig, title="Electricity Consumption Anomalies Timeline", height=420)
    fig.update_xaxes(title="Timestamp")
    fig.update_yaxes(title="Consumption (kWh)")
    return fig


def plot_anomaly_types(df: pd.DataFrame) -> go.Figure:
    """Distribution of detected anomaly behavioral categories."""
    anom_only = df[df["anomaly_flag"] == 1]
    type_counts = anom_only["anomaly_type"].value_counts().reset_index()
    type_counts.columns = ["anomaly_type", "count"]

    fig = go.Figure(go.Pie(
        labels=type_counts["anomaly_type"],
        values=type_counts["count"],
        hole=0.55,
        marker=dict(colors=[THEME["danger"], THEME["warning"], THEME["purple"], THEME["accent"]]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>"
    ))

    _apply_dark_layout(fig, title="Detected Anomaly Categorization", height=380)
    return fig


def plot_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """Renders confusion matrix as a sleek dark heatmap."""
    labels = ["Normal", "Anomaly"]
    cm_arr = np.array(cm)
    total = cm_arr.sum()

    text_matrix = [
        [f"{cm_arr[i][j]:,}<br>({cm_arr[i][j]/total*100:.1f}%)" for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=["Pred Normal (0)", "Pred Anomaly (1)"],
        y=["Actual Normal (0)", "Actual Anomaly (1)"],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, "#161b26"], [0.5, "#0284c7"], [1, "#00d2be"]],
        showscale=False
    ))

    _apply_dark_layout(fig, title="Isolation Forest Confusion Matrix vs Ground Truth", height=380)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Renders Random Forest feature importance bar chart."""
    top_df = importance_df.sort_values("importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=top_df["importance"],
        y=top_df["feature"],
        orientation="h",
        marker=dict(
            color=top_df["importance"],
            colorscale=[[0, "#0284c7"], [1, "#00d2be"]],
            showscale=False
        ),
        hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
    ))

    _apply_dark_layout(fig, title=f"Top {top_n} Predictive Features (Random Forest)", height=420)
    fig.update_xaxes(title="Feature Importance (MDI / Gini)")
    fig.update_yaxes(title="Feature")
    return fig


def plot_meter_weekly_heatmap(meter_df: pd.DataFrame) -> go.Figure:
    """Heatmap of average consumption by Day of Week vs Hour of Day for a specific meter."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = meter_df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="consumption_kwh",
        aggfunc="mean"
    ).reindex(index=range(7), columns=range(24))

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=[days[i] for i in range(7)],
        colorscale="Viridis",
        hoverongaps=False,
        hovertemplate="%{y} at %{x}<br>Avg: %{z:.3f} kWh<extra></extra>"
    ))

    _apply_dark_layout(fig, title="Household Usage Signature (Day of Week vs Hour)", height=380)
    return fig
