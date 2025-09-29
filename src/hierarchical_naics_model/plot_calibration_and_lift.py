from __future__ import annotations
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_reliability_curve(
    df_rel: pd.DataFrame,
    *,
    title: str = "Reliability Curve",
    show_ece: bool = True,
    custom_layout: Optional[Dict[str, Any]] = None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_rel["p_avg"],
            y=df_rel["y_rate"],
            mode="lines+markers",
            name="Empirical",
            line=dict(color="blue"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="gray"),
        )
    )
    if show_ece and "abs_gap" in df_rel:
        ece = (df_rel["count"] / df_rel["count"].sum() * df_rel["abs_gap"]).sum()
        fig.add_annotation(
            x=0.5, y=0.05, text=f"ECE: {ece:.4f}", showarrow=False, font=dict(size=14)
        )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Probability",
        yaxis_title="Empirical Rate",
        template="plotly_white",
        **(custom_layout or {}),
    )
    return fig


def plot_lift_and_gain(
    df_rank: pd.DataFrame,
    *,
    title: str = "Lift & Cumulative Gain",
    custom_layout: Optional[Dict[str, Any]] = None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_rank["k_pct"],
            y=df_rank["lift_at_k"],
            mode="lines+markers",
            name="Lift@k",
            line=dict(color="green"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_rank["k_pct"],
            y=df_rank["cum_gain_pct"],
            mode="lines+markers",
            name="Cumulative Gain",
            line=dict(color="orange"),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Top k%",
        yaxis_title="Metric Value",
        template="plotly_white",
        **(custom_layout or {}),
    )
    return fig


def plot_precision_at_k(
    df_rank: pd.DataFrame,
    *,
    title: str = "Precision@k",
    custom_layout: Optional[Dict[str, Any]] = None,
):
    fig = px.bar(
        df_rank,
        x="k_pct",
        y="precision_at_k",
        title=title,
        labels={"k_pct": "Top k%", "precision_at_k": "Precision@k"},
        template="plotly_white",
    )
    if custom_layout:
        fig.update_layout(**custom_layout)
    return fig


def plot_brier_logloss(
    summary: Dict[str, Any],
    *,
    title: str = "Brier & Log-Loss",
    custom_layout: Optional[Dict[str, Any]] = None,
):
    metrics = ["brier", "log_loss"]
    values = [summary[m] for m in metrics]
    fig = px.bar(
        x=metrics,
        y=values,
        title=title,
        labels={"x": "Metric", "y": "Value"},
        template="plotly_white",
    )
    if custom_layout:
        fig.update_layout(**custom_layout)
    return fig
