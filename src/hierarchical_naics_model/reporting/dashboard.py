from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

DashboardPayload = MutableMapping[str, Any]


def _as_float(value: Any) -> float:
    try:
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                return float("nan")
            return float(arr.reshape(-1)[0]) if arr.size == 1 else float(arr.mean())
        return float(value)
    except Exception:  # noqa: BLE001 - defensive conversion
        return float("nan")


def _extract_loo_stats(loo_report: Any) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for attr, alias in (
        ("loo", "loo"),
        ("loo_se", "loo_se"),
        ("p_loo", "p_loo"),
        ("elpd", "elpd"),
        ("elpd_loo", "elpd_loo"),
        ("elpd_loo_se", "elpd_loo_se"),
    ):
        if hasattr(loo_report, attr):
            stats[alias] = _as_float(getattr(loo_report, attr))
    if "loo" not in stats and "elpd_loo" in stats:
        stats["loo"] = stats["elpd_loo"]
    if hasattr(loo_report, "pareto_k"):
        try:
            stats["pareto_k_max"] = float(np.asarray(loo_report.pareto_k).max())
        except Exception:  # noqa: BLE001 - defensive
            stats["pareto_k_max"] = float("nan")
    return stats


def _prepare_reliability_plot(reliability: pl.DataFrame) -> Dict[str, list[float]]:
    df = reliability.with_columns(
        ((pl.col("bin_low") + pl.col("bin_high")) / 2.0).alias("mid")
    )
    return {
        "mid": df["mid"].to_list(),
        "mean_p": df["mean_p"].to_list(),
        "mean_y": df["mean_y"].to_list(),
        "gap": df["gap"].to_list(),
    }


def _prepare_ranking_plot(summary: pl.DataFrame) -> Dict[str, list[float]]:
    ordered = summary.sort("k_pct")
    return {
        "k_pct": ordered["k_pct"].to_list(),
        "lift": ordered["lift"].to_list(),
        "cum_gain": ordered["cum_gain"].to_list(),
    }


def build_dashboard(
    *,
    train_summary: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ranking: Mapping[str, Any],
    loo: Any,
    scored_test: pl.DataFrame,
    parameter_alignment: Mapping[str, float] | None = None,
    output_dir: Path | str | None = None,
) -> DashboardPayload:
    reliability_val = calibration["reliability"]
    ranking_val = ranking["summary"]

    reliability_df = (
        reliability_val
        if isinstance(reliability_val, pl.DataFrame)
        else pl.DataFrame(reliability_val)
    )
    ranking_df = (
        ranking_val
        if isinstance(ranking_val, pl.DataFrame)
        else pl.DataFrame(ranking_val)
    )

    scored = scored_test.clone()
    loo_stats = _extract_loo_stats(loo)

    reliability_data = _prepare_reliability_plot(reliability_df)
    ranking_data = _prepare_ranking_plot(ranking_df)

    test_probs = scored["p"].cast(pl.Float64).to_list()
    true_probs = (
        scored["p_true"].cast(pl.Float64).to_list()
        if "p_true" in scored.columns
        else [float("nan")] * len(test_probs)
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Calibration curve",
            "Lift & cumulative gain",
            "Predicted probability distribution",
            "Predicted vs. true probability",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "xy"}, {"secondary_y": True}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=reliability_data["mid"],
            y=reliability_data["mean_y"],
            mode="lines+markers",
            name="Observed",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reliability_data["mid"],
            y=reliability_data["mean_p"],
            mode="lines+markers",
            name="Predicted",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ranking_data["k_pct"],
            y=ranking_data["lift"],
            mode="lines+markers",
            name="Lift",
        ),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ranking_data["k_pct"],
            y=ranking_data["cum_gain"],
            mode="lines+markers",
            name="Cumulative gain",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.add_trace(
        go.Histogram(x=test_probs, nbinsx=20, name="Prediction density"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=true_probs,
            y=test_probs,
            mode="markers",
            name="Pred vs true",
            marker=dict(size=6, opacity=0.6),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Ideal",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(template="plotly_white")
    fig.update_xaxes(title="Probability bin", row=1, col=1)
    fig.update_yaxes(title="Mean", row=1, col=1)
    fig.update_xaxes(title="Top k%", row=1, col=2)
    fig.update_yaxes(title="Lift", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title="Cumulative gain", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title="p_hat", row=2, col=1)
    fig.update_yaxes(title="Count", row=2, col=1)
    fig.update_xaxes(title="True probability", row=2, col=2)
    fig.update_yaxes(title="Predicted probability", row=2, col=2)

    parameter_alignment = dict(parameter_alignment or {})
    ece = float(calibration["ece"])
    brier = float(calibration["brier"])
    log_loss = float(calibration["log_loss"])

    fit_text_lines = [
        "Fit overview",
        f"N train: {train_summary.get('n_train', 'NA')}",
        f"Train hit rate: {train_summary.get('train_positive_rate', float('nan')):.3f}",
        f"LOO: {loo_stats.get('loo', float('nan')):.2f} ± {loo_stats.get('loo_se', float('nan')):.2f}",
    ]
    if parameter_alignment:
        fit_text_lines.extend(
            f"{name}: {value:.2f}" for name, value in parameter_alignment.items()
        )

    validation_text = [
        "Validation metrics",
        f"ECE: {ece:.3f}",
        f"Brier: {brier:.3f}",
        f"Log loss: {log_loss:.3f}",
    ]

    if "pareto_k_max" in loo_stats and not np.isnan(loo_stats["pareto_k_max"]):
        validation_text.append(f"Pareto k max: {loo_stats['pareto_k_max']:.2f}")

    if "any_backoff" in scored.columns:
        backoff_rate = float(scored["any_backoff"].cast(pl.Float64).mean())
    else:
        backoff_rate = 0.0
    test_text = [
        "Test performance",
        f"N test: {train_summary.get('n_test', 'NA')}",
        f"Test hit rate: {train_summary.get('test_positive_rate', float('nan')):.3f}",
        f"Backoff rate: {backoff_rate:.3f}",
    ]

    fig.add_annotation(
        x=0.01,
        y=1.15,
        xref="paper",
        yref="paper",
        text="<br>".join(fit_text_lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(240,240,240,0.7)",
        borderpad=8,
    )
    fig.add_annotation(
        x=0.5,
        y=1.15,
        xref="paper",
        yref="paper",
        text="<br>".join(validation_text),
        showarrow=False,
        align="left",
        bgcolor="rgba(240,240,240,0.7)",
        borderpad=8,
    )
    fig.add_annotation(
        x=0.99,
        y=1.15,
        xref="paper",
        yref="paper",
        text="<br>".join(test_text),
        showarrow=False,
        align="right",
        bgcolor="rgba(240,240,240,0.7)",
        borderpad=8,
    )

    html_path: Path | None = None
    if output_dir is not None:
        html_path = Path(output_dir) / "model_dashboard.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(
            fig, file=str(html_path), auto_open=False, include_plotlyjs="cdn"
        )

    fit_stats: Dict[str, Any] = dict(train_summary)
    fit_stats.update(loo_stats)
    fit_stats.update(parameter_alignment)

    validation_stats = {
        "loo": loo_stats.get("loo", float("nan")),
        "loo_se": loo_stats.get("loo_se", float("nan")),
        "pareto_k_max": loo_stats.get("pareto_k_max", float("nan")),
        "ece": ece,
        "brier": brier,
        "log_loss": log_loss,
    }

    test_metrics = {
        "brier": brier,
        "ece": ece,
        "log_loss": log_loss,
        "backoff_rate": backoff_rate,
        "avg_pred": float(scored["p"].mean()),
    }

    payload: DashboardPayload = {
        "figure": fig,
        "fit_stats": fit_stats,
        "validation": validation_stats,
        "test_metrics": test_metrics,
    }
    if html_path is not None:
        payload["artifacts"] = {"html_path": html_path}
    return payload
