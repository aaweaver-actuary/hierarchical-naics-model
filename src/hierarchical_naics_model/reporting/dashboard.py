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

    naics_effect_cols = sorted(
        [
            col
            for col in scored.columns
            if col.startswith("naics_L") and col.endswith("_effect")
        ],
        key=lambda name: int(name.split("_")[1][1:]),
    )
    zip_effect_cols = sorted(
        [
            col
            for col in scored.columns
            if col.startswith("zip_L") and col.endswith("_effect")
        ],
        key=lambda name: int(name.split("_")[1][1:]),
    )
    component_cols = naics_effect_cols + zip_effect_cols

    if component_cols:
        sum_expr = pl.sum_horizontal([pl.col(col) for col in component_cols])
        intercept_expr = pl.col("eta") - sum_expr
    else:
        intercept_expr = pl.col("eta")

    intercept_mean = float(scored.select(intercept_expr.mean()).item())
    intercept_abs = float(scored.select(intercept_expr.abs().mean()).item())

    component_display_order: list[tuple[str, float, float]] = []
    if np.isfinite(intercept_mean) and np.isfinite(intercept_abs):
        component_display_order.append(("Intercept β0", intercept_mean, intercept_abs))

    def _effect_label(raw: str) -> str:
        family, level, _ = raw.split("_")
        return f"{family.upper()} {level.upper()}"

    for col in naics_effect_cols + zip_effect_cols:
        mean_val = float(scored.select(pl.col(col).mean()).item())
        abs_val = float(scored.select(pl.col(col).abs().mean()).item())
        if not np.isfinite(mean_val) and not np.isfinite(abs_val):
            continue
        component_display_order.append((_effect_label(col), mean_val, abs_val))

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Calibration curve",
            "Lift & cumulative gain",
            "Hierarchy contribution profile",
            "Predicted probability distribution",
            "Predicted vs. true probability",
            "Hierarchy decision flow",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
        specs=[
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}],
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

    if component_display_order:
        fig.add_trace(
            go.Bar(
                x=[label for label, _, _ in component_display_order],
                y=[mean for _, mean, _ in component_display_order],
                name="Mean contribution",
            ),
            row=1,
            col=3,
        )
    else:
        fig.add_trace(
            go.Bar(x=["None"], y=[0.0], name="Mean contribution"),
            row=1,
            col=3,
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

    node_labels = ["Logit η"]
    node_index = {"Logit η": 0}
    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_labels: list[str] = []

    def _ensure_node(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(node_labels)
            node_labels.append(label)
        return node_index[label]

    target_idx = node_index["Logit η"]
    for label, _, mean_abs in component_display_order:
        if not np.isfinite(mean_abs) or mean_abs <= 0.0:
            continue
        src_idx = _ensure_node(label)
        sources.append(src_idx)
        targets.append(target_idx)
        values.append(mean_abs)
        link_labels.append(f"{label}: mean |effect| {mean_abs:.3f}")

    if values:
        fig.add_trace(
            go.Sankey(
                node=dict(label=node_labels),
                link=dict(
                    source=sources, target=targets, value=values, label=link_labels
                ),
                arrangement="snap",
            ),
            row=2,
            col=3,
        )
    else:
        fig.add_trace(
            go.Indicator(mode="number", value=0.0, title={"text": "No hierarchy flow"}),
            row=2,
            col=3,
        )

    fig.update_layout(template="plotly_white")
    fig.update_xaxes(title="Probability bin", row=1, col=1)
    fig.update_yaxes(title="Mean", row=1, col=1)
    fig.update_xaxes(title="Top k%", row=1, col=2)
    fig.update_yaxes(title="Lift", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title="Cumulative gain", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title="Component", row=1, col=3)
    fig.update_yaxes(title="Mean contribution", row=1, col=3)
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

    decision_flow = {
        "node_labels": node_labels,
        "sources": sources,
        "targets": targets,
        "values": values,
        "mean_contributions": {
            label: mean for label, mean, _ in component_display_order
        },
    }

    payload: DashboardPayload = {
        "figure": fig,
        "fit_stats": fit_stats,
        "validation": validation_stats,
        "test_metrics": test_metrics,
        "decision_flow": decision_flow,
    }
    if html_path is not None:
        payload["artifacts"] = {"html_path": html_path}
    return payload
