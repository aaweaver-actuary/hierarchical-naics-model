from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import arviz as az
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
    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            stats["pareto_k_max"] = float("nan")
    return stats


def _prepare_reliability_plot(reliability: pl.DataFrame) -> dict[str, list[float]]:
    df = reliability.with_columns(
        ((pl.col("bin_low") + pl.col("bin_high")) / 2.0).alias("mid")
    )
    return {
        "mid": df["mid"].to_list(),
        "mean_p": df["mean_p"].to_list(),
        "mean_y": df["mean_y"].to_list(),
        "gap": df["gap"].to_list(),
    }


def _prepare_ranking_plot(summary: pl.DataFrame) -> dict[str, list[float]]:
    ordered = summary.sort("k_pct")
    return {
        "k_pct": ordered["k_pct"].to_list(),
        "lift": ordered["lift"].to_list(),
        "cum_gain": ordered["cum_gain"].to_list(),
    }


def _build_decision_flow(component_display_order: Sequence[tuple[str, float, float]]):
    node_labels = ["Logit η"]
    node_index = {"Logit η": 0}
    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []

    def ensure_node(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(node_labels)
            node_labels.append(label)
        return node_index[label]

    target_idx = node_index["Logit η"]
    mean_contrib = []
    for label, mean_val, mean_abs in component_display_order:
        mean_contrib.append(
            {"label": label, "mean": float(mean_val), "abs": float(mean_abs)}
        )
        if not np.isfinite(mean_abs) or mean_abs <= 0.0:
            continue
        src_idx = ensure_node(label)
        sources.append(src_idx)
        targets.append(target_idx)
        values.append(float(mean_abs))

    if values:
        fig = go.Figure(
            go.Sankey(
                node=dict(label=node_labels),
                link=dict(source=sources, target=targets, value=values),
                arrangement="snap",
            )
        )
    else:
        fig = go.Figure(
            go.Indicator(mode="number", value=0.0, title={"text": "No hierarchy flow"})
        )
    fig.update_layout(template="plotly_white")

    decision_flow = {
        "node_labels": node_labels,
        "sources": sources,
        "targets": targets,
        "values": values,
        "mean_contributions": mean_contrib,
    }
    return fig, decision_flow


def _build_hierarchy_sunburst(
    *,
    prefix: str,
    cut_points: Sequence[int],
    level_maps: Sequence[Mapping[str, int]],
    effect_vectors: Sequence[Sequence[float]],
) -> go.Figure:
    root_label = prefix
    labels = [root_label]
    parents = [""]
    values = [0.0]
    total = 0.0
    for level, mapping in enumerate(level_maps):
        effects = (
            np.asarray(effect_vectors[level], dtype=float)
            if level < len(effect_vectors)
            else np.zeros(len(mapping), dtype=float)
        )
        parent_cut = cut_points[level - 1] if level > 0 else None
        for label, idx in mapping.items():
            node_label = f"{prefix} {label}"
            parent_label = (
                root_label if level == 0 else f"{prefix} {label[:parent_cut]}"
            )
            val = float(abs(effects[idx]))
            labels.append(node_label)
            parents.append(parent_label)
            values.append(val)
            total += val
    values[0] = total if total > 0 else 1.0
    fig = go.Figure(
        go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total")
    )
    fig.update_layout(template="plotly_white")
    return fig


def _summarize_hierarchy_levels(
    *,
    cut_points: Sequence[int],
    level_maps: Sequence[Mapping[str, int]],
    base: Sequence[float],
    deltas: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    levels: list[dict[str, Any]] = []
    total_nodes = 0
    overall_max = 0.0

    def _vector_for_level(level_idx: int, mapping: Mapping[str, int]) -> np.ndarray:
        if level_idx == 0:
            vec = np.asarray(base, dtype=float)
        else:
            delta_idx = level_idx - 1
            if delta_idx < len(deltas):
                vec = np.asarray(deltas[delta_idx], dtype=float)
            else:
                vec = np.zeros(0, dtype=float)
        if vec.size != len(mapping):
            padded = np.zeros(len(mapping), dtype=float)
            limit = min(len(mapping), vec.size)
            if limit > 0:
                padded[:limit] = vec[:limit]
            return padded
        return vec

    for idx, mapping in enumerate(level_maps):
        level_vec = _vector_for_level(idx, mapping)
        abs_vec = np.abs(level_vec)
        total_nodes += len(mapping)
        level_max = float(abs_vec.max()) if abs_vec.size else 0.0
        overall_max = max(overall_max, level_max)
        mean_abs = float(abs_vec.mean()) if abs_vec.size else 0.0
        cut_point = int(cut_points[idx]) if idx < len(cut_points) else None
        levels.append(
            {
                "cut_point": cut_point,
                "group_count": len(mapping),
                "mean_abs_effect": mean_abs,
                "max_abs_effect": level_max,
            }
        )

    return {
        "total_nodes": total_nodes,
        "max_abs_effect": overall_max,
        "levels": levels,
    }


def _build_variable_payload(
    idata: Any | None,
    *,
    max_variables: int = 60,
    max_samples: int = 400,
    max_trace: int | None = None,
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    max_total_draws = 0
    if idata is None or not hasattr(idata, "groups"):
        return {
            "variables": variables,
            "max_trace": max_total_draws,
            "base_options": [],
        }
    if "posterior" not in idata.groups():
        return {
            "variables": variables,
            "max_trace": max_total_draws,
            "base_options": [],
        }

    posterior = idata.posterior
    prior_group = idata.prior if "prior" in idata.groups() else None
    try:
        rhat_ds = az.rhat(idata, method="rank")
    except Exception:  # noqa: BLE001
        rhat_ds = None
    try:
        ess_bulk_ds = az.ess(idata, method="bulk")
    except Exception:  # noqa: BLE001
        ess_bulk_ds = None
    ppc_group = (
        idata.posterior_predictive if "posterior_predictive" in idata.groups() else None
    )
    observed_group = idata.observed_data if "observed_data" in idata.groups() else None

    def _autocorrelation_values(series: np.ndarray, max_lag: int) -> list[float]:
        values: list[float] = []
        cleaned = np.asarray(series, dtype=float)
        if cleaned.size == 0:
            return [1.0] + [0.0] * max(0, max_lag)
        cleaned = cleaned[np.isfinite(cleaned)]
        if cleaned.size == 0:
            return [1.0] + [0.0] * max(0, max_lag)
        cleaned = cleaned - cleaned.mean()
        denom = float(np.dot(cleaned, cleaned))
        if denom <= 0 or not np.isfinite(denom):
            return [1.0] + [0.0] * max(0, max_lag)
        values.append(1.0)
        for lag in range(1, max_lag + 1):
            lagged = float(np.dot(cleaned[:-lag], cleaned[lag:]))
            ac_val = lagged / denom if denom != 0 else 0.0
            values.append(float(ac_val))
        return values

    def _autocorrelation_profile(entry_arr: np.ndarray) -> dict[str, Any]:
        if entry_arr.ndim < 2 or entry_arr.shape[1] == 0:
            return {
                "max_lag": 0,
                "lags": [0],
                "per_chain": [],
                "interpretation": (
                    "Autocorrelation near zero beyond the first lag indicates good mixing."
                ),
            }
        draws = int(entry_arr.shape[1])
        chains = int(entry_arr.shape[0])
        max_lag_target = int(min(100, max(draws * 0.1, 1.0)))
        max_lag = int(min(max_lag_target, max(draws - 1, 0)))
        lags = list(range(max_lag + 1))
        per_chain = []
        for chain_idx in range(chains):
            chain_series = np.asarray(entry_arr[chain_idx, :], dtype=float)
            ac_values = _autocorrelation_values(chain_series, max_lag)
            if ac_values:
                ac_values[0] = 1.0
            per_chain.append(
                {
                    "chain_index": int(chain_idx),
                    "values": [float(val) for val in ac_values],
                }
            )
        guidance = (
            "Look for autocorrelation values that drop toward zero within a few lags; "
            "sustained high values imply sluggish mixing and the need for more draws."
        )
        return {
            "max_lag": max_lag,
            "lags": lags,
            "per_chain": per_chain,
            "interpretation": guidance,
        }

    def _gaussian_kde(samples: np.ndarray, support: np.ndarray) -> np.ndarray:
        if samples.size <= 1 or support.size == 0:
            return np.zeros_like(support, dtype=float)
        finite = samples[np.isfinite(samples)]
        if finite.size <= 1:
            return np.zeros_like(support, dtype=float)
        sd = float(np.std(finite, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = max(float(np.abs(finite).mean()), 1e-3)
        bandwidth = 1.06 * sd * (float(finite.size) ** (-1.0 / 5.0))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = sd if sd > 0 else 1.0
        inv_scale = 1.0 / (bandwidth * np.sqrt(2.0 * np.pi))
        diffs = (support[:, None] - finite[None, :]) / bandwidth
        kernel = np.exp(-0.5 * diffs**2)
        density = inv_scale * kernel.mean(axis=1)
        return density

    def _posterior_predictive_overlay(
        var_name: str, index: tuple[Any, ...]
    ) -> dict[str, Any] | None:
        if ppc_group is None or var_name not in ppc_group.data_vars:
            return None  # pragma: no cover - defensive fallback
        try:
            predictive_values = np.asarray(ppc_group[var_name].values, dtype=float)
        except Exception:  # noqa: BLE001
            return None  # pragma: no cover - invalid posterior predictive values
        slicing = (slice(None), slice(None)) + index
        try:
            predictive_slice = predictive_values[slicing]
        except Exception:  # noqa: BLE001
            predictive_slice = predictive_values
        predictive_flat = np.asarray(predictive_slice, dtype=float).reshape(-1)
        predictive_flat = predictive_flat[np.isfinite(predictive_flat)]
        if predictive_flat.size == 0:
            return None  # pragma: no cover - no predictive draws

        observed_flat = np.array([], dtype=float)
        if observed_group is not None and var_name in observed_group.data_vars:
            try:
                observed_values = np.asarray(
                    observed_group[var_name].values, dtype=float
                )
                if index:
                    try:
                        observed_values = observed_values[index]
                    except Exception:  # noqa: BLE001
                        observed_values = observed_values
            except Exception:  # noqa: BLE001
                observed_values = np.array([], dtype=float)
            observed_flat = np.asarray(observed_values, dtype=float).reshape(-1)
            observed_flat = observed_flat[np.isfinite(observed_flat)]

        if observed_flat.size:
            combined = np.concatenate([predictive_flat, observed_flat])
        else:
            combined = predictive_flat

        if combined.size == 0:
            return None  # pragma: no cover - cannot derive support

        x_min = float(np.min(combined))
        x_max = float(np.max(combined))
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return None  # pragma: no cover - invalid support bounds
        if x_max <= x_min:
            span = (
                1.0
                if not np.isfinite(predictive_flat.std())
                else float(predictive_flat.std())
            )
            if span <= 0 or not np.isfinite(span):
                span = 1.0
            x_min -= span
            x_max += span
        x_support = np.linspace(x_min, x_max, num=200, dtype=float)
        kde_values = _gaussian_kde(predictive_flat, x_support)
        observed_samples = (
            observed_flat[:500].astype(float).tolist() if observed_flat.size else []
        )
        return {
            "kde": {
                "x": x_support.astype(float).tolist(),
                "y": kde_values.astype(float).tolist(),
                "normalization": "density",
            },
            "observed": {
                "samples": observed_samples,
                "histnorm": "probability density",
            },
            "x_range": [x_min, x_max],
            "sample_count": int(predictive_flat.size),
        }

    def _diag_value(ds: Any, name: str, index: tuple[Any, ...]) -> float:
        if ds is None or name not in ds:
            return float("nan")
        values = np.asarray(ds[name].values, dtype=float)
        if values.size == 0:
            return float("nan")
        if values.ndim == 0:
            return float(values)
        if index:
            try:
                return float(values[index])
            except Exception:  # noqa: BLE001
                return float(values.reshape(-1)[0])
        return float(values.reshape(-1)[0])

    def _prior_parameters(var_name: str) -> tuple[float, float]:
        mean = 0.0
        sd = 1.0
        if prior_group is None or var_name not in prior_group.data_vars:
            return mean, sd
        try:
            values = np.asarray(prior_group[var_name].values, dtype=float)
        except Exception:  # noqa: BLE001
            return mean, sd
        values = values.reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return mean, sd
        mean = float(values.mean())
        if values.size > 1:
            sd_val = float(values.std(ddof=1))
        else:
            sd_val = float(values.std(ddof=0))
        if not np.isfinite(sd_val) or sd_val <= 0:
            sd_val = 1.0
        return mean, sd_val

    def _prior_curve(
        mean: float, sd: float, *, fallback_mean: float, fallback_sd: float
    ) -> dict[str, list[float]]:
        curve_sd = sd if np.isfinite(sd) and sd > 0 else fallback_sd
        if not np.isfinite(curve_sd) or curve_sd <= 0:
            curve_sd = 1.0
        curve_mean = mean if np.isfinite(mean) else fallback_mean
        if not np.isfinite(curve_mean):
            curve_mean = 0.0
        x_min = curve_mean - 4.0 * curve_sd
        x_max = curve_mean + 4.0 * curve_sd
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            span_sd = (
                fallback_sd if np.isfinite(fallback_sd) and fallback_sd > 0 else 1.0
            )
            if not np.isfinite(fallback_mean):
                fallback_mean = 0.0
            x_min = fallback_mean - 4.0 * span_sd
            x_max = fallback_mean + 4.0 * span_sd
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = -4.0, 4.0
        x_vals = np.linspace(float(x_min), float(x_max), num=200, dtype=float)
        denom = np.sqrt(2.0 * np.pi) * curve_sd
        if denom == 0 or not np.isfinite(denom):
            curve_sd = 1.0
            denom = np.sqrt(2.0 * np.pi) * curve_sd
        y_vals = 1.0 / denom * np.exp(-0.5 * ((x_vals - curve_mean) / curve_sd) ** 2)
        return {
            "x": x_vals.astype(float).tolist(),
            "y": y_vals.astype(float).tolist(),
        }

    for name, da in posterior.data_vars.items():
        if da.ndim < 2:  # expect chain/draw at least
            continue
        arr = np.asarray(da.values)
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        if arr.ndim > 2:
            entry_shape = arr.shape[2:]
            entry_count = int(np.prod(entry_shape))
        else:
            entry_shape = ()
            entry_count = 1
        chain_dim = arr.shape[0]
        for entry_idx in range(entry_count):
            if len(variables) >= max_variables:
                break
            idx_tuple = np.unravel_index(entry_idx, entry_shape) if entry_shape else ()
            entry_slice = (slice(None), slice(None)) + idx_tuple
            entry_arr = arr[entry_slice]  # shape (chain, draw)
            samples_flat = entry_arr.reshape(-1)
            if samples_flat.size == 0:
                continue
            mean_val = float(samples_flat.mean())
            sd_val = float(samples_flat.std(ddof=1))
            percentiles = np.asarray(
                np.percentile(samples_flat, [5, 50, 95]), dtype=float
            )
            q05 = float(percentiles[0])
            q50 = float(percentiles[1])
            q95 = float(percentiles[2])
            sample_list = samples_flat[:max_samples].astype(float).tolist()
            total_draws = int(entry_arr.shape[1]) if entry_arr.ndim >= 2 else 0
            max_total_draws = max(max_total_draws, total_draws)
            full_chain_values = [
                entry_arr[c, :].astype(float).tolist() for c in range(chain_dim)
            ]
            chain_limit = min(chain_dim, 4)
            chain_samples = full_chain_values[:chain_limit]
            draw_indices = list(range(total_draws))
            index_label = (
                "" if not idx_tuple else "[" + ",".join(str(i) for i in idx_tuple) + "]"
            )
            label = f"{name}{index_label}"
            variable_id = label
            prior_mean, prior_sd = _prior_parameters(name)
            prior_curve = _prior_curve(
                prior_mean,
                prior_sd,
                fallback_mean=mean_val,
                fallback_sd=sd_val if np.isfinite(sd_val) and sd_val > 0 else 1.0,
            )
            prior_info = {
                "mean": float(prior_mean if np.isfinite(prior_mean) else mean_val),
                "sd": float(
                    prior_sd if np.isfinite(prior_sd) and prior_sd > 0 else 1.0
                ),
                "curve": prior_curve,
            }
            autocorr_info = _autocorrelation_profile(entry_arr)
            ppc_overlay = _posterior_predictive_overlay(name, idx_tuple)
            variables.append(
                {
                    "id": variable_id,
                    "label": label,
                    "mean": mean_val,
                    "sd": sd_val,
                    "q05": float(q05),
                    "median": float(q50),
                    "q95": float(q95),
                    "samples": sample_list,
                    "chains": chain_samples,
                    "trace": {
                        "total_draws": total_draws,
                        "total_chains": chain_dim,
                        "draw_indices": draw_indices,
                        "chains": [
                            {
                                "chain_index": int(chain_idx),
                                "values": full_chain_values[chain_idx],
                            }
                            for chain_idx in range(chain_dim)
                        ],
                        "hover": {
                            "mode": "x unified",
                            "tooltip": "all_chains",
                        },
                    },
                    "prior": prior_info,
                    "r_hat": _diag_value(rhat_ds, name, idx_tuple),
                    "ess_bulk": _diag_value(ess_bulk_ds, name, idx_tuple),
                    "autocorrelation": autocorr_info,
                    "ppc": ppc_overlay,
                }
            )
            group = grouped.setdefault(
                name,
                {
                    "base": name,
                    "has_indices": False,
                    "variables": [],
                },
            )
            if idx_tuple:
                group["has_indices"] = True
            display_label = index_label if index_label else name
            group["variables"].append(
                {
                    "id": variable_id,
                    "label": display_label,
                    "index": [int(i) for i in idx_tuple],
                }
            )
        if len(variables) >= max_variables:
            break
    base_options = list(grouped.values())
    base_options.sort(key=lambda item: item["base"])
    for group in base_options:
        group["variables"].sort(key=lambda entry: entry["label"])
    return {
        "variables": variables,
        "max_trace": max_total_draws,
        "base_options": base_options,
    }


def _build_inference_suggestions(
    scored: pl.DataFrame,
    *,
    top_k: int = 5,
) -> Dict[str, list[Dict[str, Any]]]:
    if scored.height == 0 or "p" not in scored.columns:
        return {"naics": [], "zip": []}

    agg_base = [
        pl.col("p").cast(pl.Float64).max().alias("max_p"),
        pl.col("p").cast(pl.Float64).mean().alias("avg_p"),
        pl.len().alias("count"),
    ]

    def _assemble(column: str) -> list[Dict[str, Any]]:
        if column not in scored.columns:
            return []
        subset = scored.filter(pl.col(column).is_not_null()).with_columns(
            pl.col(column).cast(pl.Utf8).alias(column)
        )
        if subset.height == 0:
            return []

        agg_exprs = list(agg_base)
        if "eta" in subset.columns:
            agg_exprs.append(pl.col("eta").cast(pl.Float64).mean().alias("avg_eta"))
        if "any_backoff" in subset.columns:
            agg_exprs.append(
                pl.col("any_backoff").cast(pl.Float64).mean().alias("backoff_rate")
            )

        grouped = (
            subset.group_by(column)
            .agg(agg_exprs)
            .sort("max_p", descending=True)
            .head(top_k)
        )

        suggestions: list[Dict[str, Any]] = []
        for row in grouped.iter_rows(named=True):
            suggestion: Dict[str, Any] = {
                "code": str(row[column]),
                "probability": float(row["max_p"]),
                "avg_probability": float(row["avg_p"]),
                "count": int(row["count"]),
            }
            if "avg_eta" in row and row["avg_eta"] is not None:
                suggestion["avg_eta"] = float(row["avg_eta"])
            if "backoff_rate" in row and row["backoff_rate"] is not None:
                suggestion["backoff_rate"] = float(row["backoff_rate"])
            suggestions.append(suggestion)
        return suggestions

    return {
        "naics": _assemble("NAICS"),
        "zip": _assemble("ZIP"),
    }


def _build_inference_bundle(
    *,
    effects: Mapping[str, Any],
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    prefix_fill: str,
    scored: pl.DataFrame | None = None,
    top_suggestions: int = 5,
) -> Dict[str, Any]:
    suggestions = (
        _build_inference_suggestions(scored, top_k=top_suggestions)
        if scored is not None
        else {"naics": [], "zip": []}
    )
    return {
        "effects": {
            "beta0": float(effects["beta0"]),
            "naics_base": np.asarray(effects["naics_base"], dtype=float).tolist(),
            "naics_deltas": [
                np.asarray(arr, dtype=float).tolist()
                for arr in effects.get("naics_deltas", [])
            ],
            "zip_base": np.asarray(effects["zip_base"], dtype=float).tolist(),
            "zip_deltas": [
                np.asarray(arr, dtype=float).tolist()
                for arr in effects.get("zip_deltas", [])
            ],
        },
        "naics_cut_points": list(int(c) for c in naics_cut_points),
        "zip_cut_points": list(int(c) for c in zip_cut_points),
        "naics_level_maps": [
            {str(k): int(v) for k, v in mapping.items()} for mapping in naics_level_maps
        ],
        "zip_level_maps": [
            {str(k): int(v) for k, v in mapping.items()} for mapping in zip_level_maps
        ],
        "prefix_fill": prefix_fill,
        "suggestions": suggestions,
    }


def _format_float(value: Any, fmt: str = ".3f") -> str:
    try:
        return format(float(value), fmt)
    except Exception:  # noqa: BLE001
        return "NA"


def _model_summary_cards(
    fit_stats: Mapping[str, Any],
    validation_stats: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> str:
    cards = [
        ("Training rows", fit_stats.get("n_train", "NA")),
        ("Training hit rate", _format_float(fit_stats.get("train_positive_rate"))),
        ("LOO", _format_float(fit_stats.get("loo"))),
        ("Brier", _format_float(validation_stats.get("brier"))),
        ("ECE", _format_float(validation_stats.get("ece"))),
        ("Backoff rate", _format_float(test_metrics.get("backoff_rate"))),
    ]
    fragments = []
    for title, value in cards:
        fragments.append(
            """
            <div class="card">
              <div class="card-title">{title}</div>
              <div class="card-value">{value}</div>
            </div>
            """.format(title=html.escape(str(title)), value=html.escape(str(value)))
        )
    return "\n".join(fragments)


def _render_dashboard_html(
    *,
    plots_json: str,
    variable_json: str,
    inference_json: str,
    decision_flow: str,
    model_cards_html: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Hierarchical NAICS Model Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.30.0.min.js\"></script>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #fafafa; color: #222; }}
    #dashboard {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .tab-bar {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    .tab-button {{ border: none; background: #e0e0e0; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-size: 15px; }}
    .tab-button.active {{ background: #2563eb; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.12); }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .plot {{ min-height: 360px; margin: 20px 0; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    .card-title {{ font-size: 13px; color: #666; }}
    .card-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .variable-panel {{ display: flex; flex-direction: column; gap: 16px; }}
    .variable-controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    select {{ padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }}
    .variable-controls select {{ transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out; }}
    .variable-controls select:focus,
    .variable-controls select:hover {{ border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18); outline: none; }}
    .variable-controls select.active {{ border-color: #1e40af; box-shadow: 0 0 0 2px rgba(30, 64, 175, 0.22); }}
    .variable-summary {{ background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); font-size: 13px; }}
    .hierarchy-flex {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .inference-form {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }}
    .inference-form input {{ padding: 6px 10px; border-radius: 4px; border: 1px solid #bbb; font-size: 14px; width: 140px; }}
    .inference-form button {{ padding: 8px 16px; border-radius: 6px; background: #2563eb; color: white; border: none; cursor: pointer; }}
    .inference-form button:hover {{ background: #1e4fc6; }}
    .inference-output {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); font-size: 14px; line-height: 1.6; }}
    table.flow-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    table.flow-table th, table.flow-table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    table.flow-table th {{ background: #f0f4ff; }}
  </style>
</head>
<body>
  <div id=\"dashboard\">
    <div class=\"tab-bar\">
      <button class=\"tab-button active\" data-tab=\"model-fit\">Model Fit</button>
      <button class=\"tab-button\" data-tab=\"variables\">Variables</button>
      <button class=\"tab-button\" data-tab=\"hierarchy\">Hierarchy</button>
      <button class=\"tab-button\" data-tab=\"validation\">Validation</button>
      <button class=\"tab-button\" data-tab=\"inference\">Inference</button>
    </div>

    <div class=\"tab-content active\" id=\"tab-model-fit\">
      <div class=\"card-grid\">{model_cards_html}</div>
      <div id=\"plot-model-fit-contrib\" class=\"plot\"></div>
    </div>

    <div class=\"tab-content\" id=\"tab-variables\">
      <div class=\"variable-panel\">
                <div class=\"variable-controls\">
                    <label for=\"variable-base-select\">Parameter:</label>
                    <select id=\"variable-base-select\"></select>
                    <label for=\"variable-index-select\" id=\"variable-index-label\" style=\"display:none;\">Index:</label>
                    <select id=\"variable-index-select\" style=\"display:none;\"></select>
                </div>
        <div class=\"variable-summary\" id=\"variable-summary\"></div>
        <div id=\"plot-variable-density\" class=\"plot\"></div>
        <div id=\"plot-variable-trace\" class=\"plot\"></div>
      </div>
    </div>

    <div class=\"tab-content\" id=\"tab-hierarchy\">
      <div class=\"hierarchy-flex\">
        <div id=\"plot-hierarchy-naics\" class=\"plot\"></div>
        <div id=\"plot-hierarchy-zip\" class=\"plot\"></div>
        <div id=\"plot-hierarchy-flow\" class=\"plot\"></div>
      </div>
      <div id=\"hierarchy-flow-table\"></div>
    </div>

    <div class=\"tab-content\" id=\"tab-validation\">
      <div id=\"plot-validation-calibration\" class=\"plot\"></div>
      <div id=\"plot-validation-ranking\" class=\"plot\"></div>
      <div id=\"plot-validation-hist\" class=\"plot\"></div>
      <div id=\"plot-validation-scatter\" class=\"plot\"></div>
    </div>

    <div class=\"tab-content\" id=\"tab-inference\">
      <div class=\"inference-form\">
        <label for=\"inference-naics\">NAICS:</label>
        <input id=\"inference-naics\" placeholder=\"e.g., 522120\" />
        <label for=\"inference-zip\">ZIP:</label>
        <input id=\"inference-zip\" placeholder=\"e.g., 45242\" />
        <button id=\"inference-score\">Score submission</button>
      </div>
      <div id=\"inference-output\" class=\"inference-output\">Enter NAICS and ZIP to score a submission.</div>
    </div>
  </div>

  <script>
    const PLOTS = {plots_json};
    const VARIABLE_DATA = {variable_json};
    const VARIABLES = VARIABLE_DATA.variables || [];
    const VARIABLE_MAX_TRACE = VARIABLE_DATA.max_trace || 0;
            const RAW_VARIABLE_GROUPS = VARIABLE_DATA.base_options || [];
                const VARIABLE_GROUPS = RAW_VARIABLE_GROUPS.length
                    ? RAW_VARIABLE_GROUPS
                            : (VARIABLES.map((item) => ({{
                                    base: item.label,
                                    has_indices: false,
                                    variables: [{{ id: item.id, label: item.label, index: [] }}],
                                }})));
            const VARIABLE_GROUP_MAP = Object.fromEntries(VARIABLE_GROUPS.map((group) => [group.base, group]));
            const VARIABLE_LOOKUP = Object.fromEntries(VARIABLES.map((item) => [item.id, item]));
    const INFERENCE = {inference_json};
    const DECISION_FLOW = {decision_flow};

        const renderedTabs = {{ modelFit: false, variables: false, hierarchy: false, validation: false, inference: false }};
        const variableState = {{ base: null, variableId: null }};
        const escapeBuffer = document.createElement('div');

        function escapeHtml(value) {{
            escapeBuffer.textContent = value == null ? '' : String(value);
            return escapeBuffer.innerHTML;
        }}

        function renderModelFit() {{
            if (renderedTabs.modelFit) return;
            Plotly.newPlot('plot-model-fit-contrib', PLOTS.contrib.data, PLOTS.contrib.layout, {{ responsive: true }});
            renderedTabs.modelFit = true;
        }}

        function renderVariables() {{
            if (renderedTabs.variables) return;
            const baseSelect = document.getElementById('variable-base-select');
            const indexSelect = document.getElementById('variable-index-select');
            const indexLabel = document.getElementById('variable-index-label');
            const summaryElement = document.getElementById('variable-summary');

            baseSelect.setAttribute('aria-label', 'Parameter group');
            baseSelect.setAttribute('aria-controls', 'variable-summary');
            indexSelect.setAttribute('aria-label', 'Parameter index');
            indexSelect.setAttribute('aria-controls', 'variable-summary');

            summaryElement.setAttribute('role', 'status');
            summaryElement.setAttribute('aria-live', 'polite');

            if (VARIABLES.length === 0) {{
                baseSelect.innerHTML = '<option>No posterior variables available</option>';
                baseSelect.disabled = true;
                indexSelect.style.display = 'none';
                indexLabel.style.display = 'none';
                summaryElement.textContent = 'Posterior diagnostics unavailable.';
                renderedTabs.variables = true;
                return;
            }}

            baseSelect.innerHTML = '';
            const multipleGroups = VARIABLE_GROUPS.length > 1;
            if (multipleGroups) {{
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = 'Choose parameter…';
                placeholder.disabled = true;
                baseSelect.appendChild(placeholder);
            }}

            VARIABLE_GROUPS.forEach((group) => {{
                const opt = document.createElement('option');
                opt.value = group.base;
                const label = group.has_indices ? group.base + ' (' + group.variables.length + ')' : group.base;
                opt.textContent = label;
                opt.title = 'Inspect ' + group.base;
                baseSelect.appendChild(opt);
            }});

            function toggleIndexVisibility(show) {{
                indexSelect.style.display = show ? '' : 'none';
                indexLabel.style.display = show ? '' : 'none';
                indexSelect.disabled = !show;
                indexLabel.setAttribute('aria-hidden', show ? 'false' : 'true');
            }}

            function selectVariableById(variableId) {{
                if (!variableId) {{
                    summaryElement.innerHTML = '<em>Select a parameter to inspect.</em>';
                    summaryElement.dataset.variableId = '';
                    variableState.variableId = null;
                    Plotly.purge('plot-variable-density');
                    Plotly.purge('plot-variable-trace');
                    return;
                }}

                const data = VARIABLE_LOOKUP[variableId];
                if (!data) {{
                    summaryElement.innerHTML = '<em>Parameter not found.</em>';
                    summaryElement.dataset.variableId = '';
                    variableState.variableId = null;
                    Plotly.purge('plot-variable-density');
                    Plotly.purge('plot-variable-trace');
                    return;
                }}

                variableState.variableId = variableId;
                variableState.base = baseSelect.value || null;
                summaryElement.dataset.variableId = variableId;
                summaryElement.dataset.base = variableState.base || '';

                const formattedLines = [
                    '<strong>Variable:</strong> ' + escapeHtml(data.label),
                    '<strong>Mean:</strong> ' + (Number.isFinite(data.mean) ? data.mean.toFixed(4) : 'NA') + ' &nbsp; <strong>SD:</strong> ' + (Number.isFinite(data.sd) ? data.sd.toFixed(4) : 'NA'),
                    '<strong>5th pct:</strong> ' + (Number.isFinite(data.q05) ? data.q05.toFixed(4) : 'NA') + ' &nbsp; <strong>Median:</strong> ' + (Number.isFinite(data.median) ? data.median.toFixed(4) : 'NA') + ' &nbsp; <strong>95th pct:</strong> ' + (Number.isFinite(data.q95) ? data.q95.toFixed(4) : 'NA')
                ];
                summaryElement.innerHTML = formattedLines.join('<br/>');

                renderVariableDetail(variableId);
            }}

            function updateIndexSelect(baseName, preferredId) {{
                const group = VARIABLE_GROUP_MAP[baseName];
                baseSelect.title = group ? 'Parameter group: ' + group.base : 'Parameter group';
                indexSelect.innerHTML = '';

                if (!group) {{
                    toggleIndexVisibility(false);
                    selectVariableById(null);
                    return;
                }}

                const variants = Array.isArray(group.variables) ? group.variables.slice() : [];
                variants.sort((a, b) => a.label.localeCompare(b.label, undefined, {{ numeric: true, sensitivity: 'base' }}));

                if (!group.has_indices || variants.length <= 1) {{
                    toggleIndexVisibility(false);
                    const first = variants[0];
                    selectVariableById(first ? first.id : null);
                    return;
                }}

                toggleIndexVisibility(true);
                variants.forEach((entry) => {{
                    const opt = document.createElement('option');
                    opt.value = entry.id;
                    opt.textContent = entry.label;
                    opt.title = 'Inspect ' + group.base + (entry.label.startsWith('[') ? ' ' + entry.label : ' [' + entry.label + ']');
                    indexSelect.appendChild(opt);
                }});

                let nextId = preferredId && variants.some((entry) => entry.id === preferredId) ? preferredId : null;
                if (!nextId && variants[0]) {{
                    nextId = variants[0].id;
                }}
                if (nextId) {{
                    indexSelect.value = nextId;
                }}
                selectVariableById(nextId);
            }}

            baseSelect.addEventListener('change', () => {{
                const targetBase = baseSelect.value;
                updateIndexSelect(targetBase, null);
            }});

            indexSelect.addEventListener('change', () => {{
                selectVariableById(indexSelect.value);
            }});

            const initialBase = variableState.base && VARIABLE_GROUP_MAP[variableState.base]
                ? variableState.base
                : (VARIABLE_GROUPS[0] ? VARIABLE_GROUPS[0].base : null);

            if (initialBase) {{
                baseSelect.value = initialBase;
                updateIndexSelect(initialBase, variableState.variableId);
            }} else if (VARIABLES[0]) {{
                selectVariableById(VARIABLES[0].id);
            }} else {{
                selectVariableById(null);
            }}

            renderedTabs.variables = true;
        }}

        function renderVariableDetail(variableId) {{
            const data = VARIABLE_LOOKUP[variableId];
            if (!data) {{
                Plotly.purge('plot-variable-density');
                Plotly.purge('plot-variable-trace');
                return;
            }}

            const posteriorSamples = Array.isArray(data.samples) ? data.samples : [];
            const priorCurve = data.prior && data.prior.curve;
            const hasPriorCurve = !!(
                priorCurve &&
                Array.isArray(priorCurve.x) &&
                Array.isArray(priorCurve.y) &&
                priorCurve.x.length === priorCurve.y.length
            );
            const ppcMeta = data.ppc || null;
            const hasPpcKde = !!(
                ppcMeta &&
                ppcMeta.kde &&
                Array.isArray(ppcMeta.kde.x) &&
                Array.isArray(ppcMeta.kde.y) &&
                ppcMeta.kde.x.length === ppcMeta.kde.y.length
            );
            const observedSamples = ppcMeta && ppcMeta.observed && Array.isArray(ppcMeta.observed.samples)
                ? ppcMeta.observed.samples
                : [];
            const observedHistnorm = ppcMeta && ppcMeta.observed && typeof ppcMeta.observed.histnorm === 'string'
                ? ppcMeta.observed.histnorm
                : 'probability density';
            const ppcRange = ppcMeta && Array.isArray(ppcMeta.x_range)
                ? ppcMeta.x_range
                : null;

            const densityTraces = [
                {{
                    x: posteriorSamples,
                    type: 'histogram',
                    histnorm: 'probability density',
                    marker: {{ color: '#2563eb' }},
                    opacity: 0.55,
                    name: 'Posterior',
                    hovertemplate: 'Posterior density<br>Value: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>',
                }},
            ];

            if (observedSamples.length) {{
                densityTraces.push({{
                    x: observedSamples,
                    type: 'histogram',
                    histnorm: observedHistnorm,
                    marker: {{ color: '#10b981' }},
                    opacity: 0.45,
                    name: 'Observed',
                    hovertemplate: 'Observed<br>Value: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>',
                }});
            }}

            if (hasPriorCurve) {{
                densityTraces.push({{
                    x: priorCurve.x,
                    y: priorCurve.y,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Prior',
                    line: {{ color: '#d9480f', width: 2, dash: 'dot' }},
                    hovertemplate: 'Prior density<br>Value: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>',
                }});
            }}

            if (hasPpcKde) {{
                densityTraces.push({{
                    x: ppcMeta.kde.x,
                    y: ppcMeta.kde.y,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Posterior predictive',
                    line: {{ color: '#7c3aed', width: 2 }},
                    hovertemplate: 'Posterior predictive<br>Value: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>',
                }});
            }}

            const densityLayout = {{
                template: 'plotly_white',
                title: data.label + ' posterior density',
                bargap: 0.05,
                barmode: 'overlay',
                xaxis: {{ title: 'Parameter value' }},
                yaxis: {{ title: 'Density' }},
                legend: {{ orientation: 'h', x: 0, y: 1.12, font: {{ size: 12 }} }},
                margin: {{ t: 60 }},
                hovermode: 'closest',
            }};

            const combinedX = posteriorSamples.slice();
            if (hasPriorCurve) {{
                combinedX.push(...priorCurve.x);
            }}
            if (observedSamples.length) {{
                combinedX.push(...observedSamples);
            }}
            if (hasPpcKde) {{
                combinedX.push(...ppcMeta.kde.x);
            }}
            const finiteX = combinedX.filter((val) => Number.isFinite(val));
            let xRange = null;
            if (ppcRange && ppcRange.length === 2 && ppcRange.every((val) => Number.isFinite(val))) {{
                xRange = ppcRange;
            }} else if (finiteX.length) {{
                const minX = Math.min(...finiteX);
                const maxX = Math.max(...finiteX);
                if (minX < maxX) {{
                    xRange = [minX, maxX];
                }}
            }}
            if (xRange) {{
                densityLayout.xaxis.range = xRange;
            }}

            Plotly.react('plot-variable-density', densityTraces, densityLayout, {{ responsive: true }});

            const chains = Array.isArray(data.chains) ? data.chains : [];
            const traceInfo = data.trace || null;
            const traceChains = traceInfo && Array.isArray(traceInfo.chains)
                ? traceInfo.chains
                : [];

            if (traceChains.length) {{
                const drawIndices = Array.isArray(traceInfo.draw_indices)
                    ? traceInfo.draw_indices
                    : null;

                const traceTraces = traceChains.map((series, idx) => {{
                    const values = Array.isArray(series && series.values) ? series.values : [];
                    const fallbackIndices = values.map((_, i) => i);
                    const seriesDraws = Array.isArray(series && series.draw_indices)
                        ? series.draw_indices
                        : null;
                    const xValues = drawIndices && drawIndices.length === values.length
                        ? drawIndices
                        : (seriesDraws && seriesDraws.length === values.length ? seriesDraws : fallbackIndices);
                    const chainIndex = typeof series.chain_index === 'number' ? series.chain_index : idx;
                    return {{
                        x: xValues,
                        y: values,
                        mode: 'lines',
                        name: 'Chain ' + (chainIndex + 1),
                        line: {{ width: 1 }},
                    }};
                }});

                const totalDraws = typeof traceInfo.total_draws === 'number'
                    ? traceInfo.total_draws
                    : (traceTraces[0] && Array.isArray(traceTraces[0].y) ? traceTraces[0].y.length : VARIABLE_MAX_TRACE);
                const hoverMode = traceInfo.hover && typeof traceInfo.hover.mode === 'string'
                    ? traceInfo.hover.mode
                    : 'x';

                Plotly.react('plot-variable-trace', traceTraces, {{
                    template: 'plotly_white',
                    title: data.label + ' trace (' + totalDraws + ' draws)',
                    xaxis: {{ title: 'Draw' }},
                    yaxis: {{ title: 'Value' }},
                    legend: {{ orientation: 'h', x: 0, y: 1.15, font: {{ size: 11 }} }},
                    hovermode: hoverMode,
                }}, {{ responsive: true }});
            }} else if (chains.length) {{
                const traceTraces = chains.map((samples, idx) => {{
                    const values = Array.isArray(samples) ? samples : [];
                    return {{
                        x: values.map((_, i) => i),
                        y: values,
                        mode: 'lines',
                        name: 'Chain ' + (idx + 1),
                        line: {{ width: 1 }},
                    }};
                }});

                Plotly.react('plot-variable-trace', traceTraces, {{
                    template: 'plotly_white',
                    title: data.label + ' trace (' + (VARIABLE_MAX_TRACE || traceTraces[0].y.length) + ' draws)',
                    xaxis: {{ title: 'Draw' }},
                    yaxis: {{ title: 'Value' }},
                    legend: {{ orientation: 'h', x: 0, y: 1.15, font: {{ size: 11 }} }},
                }}, {{ responsive: true }});
            }} else {{
                Plotly.purge('plot-variable-trace');
            }}
        }}

        function renderHierarchy() {{
      if (renderedTabs.hierarchy) return;
      Plotly.newPlot('plot-hierarchy-naics', PLOTS.naics_hierarchy.data, PLOTS.naics_hierarchy.layout, {{responsive: true}});
      Plotly.newPlot('plot-hierarchy-zip', PLOTS.zip_hierarchy.data, PLOTS.zip_hierarchy.layout, {{responsive: true}});
      Plotly.newPlot('plot-hierarchy-flow', PLOTS.sankey.data, PLOTS.sankey.layout, {{responsive: true}});
      const table = document.getElementById('hierarchy-flow-table');
      if (DECISION_FLOW.mean_contributions && DECISION_FLOW.mean_contributions.length) {{
        let rows = '<table class="flow-table"><thead><tr><th>Component</th><th>Mean</th><th>|Mean|</th></tr></thead><tbody>';
        DECISION_FLOW.mean_contributions.forEach(row => {{
          rows += `<tr><td>${{row.label}}</td><td>${{row.mean.toFixed(4)}}</td><td>${{Math.abs(row.abs).toFixed(4)}}`;
          rows += '</td></tr>';
        }});
        rows += '</tbody></table>';
        table.innerHTML = rows;
      }} else {{
        table.textContent = 'No hierarchy contributions available.';
      }}
      renderedTabs.hierarchy = true;
    }}

    function renderValidation() {{
      if (renderedTabs.validation) return;
      Plotly.newPlot('plot-validation-calibration', PLOTS.calibration.data, PLOTS.calibration.layout, {{responsive: true}});
      Plotly.newPlot('plot-validation-ranking', PLOTS.ranking.data, PLOTS.ranking.layout, {{responsive: true}});
      Plotly.newPlot('plot-validation-hist', PLOTS.hist.data, PLOTS.hist.layout, {{responsive: true}});
      Plotly.newPlot('plot-validation-scatter', PLOTS.scatter.data, PLOTS.scatter.layout, {{responsive: true}});
      renderedTabs.validation = true;
    }}

    function logistic(x) {{
      if (x > 0) {{
        const z = Math.exp(-x);
        return 1 / (1 + z);
      }}
      const z = Math.exp(x);
      return z / (1 + z);
    }}

    function padRight(value, length, fill) {{
      const raw = (value || '').trim();
      if (raw.length >= length) return raw.slice(0, length);
      return (raw + fill.repeat(length)).slice(0, length);
    }}

    function renderInference() {{
      if (renderedTabs.inference) return;
      document.getElementById('inference-score').addEventListener('click', () => {{
        const naics = (document.getElementById('inference-naics').value || '').trim();
        const zip = (document.getElementById('inference-zip').value || '').trim();
        const result = scoreSubmission(naics, zip);
        document.getElementById('inference-output').innerHTML = result;
      }});
      renderedTabs.inference = true;
    }}

    function scoreSubmission(naics, zip) {{
      const out = [];
      if (!naics) out.push('<em>NAICS missing</em>');
      if (!zip) out.push('<em>ZIP missing</em>');
      const mapsN = INFERENCE.naics_level_maps;
      const mapsZ = INFERENCE.zip_level_maps;
      const cutsN = INFERENCE.naics_cut_points;
      const cutsZ = INFERENCE.zip_cut_points;
      const vec = INFERENCE.effects;
      let eta = vec.beta0;
      let anyBackoff = false;
      const contributions = [];
      const paddedNaics = cutsN.length ? padRight(naics, cutsN[cutsN.length - 1], INFERENCE.prefix_fill) : '';
      cutsN.forEach((cut, idx) => {{
        const label = paddedNaics.slice(0, cut);
        const map = mapsN[idx] || {{}};
        const vector = idx === 0 ? vec.naics_base : (vec.naics_deltas[idx - 1] || []);
        const pos = map[label];
        let contrib = 0.0;
        if (pos !== undefined && vector[pos] !== undefined) {{
          contrib = Number(vector[pos]);
          eta += contrib;
        }} else {{
          anyBackoff = true;
        }}
        contributions.push({{ label: `NAICS L${{cut}}`, value: contrib }});
      }});
      const paddedZip = cutsZ.length ? padRight(zip, cutsZ[cutsZ.length - 1], INFERENCE.prefix_fill) : '';
      cutsZ.forEach((cut, idx) => {{
        const label = paddedZip.slice(0, cut);
        const map = mapsZ[idx] || {{}};
        const vector = idx === 0 ? vec.zip_base : (vec.zip_deltas[idx - 1] || []);
        const pos = map[label];
        let contrib = 0.0;
        if (pos !== undefined && vector[pos] !== undefined) {{
          contrib = Number(vector[pos]);
          eta += contrib;
        }} else {{
          anyBackoff = true;
        }}
        contributions.push({{ label: `ZIP L${{cut}}`, value: contrib }});
      }});
      const prob = logistic(eta);
      let htmlOut = `<strong>Logit η:</strong> ${{eta.toFixed(4)}}<br/><strong>Probability:</strong> ${{prob.toFixed(4)}}`;
      if (anyBackoff) {{
        htmlOut += '<br/><span style="color:#d9480f">Backoff applied for at least one level.</span>';
      }}
      htmlOut += '<hr/><strong>Contributions</strong><ul>';
      contributions.forEach(item => {{
        htmlOut += `<li>${{item.label}}: ${{item.value.toFixed(4)}}</li>`;
      }});
      htmlOut += '</ul>';
      return htmlOut;
    }}

    function activateTab(name) {{
      document.querySelectorAll('.tab-button').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.tab === name);
      }});
      document.querySelectorAll('.tab-content').forEach(panel => {{
        panel.classList.toggle('active', panel.id === `tab-${{name}}`);
      }});
      if (name === 'model-fit') renderModelFit();
      else if (name === 'variables') renderVariables();
      else if (name === 'hierarchy') renderHierarchy();
      else if (name === 'validation') renderValidation();
      else if (name === 'inference') renderInference();
    }}

    document.querySelectorAll('.tab-button').forEach(btn => {{
      btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    }});

    renderModelFit();
  </script>
</body>
</html>
"""


def build_dashboard(
    *,
    train_summary: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ranking: Mapping[str, Any],
    loo: Any,
    scored_test: pl.DataFrame,
    parameter_alignment: Mapping[str, float] | None = None,
    output_dir: Path | str | None = None,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Mapping[str, Any],
    idata: Any | None = None,
    prefix_fill: str = "0",
) -> DashboardPayload:
    reliability_df = (
        calibration["reliability"]
        if isinstance(calibration["reliability"], pl.DataFrame)
        else pl.DataFrame(calibration["reliability"])
    )
    ranking_df = (
        ranking["summary"]
        if isinstance(ranking["summary"], pl.DataFrame)
        else pl.DataFrame(ranking["summary"])
    )

    scored = scored_test.clone()
    loo_stats = _extract_loo_stats(loo)

    reliability_data = _prepare_reliability_plot(reliability_df)
    ranking_data = _prepare_ranking_plot(ranking_df)

    component_cols = [
        col
        for col in scored.columns
        if col.endswith("_effect")
        and (col.startswith("naics_L") or col.startswith("zip_L"))
    ]
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

    def effect_label(raw: str) -> str:
        family, level, _ = raw.split("_")
        return f"{family.upper()} {level.upper()}"

    for col in sorted(component_cols, key=lambda name: int(name.split("_")[1][1:])):
        mean_val = float(scored.select(pl.col(col).mean()).item())
        abs_val = float(scored.select(pl.col(col).abs().mean()).item())
        if not np.isfinite(mean_val) and not np.isfinite(abs_val):
            continue
        component_display_order.append((effect_label(col), mean_val, abs_val))

    fig_sankey, decision_flow = _build_decision_flow(component_display_order)

    fig_calibration = go.Figure(
        data=[
            go.Scatter(
                x=reliability_data["mid"],
                y=reliability_data["mean_y"],
                mode="lines+markers",
                name="Observed",
            ),
            go.Scatter(
                x=reliability_data["mid"],
                y=reliability_data["mean_p"],
                mode="lines+markers",
                name="Predicted",
            ),
        ]
    )
    fig_calibration.update_layout(
        template="plotly_white", xaxis_title="Probability bin", yaxis_title="Mean"
    )

    fig_ranking = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig_ranking.add_trace(
        go.Scatter(
            x=ranking_data["k_pct"],
            y=ranking_data["lift"],
            mode="lines+markers",
            name="Lift",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig_ranking.add_trace(
        go.Scatter(
            x=ranking_data["k_pct"],
            y=ranking_data["cum_gain"],
            mode="lines+markers",
            name="Cumulative gain",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig_ranking.update_layout(template="plotly_white")
    fig_ranking.update_xaxes(title_text="Top k%", row=1, col=1)
    fig_ranking.update_yaxes(title_text="Lift", row=1, col=1, secondary_y=False)
    fig_ranking.update_yaxes(
        title_text="Cumulative gain", row=1, col=1, secondary_y=True
    )

    fig_contrib = go.Figure(
        data=[
            go.Bar(
                x=[label for label, _, _ in component_display_order]
                or ["No components"],
                y=[mean for _, mean, _ in component_display_order] or [0.0],
                name="Mean contribution",
            )
        ]
    )
    fig_contrib.update_layout(
        template="plotly_white",
        xaxis_title="Component",
        yaxis_title="Mean contribution",
    )

    test_probs = scored["p"].cast(pl.Float64).to_list()
    true_probs = (
        scored["p_true"].cast(pl.Float64).to_list()
        if "p_true" in scored.columns
        else [float("nan")] * len(test_probs)
    )

    calibration_summary = {
        "bin_count": reliability_df.height,
        "mean_gap": float(reliability_df["gap"].abs().mean())
        if "gap" in reliability_df.columns
        else 0.0,
        "max_gap": float(reliability_df["gap"].abs().max())
        if "gap" in reliability_df.columns and reliability_df.height > 0
        else 0.0,
    }

    ranking_ordered = ranking_df.sort("k_pct") if ranking_df.height else ranking_df
    ranking_meta = {
        "k_values": [int(val) for val in ranking_ordered["k_pct"].to_list()],
        "best_k": int(ranking_ordered.sort("lift", descending=True)["k_pct"][0])
        if "lift" in ranking_ordered.columns and ranking_ordered.height > 0
        else None,
        "best_lift": float(ranking_ordered["lift"].max())
        if "lift" in ranking_ordered.columns and ranking_ordered.height > 0
        else float("nan"),
    }

    prob_array = np.asarray(test_probs, dtype=float)
    distribution_meta = {
        "count": int(prob_array.size),
        "p_mean": float(prob_array.mean()) if prob_array.size else float("nan"),
        "p_std": float(prob_array.std(ddof=1)) if prob_array.size > 1 else 0.0,
        "y_mean": float(scored["y_true"].mean())
        if "y_true" in scored.columns
        else float("nan"),
    }

    fig_hist = go.Figure(
        data=[go.Histogram(x=test_probs, nbinsx=20, name="Prediction density")]
    )
    fig_hist.update_layout(
        template="plotly_white", xaxis_title="p_hat", yaxis_title="Count"
    )

    fig_scatter = go.Figure(
        data=[
            go.Scatter(
                x=true_probs,
                y=test_probs,
                mode="markers",
                name="Pred vs true",
                marker=dict(size=6, opacity=0.6),
            ),
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Ideal",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ),
        ]
    )
    fig_scatter.update_layout(
        template="plotly_white",
        xaxis_title="True probability",
        yaxis_title="Predicted probability",
    )

    fig_naics = _build_hierarchy_sunburst(
        prefix="NAICS",
        cut_points=naics_cut_points,
        level_maps=naics_level_maps,
        effect_vectors=[effects["naics_base"], *effects.get("naics_deltas", [])],
    )
    fig_zip = _build_hierarchy_sunburst(
        prefix="ZIP",
        cut_points=zip_cut_points,
        level_maps=zip_level_maps,
        effect_vectors=[effects["zip_base"], *effects.get("zip_deltas", [])],
    )

    naics_summary = _summarize_hierarchy_levels(
        cut_points=naics_cut_points,
        level_maps=naics_level_maps,
        base=effects["naics_base"],
        deltas=effects.get("naics_deltas", []),
    )
    zip_summary = _summarize_hierarchy_levels(
        cut_points=zip_cut_points,
        level_maps=zip_level_maps,
        base=effects["zip_base"],
        deltas=effects.get("zip_deltas", []),
    )

    plots = {
        "calibration": json.loads(pio.to_json(fig_calibration, pretty=False)),
        "ranking": json.loads(pio.to_json(fig_ranking, pretty=False)),
        "contrib": json.loads(pio.to_json(fig_contrib, pretty=False)),
        "hist": json.loads(pio.to_json(fig_hist, pretty=False)),
        "scatter": json.loads(pio.to_json(fig_scatter, pretty=False)),
        "sankey": json.loads(pio.to_json(fig_sankey, pretty=False)),
        "naics_hierarchy": json.loads(pio.to_json(fig_naics, pretty=False)),
        "zip_hierarchy": json.loads(pio.to_json(fig_zip, pretty=False)),
    }

    variable_payload = _build_variable_payload(idata)
    inference_bundle = _build_inference_bundle(
        effects=effects,
        naics_cut_points=naics_cut_points,
        zip_cut_points=zip_cut_points,
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        prefix_fill=prefix_fill,
        scored=scored,
    )

    fit_stats: Dict[str, Any] = dict(train_summary)
    fit_stats.update(loo_stats)
    fit_stats.update(parameter_alignment or {})

    ece = float(calibration["ece"])
    brier = float(calibration["brier"])
    log_loss = float(calibration["log_loss"])

    validation_stats = {
        "loo": fit_stats.get("loo", float("nan")),
        "loo_se": fit_stats.get("loo_se", float("nan")),
        "pareto_k_max": fit_stats.get("pareto_k_max", float("nan")),
        "ece": ece,
        "brier": brier,
        "log_loss": log_loss,
    }

    backoff_rate = (
        float(scored["any_backoff"].cast(pl.Float64).mean())
        if "any_backoff" in scored.columns
        else 0.0
    )
    test_metrics = {
        "brier": brier,
        "ece": ece,
        "log_loss": log_loss,
        "backoff_rate": backoff_rate,
        "avg_pred": float(scored["p"].mean()),
    }

    model_cards_html = _model_summary_cards(fit_stats, validation_stats, test_metrics)

    html_doc = _render_dashboard_html(
        plots_json=json.dumps(plots),
        variable_json=json.dumps(variable_payload),
        inference_json=json.dumps(inference_bundle),
        decision_flow=json.dumps(decision_flow),
        model_cards_html=model_cards_html,
    )

    tabs = [
        {
            "id": "model-fit",
            "label": "Model Fit",
            "description": "High-level diagnostics, summary cards, and contribution bar chart.",
            "default": True,
            "cards": True,
            "plots": ["plot-model-fit-contrib"],
        },
        {
            "id": "variables",
            "label": "Variables",
            "description": "Posterior density and trace diagnostics for selected parameters.",
            "default": False,
            "cards": False,
            "plots": ["plot-variable-density", "plot-variable-trace"],
        },
        {
            "id": "hierarchy",
            "label": "Hierarchy",
            "description": "Hierarchy sunbursts and contribution flow to inspect NAICS/ZIP structure.",
            "default": False,
            "cards": False,
            "plots": [
                "plot-hierarchy-naics",
                "plot-hierarchy-zip",
                "plot-hierarchy-flow",
            ],
        },
        {
            "id": "validation",
            "label": "Validation",
            "description": "Calibration, ranking lift, and predictive diagnostics charts.",
            "default": False,
            "cards": False,
            "plots": [
                "plot-validation-calibration",
                "plot-validation-ranking",
                "plot-validation-hist",
                "plot-validation-scatter",
            ],
        },
        {
            "id": "inference",
            "label": "Inference",
            "description": "Interactive scoring form with posterior contributions for new submissions.",
            "default": False,
            "cards": False,
            "plots": ["inference-output"],
        },
    ]

    html_path: Path | None = None
    if output_dir is not None:
        html_path = Path(output_dir) / "model_dashboard.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_doc, encoding="utf-8")

    payload: DashboardPayload = {
        "html": html_doc,
        "fit_stats": fit_stats,
        "validation": validation_stats,
        "test_metrics": test_metrics,
        "decision_flow": decision_flow,
        "inference": inference_bundle,
        "variables": variable_payload,
        "hierarchy": {"naics": naics_summary, "zip": zip_summary},
        "validation_detail": {
            "calibration": calibration_summary,
            "ranking": ranking_meta,
            "distribution": distribution_meta,
        },
        "tabs": tabs,
    }
    if html_path is not None:
        payload["artifacts"] = {"html_path": html_path}
    return payload
