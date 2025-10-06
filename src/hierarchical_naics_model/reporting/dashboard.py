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
    max_trace: int = 200,
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    if idata is None or not hasattr(idata, "groups"):
        return {"variables": variables, "max_trace": max_trace}
    if "posterior" not in idata.groups():
        return {"variables": variables, "max_trace": max_trace}

    posterior = idata.posterior
    try:
        rhat_ds = az.rhat(idata, method="rank")
    except Exception:  # noqa: BLE001
        rhat_ds = None
    try:
        ess_bulk_ds = az.ess(idata, method="bulk")
    except Exception:  # noqa: BLE001
        ess_bulk_ds = None

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

    for name, da in posterior.data_vars.items():
        if da.ndim < 2:  # expect chain/draw at least
            continue
        arr = np.asarray(da.values)
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        entry_shape = arr.shape[2:] if arr.ndim > 2 else (1,)
        entry_count = int(np.prod(entry_shape))
        chain_dim = arr.shape[0]
        for entry_idx in range(entry_count):
            if len(variables) >= max_variables:
                break
            idx_tuple = (
                np.unravel_index(entry_idx, entry_shape) if entry_shape != (1,) else ()
            )
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
            chain_samples = [
                entry_arr[c, :max_trace].astype(float).tolist()
                for c in range(min(chain_dim, 4))
            ]
            index_label = (
                "" if not idx_tuple else "[" + ",".join(str(i) for i in idx_tuple) + "]"
            )
            label = f"{name}{index_label}"
            variables.append(
                {
                    "id": label,
                    "label": label,
                    "mean": mean_val,
                    "sd": sd_val,
                    "q05": float(q05),
                    "median": float(q50),
                    "q95": float(q95),
                    "samples": sample_list,
                    "chains": chain_samples,
                    "prior": {"mean": 0.0, "sd": 1.0},
                    "r_hat": _diag_value(rhat_ds, name, idx_tuple),
                    "ess_bulk": _diag_value(ess_bulk_ds, name, idx_tuple),
                }
            )
        if len(variables) >= max_variables:
            break
    return {"variables": variables, "max_trace": max_trace}


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
          <label for=\"variable-select\">Select variable:</label>
          <select id=\"variable-select\"></select>
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
    const INFERENCE = {inference_json};
    const DECISION_FLOW = {decision_flow};

    const renderedTabs = {{ modelFit: false, variables: false, hierarchy: false, validation: false, inference: false }};

    function renderModelFit() {{
      if (renderedTabs.modelFit) return;
      Plotly.newPlot('plot-model-fit-contrib', PLOTS.contrib.data, PLOTS.contrib.layout, {{responsive: true}});
      renderedTabs.modelFit = true;
    }}

    function renderVariables() {{
      if (renderedTabs.variables) return;
      const select = document.getElementById('variable-select');
      if (VARIABLES.length === 0) {{
        select.innerHTML = '<option>No posterior variables available</option>';
        document.getElementById('variable-summary').textContent = 'Posterior diagnostics unavailable.';
        renderedTabs.variables = true;
        return;
      }}
      VARIABLES.forEach((item, idx) => {{
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = item.label;
        select.appendChild(opt);
      }});
      select.addEventListener('change', () => renderVariableDetail(Number(select.value)));
      renderVariableDetail(0);
      renderedTabs.variables = true;
    }}

    function renderVariableDetail(index) {{
      const data = VARIABLES[index];
      if (!data) return;
      const summary = `
        <strong>{html.escape("Variable")}: </strong>${{data.label}}<br/>
        <strong>Mean:</strong> ${{data.mean.toFixed(4)}} &nbsp; <strong>SD:</strong> ${{data.sd.toFixed(4)}}<br/>
        <strong>5th pct:</strong> ${{data.q05.toFixed(4)}} &nbsp; <strong>Median:</strong> ${{data.median.toFixed(4)}} &nbsp; <strong>95th pct:</strong> ${{data.q95.toFixed(4)}}
      `;
      document.getElementById('variable-summary').innerHTML = summary;
      if (data.samples && data.samples.length) {{
        Plotly.newPlot('plot-variable-density', [
          {{ x: data.samples, type: 'histogram', marker: {{color: '#2563eb', opacity: 0.7}}, name: 'Posterior' }}
        ], {{
          template: 'plotly_white',
          title: `${{data.label}} posterior density`,
          bargap: 0.05
        }}, {{responsive: true}});
      }} else {{
        Plotly.purge('plot-variable-density');
      }}
      if (data.chains && data.chains.length) {{
        const traces = data.chains.map((samples, idx) => ({{
          y: samples,
          x: samples.map((_, i) => i),
          mode: 'lines',
          name: `Chain ${{idx + 1}}`,
          line: {{ width: 1 }}
        }}));
        Plotly.newPlot('plot-variable-trace', traces, {{
          template: 'plotly_white',
          title: `${{data.label}} trace (first ${{VARIABLE_MAX_TRACE}} draws)`
        }}, {{responsive: true}});
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
