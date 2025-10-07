from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import polars as pl
import pytest
import xarray as xr

import arviz as az

from hierarchical_naics_model.core.hierarchy import build_hierarchical_indices
from hierarchical_naics_model.eval.calibration import calibration_report
from hierarchical_naics_model.eval.ranking import ranking_report
from hierarchical_naics_model.modeling.pymc_nested import PymcNestedDeltaStrategy
from hierarchical_naics_model.scoring.extract import extract_effect_tables_nested
from hierarchical_naics_model.scoring.predict import predict_proba_nested
from hierarchical_naics_model.synthgen.generate import (
    HierSpec,
    generate_synthetic_dataset,
)
from hierarchical_naics_model.reporting import dashboard as dashboard_module

TRACE_MAX_POINTS = getattr(dashboard_module, "TRACE_MAX_POINTS", 400)
build_dashboard = dashboard_module.build_dashboard


def _shuffle_and_split(
    df: pl.DataFrame, n_train: int, *, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    shuffled = df.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=seed)
    train = shuffled.head(n_train)
    test = shuffled.slice(n_train, shuffled.height - n_train)
    return train, test


def _effect_alignment_corr(
    *,
    estimated: np.ndarray,
    estimated_map: Mapping[str, int],
    truth: list[float],
    truth_map: Mapping[str, int],
) -> float:
    common_labels = sorted(set(estimated_map) & set(truth_map))
    if len(common_labels) < 2:
        raise RuntimeError("Not enough overlapping labels to compute correlation.")
    est_vals = np.array(
        [estimated[int(estimated_map[label])] for label in common_labels], dtype=float
    )
    true_vals = np.array(
        [truth[int(truth_map[label])] for label in common_labels], dtype=float
    )
    corr = np.corrcoef(est_vals, true_vals)[0, 1]
    return float(corr)


def test_end_to_end_pipeline_recovers_parameters(tmp_path):
    naics_spec = HierSpec(cut_points=[2, 3, 4, 5, 6], branching=[3, 3, 2, 2])
    zip_spec = HierSpec(cut_points=[1, 2, 3, 4, 5], branching=[3, 3, 2, 2])

    df, synth_artifacts = generate_synthetic_dataset(
        n=320,
        naics_spec=naics_spec,
        zip_spec=zip_spec,
        seed=902,
    )

    df = df.rename({"naics_code": "NAICS", "zip_code": "ZIP", "y": "is_written"})
    df = df.with_columns(pl.col("is_written").cast(pl.Int8))
    train_df, test_df = _shuffle_and_split(df, 240, seed=77)

    naics_cuts = naics_spec.cut_points
    zip_cuts = zip_spec.cut_points

    naics_train_idx = build_hierarchical_indices(
        train_df["NAICS"].to_list(), cut_points=naics_cuts, prefix_fill=naics_spec.fill
    )
    zip_train_idx = build_hierarchical_indices(
        train_df["ZIP"].to_list(), cut_points=zip_cuts, prefix_fill=zip_spec.fill
    )

    y_train = train_df["is_written"].to_numpy().astype(int)

    strategy = PymcNestedDeltaStrategy(default_target_accept=0.9)
    model = strategy.build_model(
        y=y_train,
        naics_levels=naics_train_idx["code_levels"],
        zip_levels=zip_train_idx["code_levels"],
        naics_group_counts=naics_train_idx["group_counts"],
        zip_group_counts=zip_train_idx["group_counts"],
    )

    idata = strategy.sample_posterior(
        model,
        draws=400,
        tune=400,
        chains=2,
        cores=1,
        progressbar=False,
        random_seed=321,
        idata_kwargs={"log_likelihood": True},
    )

    effects = extract_effect_tables_nested(idata)
    # Pad deltas in case variational sampling omitted a sparse level.
    naics_deltas = list(cast(Sequence[np.ndarray], effects["naics_deltas"]))
    desired_naics_levels = len(naics_cuts) - 1
    naics_deltas = naics_deltas[:desired_naics_levels]
    while len(naics_deltas) < desired_naics_levels:
        level_pos = len(naics_deltas) + 1
        size = naics_train_idx["group_counts"][level_pos]
        naics_deltas.append(np.zeros(size, dtype=float))
    effects["naics_deltas"] = naics_deltas

    zip_deltas = list(cast(Sequence[np.ndarray], effects["zip_deltas"]))
    desired_zip_levels = len(zip_cuts) - 1
    zip_deltas = zip_deltas[:desired_zip_levels]
    while len(zip_deltas) < desired_zip_levels:
        level_pos = len(zip_deltas) + 1
        size = zip_train_idx["group_counts"][level_pos]
        zip_deltas.append(np.zeros(size, dtype=float))
    effects["zip_deltas"] = zip_deltas

    scorable_test = test_df.drop(
        [col for col in ["eta", "p"] if col in test_df.columns]
    )

    scored_test = predict_proba_nested(
        scorable_test,
        naics_col="NAICS",
        zip_col="ZIP",
        naics_cut_points=naics_cuts,
        zip_cut_points=zip_cuts,
        naics_level_maps=naics_train_idx["maps"],
        zip_level_maps=zip_train_idx["maps"],
        effects=effects,
        prefix_fill=naics_spec.fill,
        return_components=True,
    )

    scored_test = scored_test.with_columns(
        pl.Series("eta_true", test_df["eta"]),
        pl.Series("p_true", test_df["p"]),
        pl.Series("y_true", test_df["is_written"]),
    )

    calibration = calibration_report(scored_test["y_true"], scored_test["p"], bins=6)
    ranking = ranking_report(scored_test["y_true"], scored_test["p"], ks=[10, 20, 40])
    loo_report = az.loo(idata, pointwise=False)

    eta_gap = float((scored_test["eta"] - scored_test["eta_true"]).abs().mean())
    assert eta_gap < 0.75
    brier = float(calibration["brier"])
    assert brier < 0.27

    lift_summary = ranking["summary"].filter(pl.col("k_pct") == 10)
    assert lift_summary.height == 1
    assert float(lift_summary["lift"][0]) > 0.6

    # ensure effect contributions per hierarchy level exist
    for cut in naics_cuts:
        col = f"naics_L{cut}_effect"
        assert col in scored_test.columns
    for cut in zip_cuts:
        col = f"zip_L{cut}_effect"
        assert col in scored_test.columns

    naics_truth_map = {
        str(label): int(idx)
        for label, idx in synth_artifacts["naics_maps"]["maps"][0].items()
    }
    naics_corr = _effect_alignment_corr(
        estimated=np.asarray(effects["naics_base"], dtype=float),
        estimated_map=naics_train_idx["maps"][0],
        truth=synth_artifacts["effects"]["naics_base"],
        truth_map=naics_truth_map,
    )
    assert naics_corr > 0.6

    train_pos_rate = float(train_df["is_written"].mean())
    test_pos_rate = float(test_df["is_written"].mean())

    dashboard = build_dashboard(
        train_summary={
            "n_train": train_df.height,
            "train_positive_rate": train_pos_rate,
            "n_test": test_df.height,
            "test_positive_rate": test_pos_rate,
        },
        calibration=calibration,
        ranking=ranking,
        loo=loo_report,
        scored_test=scored_test,
        parameter_alignment={"naics_base_corr": naics_corr},
        output_dir=tmp_path,
        naics_cut_points=naics_cuts,
        zip_cut_points=zip_cuts,
        naics_level_maps=naics_train_idx["maps"],
        zip_level_maps=zip_train_idx["maps"],
        effects=effects,
        idata=idata,
        prefix_fill=naics_spec.fill,
    )

    html_doc = dashboard["html"]
    assert isinstance(html_doc, str)
    assert 'data-tab="model-fit"' in html_doc
    assert 'data-tab="variables"' in html_doc
    assert 'data-tab="hierarchy"' in html_doc
    assert 'data-tab="validation"' in html_doc
    assert 'data-tab="inference"' in html_doc

    assert dashboard["fit_stats"]["n_train"] == train_df.height
    loo_value = getattr(loo_report, "loo", None)
    if loo_value is None:
        loo_value = getattr(loo_report, "elpd_loo")
    assert dashboard["validation"]["loo"] == pytest.approx(float(loo_value))
    assert dashboard["test_metrics"]["brier"] == pytest.approx(
        float(calibration["brier"])
    )
    assert dashboard["test_metrics"]["ece"] == pytest.approx(float(calibration["ece"]))

    decision_flow = dashboard.get("decision_flow")
    assert decision_flow is not None
    assert "node_labels" in decision_flow
    assert any(label.startswith("NAICS L") for label in decision_flow["node_labels"])
    assert any(label.startswith("ZIP L") for label in decision_flow["node_labels"])

    inference_bundle = dashboard.get("inference")
    assert inference_bundle is not None
    assert "effects" in inference_bundle
    assert "naics_cut_points" in inference_bundle
    assert "zip_cut_points" in inference_bundle


def test_dashboard_model_fit_tab_cards():
    """Dashboard should describe the model fit tab structure and cards."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0, 0.2],
            "bin_high": [0.2, 0.4],
            "mean_p": [0.15, 0.25],
            "mean_y": [0.18, 0.28],
            "gap": [0.03, 0.03],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10, 20],
            "lift": [1.25, 1.10],
            "cum_gain": [0.45, 0.52],
        }
    )

    class DummyLoo:
        loo = 123.4
        loo_se = 4.2
        pareto_k = np.array([0.1, 0.2])

    scored_test = pl.DataFrame(
        {
            "eta": [0.3, 0.6],
            "p": [0.574, 0.646],
            "p_true": [0.50, 0.70],
            "naics_L2_effect": [0.05, 0.06],
            "zip_L2_effect": [0.02, 0.01],
            "any_backoff": [False, True],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.1,
        "naics_base": np.array([0.05], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.02], dtype=float),
        "zip_deltas": [],
    }

    dashboard = build_dashboard(
        train_summary={
            "n_train": 200,
            "train_positive_rate": 0.18,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.03,
            "brier": 0.21,
            "log_loss": 0.58,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=None,
        prefix_fill="0",
    )

    tabs = {tab["id"]: tab for tab in dashboard.get("tabs", [])}
    assert set(tabs) == {
        "model-fit",
        "variables",
        "hierarchy",
        "validation",
        "inference",
    }

    model_tab = tabs["model-fit"]
    assert model_tab["label"] == "Model Fit"
    assert "cards" in model_tab and model_tab["cards"] is True
    assert "plots" in model_tab
    assert "plot-model-fit-contrib" in model_tab["plots"]

    html_doc = dashboard["html"]
    assert 'data-tab="model-fit"' in html_doc
    assert 'class="card-grid"' in html_doc


def test_dashboard_variable_tab_payload():
    """Posterior diagnostics should include convergence metrics for variables."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.2],
            "mean_y": [0.22],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.2],
            "cum_gain": [0.4],
        }
    )

    class DummyLoo:
        loo = 98.7
        loo_se = 3.1
        pareto_k = np.array([0.15])

    scored_test = pl.DataFrame(
        {
            "eta": [0.2, 0.5],
            "p": [0.55, 0.62],
            "p_true": [0.5, 0.65],
            "naics_L2_effect": [0.03, 0.04],
            "zip_L2_effect": [0.01, 0.02],
            "any_backoff": [False, False],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.05,
        "naics_base": np.array([0.03], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.05, 0.06, 0.07, 0.05],
                        [0.049, 0.061, 0.068, 0.052],
                    ]
                ),
            ),
            "naics_base": (
                ("chain", "draw", "idx"),
                np.array(
                    [
                        [[0.03], [0.031], [0.032], [0.030]],
                        [[0.029], [0.030], [0.031], [0.0295]],
                    ]
                ),
            ),
        }
    )
    idata = az.InferenceData(posterior=posterior)

    dashboard = build_dashboard(
        train_summary={
            "n_train": 150,
            "train_positive_rate": 0.2,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.05,
            "brier": 0.24,
            "log_loss": 0.6,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    assert "variables" in variable_payload
    expected_draws = int(posterior.sizes["draw"])
    assert variable_payload["max_trace"] == expected_draws
    assert len(variable_payload["variables"]) >= 2

    base_options = variable_payload.get("base_options")
    assert base_options is not None, (
        "Expected grouped base options for variable dropdown"
    )
    assert len(base_options) >= 1

    bases_by_name = {item["base"]: item for item in base_options}
    assert "beta0" in bases_by_name
    beta0_group = bases_by_name["beta0"]
    assert beta0_group["has_indices"] is False
    assert beta0_group["variables"], (
        "beta0 group should include at least one variable entry"
    )

    indexed_group = next(
        (item for item in base_options if item["has_indices"] and item["variables"]),
        None,
    )
    assert indexed_group is not None, (
        "Expected at least one variable group with indices"
    )
    for var_entry in indexed_group["variables"]:
        assert "[" in var_entry["label"], (
            "Indexed variables should retain bracketed labels"
        )

    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )
    assert beta0_entry["label"].startswith("beta0")
    assert beta0_entry["r_hat"] == pytest.approx(1.0, abs=0.25)
    assert beta0_entry["ess_bulk"] > 0
    assert beta0_entry["samples"]
    trace_info = beta0_entry.get("trace")
    assert trace_info is not None
    assert trace_info["total_draws"] == expected_draws
    draw_indices = trace_info["draw_indices"]
    assert draw_indices[0] == 0
    assert draw_indices[-1] == expected_draws - 1
    if expected_draws > TRACE_MAX_POINTS:
        assert len(draw_indices) == TRACE_MAX_POINTS
    else:
        assert draw_indices == list(range(expected_draws))
    assert trace_info.get("hover", {}).get("mode") == "x unified"
    prior_curve = beta0_entry["prior"].get("curve")
    assert prior_curve is not None, (
        "Expected prior density curve for posterior histogram overlay"
    )
    assert len(prior_curve.get("x", [])) >= 20
    assert len(prior_curve.get("y", [])) == len(prior_curve["x"])
    assert all(isinstance(val, float) for val in prior_curve["x"][:5])
    assert all(isinstance(val, float) for val in prior_curve["y"][:5])


def test_dashboard_variable_trace_downsamples_and_smooths():
    """Trace metadata should downsample draws while exposing smoothing support."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.22],
            "mean_y": [0.24],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.1],
            "cum_gain": [0.38],
        }
    )

    class DummyLoo:
        loo = 105.2
        loo_se = 4.5
        pareto_k = np.array([0.2])

    scored_test = pl.DataFrame(
        {
            "eta": [0.25, 0.31],
            "p": [0.56, 0.58],
            "p_true": [0.5, 0.62],
            "naics_L2_effect": [0.03, 0.04],
            "zip_L2_effect": [0.01, 0.015],
            "any_backoff": [False, False],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.04,
        "naics_base": np.array([0.03], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    draws = TRACE_MAX_POINTS + 150
    chains = 2
    beta0_values = np.linspace(0.04, 0.06, num=draws, dtype=float)
    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.vstack([beta0_values, beta0_values + 0.001]),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior)

    dashboard = build_dashboard(
        train_summary={
            "n_train": 120,
            "train_positive_rate": 0.21,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.04,
            "brier": 0.23,
            "log_loss": 0.59,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )

    trace_info = beta0_entry.get("trace")
    assert trace_info is not None, (
        "Trace metadata must be present for posterior variables"
    )

    assert trace_info.get("total_draws") == draws
    assert trace_info.get("total_chains") == chains

    draw_indices = trace_info.get("draw_indices")
    assert len(draw_indices) == TRACE_MAX_POINTS
    assert draw_indices[0] == 0
    assert draw_indices[-1] == draws - 1

    chain_series = trace_info.get("chains") or []
    assert len(chain_series) == chains
    for idx, series in enumerate(chain_series):
        values = series.get("values") or []
        series_indices = series.get("draw_indices") or draw_indices
        assert len(values) == TRACE_MAX_POINTS
        assert series_indices[0] == 0
        assert series_indices[-1] == draws - 1
        assert series.get("chain_index") == idx

    smoothed = trace_info.get("smoothed") or []
    assert len(smoothed) == chains
    for smooth_series in smoothed:
        smooth_values = smooth_series.get("values") or []
        smooth_indices = smooth_series.get("draw_indices") or []
        assert len(smooth_values) == TRACE_MAX_POINTS
        assert smooth_indices[0] == 0
        assert smooth_indices[-1] == draws - 1
        assert all(np.isfinite(val) for val in smooth_values)


def test_dashboard_variable_trace_with_prior_and_indices():
    """Trace payload should include prior-informed stats and multi-index entries."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.2],
            "mean_y": [0.22],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.15],
            "cum_gain": [0.41],
        }
    )

    class DummyLoo:
        loo = 84.2
        loo_se = 3.8
        pareto_k = np.array([0.18])

    scored_test = pl.DataFrame(
        {
            "eta": [0.2, 0.3],
            "p": [0.54, 0.57],
            "p_true": [0.5, 0.6],
            "naics_L2_effect": [0.02, 0.03],
            "zip_L2_effect": [0.01, 0.012],
            "any_backoff": [False, True],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.03,
        "naics_base": np.array([0.02], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    chains = 2
    draws = 12
    posterior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "idx"),
                np.stack(
                    [
                        np.column_stack(
                            [
                                np.linspace(0.1, 0.2, num=draws),
                                np.linspace(0.3, 0.45, num=draws),
                            ]
                        ),
                        np.column_stack(
                            [
                                np.linspace(0.11, 0.21, num=draws),
                                np.linspace(0.31, 0.46, num=draws),
                            ]
                        ),
                    ]
                ),
            )
        }
    )
    prior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "idx"),
                np.stack(
                    [
                        np.column_stack(
                            [
                                np.full(draws, 0.12, dtype=float),
                                np.full(draws, 0.4, dtype=float),
                            ]
                        ),
                        np.column_stack(
                            [
                                np.full(draws, 0.13, dtype=float),
                                np.full(draws, 0.41, dtype=float),
                            ]
                        ),
                    ]
                ),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior, prior=prior)

    dashboard = build_dashboard(
        train_summary={
            "n_train": 110,
            "train_positive_rate": 0.19,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.03,
            "brier": 0.22,
            "log_loss": 0.58,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    weights_entries = [
        item
        for item in variable_payload["variables"]
        if item["id"].startswith("weights")
    ]
    assert weights_entries, "Expected multi-index weight variables"

    # Ensure indices preserved and trace metadata includes all chains
    sample_entry = next(item for item in weights_entries if "[0]" in item["label"])
    trace = sample_entry["trace"]
    assert trace["total_chains"] == chains
    assert trace["total_draws"] == draws
    assert trace["draw_indices"] == list(range(draws))
    assert all(len(series["values"]) == draws for series in trace["chains"])
    assert all(
        (series.get("draw_indices") or list(range(draws))) == list(range(draws))
        for series in trace["chains"]
    )

    smoothed = trace.get("smoothed") or []
    assert len(smoothed) == chains
    assert all(len(series.get("values") or []) == draws for series in smoothed)
    assert all(
        (series.get("draw_indices") or list(range(draws))) == list(range(draws))
        for series in smoothed
    )

    # Prior mean/curve should leverage prior group samples
    prior_curve = sample_entry["prior"]["curve"]
    assert prior_curve["x"], "Prior curve must provide density support"
    assert prior_curve["y"], "Prior curve must provide density values"

    grouped_bases = {entry["base"]: entry for entry in variable_payload["base_options"]}
    assert grouped_bases["weights"]["has_indices"] is True


def test_dashboard_variable_autocorrelation_metadata_structure():
    """Autocorrelation diagnostics should accompany posterior variables."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.2],
            "mean_y": [0.22],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.12],
            "cum_gain": [0.39],
        }
    )

    class DummyLoo:
        loo = 91.3
        loo_se = 3.7
        pareto_k = np.array([0.16])

    scored_test = pl.DataFrame(
        {
            "eta": [0.21, 0.29],
            "p": [0.55, 0.6],
            "p_true": [0.5, 0.61],
            "naics_L2_effect": [0.025, 0.031],
            "zip_L2_effect": [0.012, 0.014],
            "any_backoff": [False, True],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.032,
        "naics_base": np.array([0.02], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.011], dtype=float),
        "zip_deltas": [],
    }

    draws = 180
    chains = 2
    beta0_values = np.sin(np.linspace(0.0, 6.0, num=draws, dtype=float)) * 0.01
    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.vstack([beta0_values, beta0_values * 1.02 + 0.0005]),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior)

    dashboard = build_dashboard(
        train_summary={
            "n_train": 200,
            "train_positive_rate": 0.19,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.035,
            "brier": 0.225,
            "log_loss": 0.57,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )

    autocorr = beta0_entry.get("autocorrelation")
    assert autocorr is not None, "Autocorrelation diagnostics should be included"

    expected_max_lag = min(100, int(draws * 0.1))
    assert autocorr.get("max_lag") == expected_max_lag

    lags = autocorr.get("lags")
    assert lags == list(range(expected_max_lag + 1)), "Lags should span from 0"

    per_chain = autocorr.get("per_chain") or []
    assert len(per_chain) == chains
    for idx, chain_data in enumerate(per_chain):
        assert chain_data.get("chain_index") == idx
        values = chain_data.get("values") or []
        assert len(values) == len(lags)
        assert all(-1.01 <= float(val) <= 1.01 for val in values)

    guidance = autocorr.get("interpretation")
    assert isinstance(guidance, str) and guidance
    assert "lag" in guidance.lower()


def test_dashboard_variable_autocorrelation_respects_lag_rule():
    """Autocorrelation should include lags up to min(100, 10% of draws)."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.19],
            "mean_y": [0.21],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.05],
            "cum_gain": [0.36],
        }
    )

    class DummyLoo:
        loo = 77.1
        loo_se = 3.5
        pareto_k = np.array([0.2])

    scored_test = pl.DataFrame(
        {
            "eta": [0.2, 0.27],
            "p": [0.53, 0.58],
            "p_true": [0.5, 0.6],
            "naics_L2_effect": [0.02, 0.028],
            "zip_L2_effect": [0.012, 0.013],
            "any_backoff": [False, False],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.025,
        "naics_base": np.array([0.018], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.009], dtype=float),
        "zip_deltas": [],
    }

    draws = 2400
    chains = 2
    base_line = np.linspace(-0.02, 0.02, num=draws, dtype=float)
    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.vstack([base_line, base_line * 0.95 + 0.0008]),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior)

    dashboard = build_dashboard(
        train_summary={
            "n_train": 220,
            "train_positive_rate": 0.2,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.04,
            "brier": 0.24,
            "log_loss": 0.6,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )
    autocorr = beta0_entry.get("autocorrelation")
    assert autocorr is not None

    expected_max_lag = min(100, int(draws * 0.1))
    assert expected_max_lag == 100

    lags = autocorr.get("lags")
    assert lags is not None
    assert lags[-1] >= expected_max_lag, (
        "Autocorrelation output should reach the required maximum lag"
    )

    per_chain = autocorr.get("per_chain") or []
    assert len(per_chain) == chains
    for chain_data in per_chain:
        values = chain_data.get("values") or []
        assert len(values) == len(lags)
        assert chain_data.get("values")[0] == pytest.approx(1.0, abs=1e-6)


def test_dashboard_variable_ppc_overlay_structure():
    """PPC overlays should provide KDE and observed histogram context."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.2],
            "mean_y": [0.22],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.14],
            "cum_gain": [0.4],
        }
    )

    class DummyLoo:
        loo = 90.2
        loo_se = 3.4
        pareto_k = np.array([0.18])

    scored_test = pl.DataFrame(
        {
            "eta": [0.18, 0.28],
            "p": [0.53, 0.6],
            "p_true": [0.5, 0.61],
            "naics_L2_effect": [0.02, 0.031],
            "zip_L2_effect": [0.011, 0.013],
            "any_backoff": [False, True],
        }
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.03,
        "naics_base": np.array([0.02], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.025, 0.03, 0.031, 0.033],
                        [0.028, 0.032, 0.034, 0.035],
                    ]
                ),
            )
        }
    )
    predictive = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.02, 0.024, 0.029, 0.035],
                        [0.021, 0.026, 0.03, 0.036],
                    ]
                ),
            )
        }
    )
    observed = xr.Dataset(
        {
            "beta0": (
                ("obs",),
                np.array([0.02, 0.027, 0.033], dtype=float),
            )
        }
    )
    idata = az.InferenceData(
        posterior=posterior,
        posterior_predictive=predictive,
        observed_data=observed,
    )

    dashboard = build_dashboard(
        train_summary={
            "n_train": 210,
            "train_positive_rate": 0.2,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.035,
            "brier": 0.23,
            "log_loss": 0.58,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )
    ppc = beta0_entry.get("ppc")
    assert ppc is not None, "Expected PPC overlay metadata"

    kde = ppc.get("kde") or {}
    assert len(kde.get("x", [])) >= 50
    assert len(kde.get("x", [])) == len(kde.get("y", []))
    assert kde.get("normalization") == "density"

    observed_meta = ppc.get("observed") or {}
    assert observed_meta.get("histnorm") == "probability density"
    assert observed_meta.get("samples") == pytest.approx([0.02, 0.027, 0.033])

    predictive_values = [0.02, 0.024, 0.029, 0.035, 0.021, 0.026, 0.03, 0.036]
    expected_min = min(min(predictive_values), 0.02)
    expected_max = max(max(predictive_values), 0.033)
    assert ppc.get("x_range") == pytest.approx(
        [expected_min, expected_max], rel=1e-6, abs=1e-6
    )


def test_dashboard_variable_ppc_overlay_range_alignment():
    """PPC overlay x-range should match observed histogram support."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.2],
            "mean_y": [0.22],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.1],
            "cum_gain": [0.38],
        }
    )

    class DummyLoo:
        loo = 88.5
        loo_se = 3.3
        pareto_k = np.array([0.15])

    scored_test = pl.DataFrame(
        {
            "eta": [0.2, 0.3],
            "p": [0.54, 0.6],
            "p_true": [0.5, 0.62],
            "naics_L2_effect": [0.02, 0.03],
            "zip_L2_effect": [0.011, 0.013],
            "any_backoff": [False, False],
        }
    )

    posterior = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.01, 0.02, 0.03, 0.04],
                        [0.015, 0.025, 0.035, 0.045],
                    ]
                ),
            )
        }
    )
    predictive = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.005, 0.011, 0.018, 0.028],
                        [0.007, 0.013, 0.02, 0.03],
                    ]
                ),
            )
        }
    )
    observed = xr.Dataset(
        {
            "beta0": (
                ("obs",),
                np.array([0.008, 0.019, 0.029, 0.031], dtype=float),
            )
        }
    )
    idata = az.InferenceData(
        posterior=posterior,
        posterior_predictive=predictive,
        observed_data=observed,
    )

    naics_maps = [{"52": 0}]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.025,
        "naics_base": np.array([0.02], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    dashboard = build_dashboard(
        train_summary={
            "n_train": 205,
            "train_positive_rate": 0.21,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.034,
            "brier": 0.229,
            "log_loss": 0.585,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=[2],
        zip_cut_points=[2],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=idata,
        prefix_fill="0",
    )

    variable_payload = dashboard["variables"]
    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )
    ppc = beta0_entry.get("ppc")
    assert ppc is not None
    x_range = ppc.get("x_range")
    assert x_range is not None

    predictive_values = [0.005, 0.011, 0.018, 0.028, 0.007, 0.013, 0.02, 0.03]
    observed_values = [0.008, 0.019, 0.029, 0.031]
    expected_min = min(min(predictive_values), min(observed_values))
    expected_max = max(max(predictive_values), max(observed_values))
    assert x_range[0] <= expected_min
    assert x_range[1] >= expected_max
    assert (x_range[1] - x_range[0]) > 0, "Range should remain positive"


def test_dashboard_hierarchy_tab_plots():
    """Hierarchy metadata should describe levels and aggregate effects."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0, 0.5],
            "bin_high": [0.5, 1.0],
            "mean_p": [0.2, 0.6],
            "mean_y": [0.25, 0.62],
            "gap": [0.05, 0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.3],
            "cum_gain": [0.48],
        }
    )

    class DummyLoo:
        loo = 76.2
        loo_se = 5.1
        pareto_k = np.array([0.12, 0.21])

    scored_test = pl.DataFrame(
        {
            "eta": [0.3, 0.45],
            "p": [0.57, 0.61],
            "p_true": [0.54, 0.6],
            "naics_L2_effect": [0.08, -0.02],
            "naics_L4_effect": [0.03, -0.01],
            "zip_L2_effect": [0.02, 0.01],
            "any_backoff": [False, True],
        }
    )

    naics_cut_points = [2, 4]
    naics_maps = [{"52": 0, "53": 1}, {"5201": 0, "5301": 1}]
    zip_cut_points = [2]
    zip_maps = [{"45": 0, "46": 1}]
    effects = {
        "beta0": 0.1,
        "naics_base": np.array([0.12, -0.05], dtype=float),
        "naics_deltas": [np.array([0.03, -0.02], dtype=float)],
        "zip_base": np.array([0.02, -0.01], dtype=float),
        "zip_deltas": [],
    }

    dashboard = build_dashboard(
        train_summary={
            "n_train": 180,
            "train_positive_rate": 0.21,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.04,
            "brier": 0.22,
            "log_loss": 0.59,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=naics_cut_points,
        zip_cut_points=zip_cut_points,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=None,
        prefix_fill="0",
    )

    hierarchy = dashboard["hierarchy"]
    assert set(hierarchy) == {"naics", "zip"}

    naics_summary = hierarchy["naics"]
    assert naics_summary["total_nodes"] == 4
    assert naics_summary["max_abs_effect"] == pytest.approx(0.12, rel=1e-6)
    naics_levels = naics_summary["levels"]
    assert len(naics_levels) == len(naics_cut_points)
    assert naics_levels[0]["cut_point"] == 2
    assert naics_levels[0]["group_count"] == 2
    assert naics_levels[1]["mean_abs_effect"] == pytest.approx(0.025, abs=1e-3)

    zip_summary = hierarchy["zip"]
    assert zip_summary["total_nodes"] == 2
    assert zip_summary["levels"][0]["group_count"] == 2


def test_dashboard_validation_tab_plots():
    """Validation metadata should summarize calibration and ranking diagnostics."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0, 0.2, 0.4],
            "bin_high": [0.2, 0.4, 0.6],
            "mean_p": [0.1, 0.3, 0.5],
            "mean_y": [0.12, 0.28, 0.52],
            "gap": [0.02, -0.02, 0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10, 20, 40],
            "lift": [1.4, 1.2, 1.05],
            "cum_gain": [0.5, 0.62, 0.73],
        }
    )

    class DummyLoo:
        loo = 64.3
        loo_se = 6.2
        pareto_k = np.array([0.18, 0.24, 0.31])

    scored_test = pl.DataFrame(
        {
            "eta": [0.2, 0.4, 0.6, 0.8],
            "p": [0.55, 0.6, 0.65, 0.7],
            "p_true": [0.5, 0.6, 0.64, 0.71],
            "naics_L2_effect": [0.04, 0.05, 0.06, 0.07],
            "zip_L2_effect": [0.01, 0.015, 0.02, 0.025],
            "any_backoff": [False, False, True, False],
        }
    )

    naics_cut_points = [2]
    naics_maps = [{"52": 0}]
    zip_cut_points = [2]
    zip_maps = [{"45": 0}]
    effects = {
        "beta0": 0.08,
        "naics_base": np.array([0.04], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.01], dtype=float),
        "zip_deltas": [],
    }

    dashboard = build_dashboard(
        train_summary={
            "n_train": 160,
            "train_positive_rate": 0.23,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.035,
            "brier": 0.2,
            "log_loss": 0.55,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=naics_cut_points,
        zip_cut_points=zip_cut_points,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=None,
        prefix_fill="0",
    )

    validation_meta = dashboard["validation_detail"]
    assert "calibration" in validation_meta
    assert validation_meta["calibration"]["bin_count"] == reliability.height
    assert validation_meta["calibration"]["mean_gap"] == pytest.approx(
        float(reliability["gap"].abs().mean())
    )

    ranking_meta = validation_meta["ranking"]
    assert ranking_meta["k_values"] == [10, 20, 40]
    assert ranking_meta["best_lift"] == pytest.approx(1.4)

    dist_meta = validation_meta["distribution"]
    assert dist_meta["p_mean"] == pytest.approx(float(scored_test["p"].mean()))
    assert dist_meta["p_std"] > 0


def test_dashboard_inference_scoring_round_trip():
    """Inference bundle should surface top-code suggestions for scoring UI."""

    reliability = pl.DataFrame(
        {
            "bin_low": [0.0],
            "bin_high": [0.5],
            "mean_p": [0.22],
            "mean_y": [0.24],
            "gap": [0.02],
        }
    )
    ranking_summary = pl.DataFrame(
        {
            "k_pct": [10],
            "lift": [1.35],
            "cum_gain": [0.46],
        }
    )

    class DummyLoo:
        loo = 70.4
        loo_se = 4.8
        pareto_k = np.array([0.14])

    scored_test = pl.DataFrame(
        {
            "NAICS": ["522120", "541110", "522110"],
            "ZIP": ["45242", "60601", "30301"],
            "eta": [0.5, 0.3, 0.2],
            "p": [0.62, 0.57, 0.55],
            "any_backoff": [False, True, False],
        }
    )

    naics_cut_points = [2, 6]
    naics_maps = [{"52": 0, "54": 1}, {"522120": 0, "541110": 1, "522110": 2}]
    zip_cut_points = [3, 5]
    zip_maps = [{"452": 0, "606": 1, "303": 2}, {"45242": 0, "60601": 1, "30301": 2}]
    effects = {
        "beta0": 0.08,
        "naics_base": np.array([0.05, -0.02], dtype=float),
        "naics_deltas": [np.array([0.04, -0.03, 0.01], dtype=float)],
        "zip_base": np.array([0.02, -0.01, 0.0], dtype=float),
        "zip_deltas": [np.array([0.015, -0.005, 0.0], dtype=float)],
    }

    dashboard = build_dashboard(
        train_summary={
            "n_train": 140,
            "train_positive_rate": 0.22,
        },
        calibration={
            "reliability": reliability,
            "ece": 0.03,
            "brier": 0.19,
            "log_loss": 0.54,
        },
        ranking={"summary": ranking_summary},
        loo=DummyLoo(),
        scored_test=scored_test,
        parameter_alignment=None,
        output_dir=None,
        naics_cut_points=naics_cut_points,
        zip_cut_points=zip_cut_points,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        idata=None,
        prefix_fill="0",
    )

    inference_bundle = dashboard["inference"]
    suggestions = inference_bundle.get("suggestions")
    assert suggestions is not None
    assert suggestions["naics"][0]["code"] == "522120"
    assert suggestions["zip"][0]["code"] == "45242"
    assert suggestions["naics"][0]["probability"] == pytest.approx(0.62, rel=1e-3)
