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
from hierarchical_naics_model.reporting.dashboard import build_dashboard


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
    assert variable_payload["max_trace"] == 200
    assert len(variable_payload["variables"]) >= 2

    beta0_entry = next(
        item for item in variable_payload["variables"] if item["id"].startswith("beta0")
    )
    assert beta0_entry["label"].startswith("beta0")
    assert beta0_entry["r_hat"] == pytest.approx(1.0, abs=0.25)
    assert beta0_entry["ess_bulk"] > 0
    assert beta0_entry["samples"]


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
