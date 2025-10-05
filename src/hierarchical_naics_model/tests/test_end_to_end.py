from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import polars as pl
import pytest

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
    )

    figure = dashboard["figure"]
    assert figure.data and len(figure.data) >= 3

    annotation_objs = (
        tuple(figure.layout.annotations) if figure.layout.annotations else ()
    )
    annotations = [getattr(ann, "text", "") for ann in annotation_objs]
    assert any("Fit" in text for text in annotations)
    assert any("Validation" in text for text in annotations)
    assert any("Test" in text for text in annotations)

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


@pytest.mark.skip(reason="stub")
def test_end_to_end_dashboard_structure():
    """Placeholder for additional dashboard-focused assertions."""
    pass
