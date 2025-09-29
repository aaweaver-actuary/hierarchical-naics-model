from __future__ import annotations

import os
from typing import Dict

import numpy as np
import polars as pl
import pymc as pm
import pytest

from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model
import pytest


def _posterior_group_mean(idata, rv_name: str) -> np.ndarray:
    """Posterior mean over chain, draw for a group-indexed RV."""
    da = idata.posterior[rv_name]  # type: ignore[index]
    return np.asarray(da.mean(dim=("chain", "draw")).values)


@pytest.fixture(scope="module")
def shrinkage_synth_data(tmp_path):
    rng = np.random.default_rng(42)
    codes = [
        ("07121", 1800),
        ("07122", 1800),
        ("07123", 1800),
        ("97121", 1800),
        ("97122", 1800),
        ("97123", 30),
    ]
    beta0_true = -1.0
    eff_5digit = {"0712": 0.25, "9712": 0.25}
    eff_6digit = {
        "07121": -0.8,
        "07122": -0.3,
        "07123": +1.0,
        "97121": -0.8,
        "97122": -0.3,
        "97123": +1.0,
    }
    naics_list, zip_list, logits = [], [], []
    for code, n in codes:
        for _ in range(n):
            naics_list.append(code)
            zip_list.append("00000")
            l5 = eff_5digit[code[:4]]
            l6 = eff_6digit[code]
            logits.append(beta0_true + l5 + l6)
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y = np.asarray(rng.binomial(1, p, size=p.shape), dtype="int8")
    df_pl = pl.DataFrame(
        {
            "is_written": y.tolist(),
            "naics": naics_list,
            "zip": zip_list,
            "p_true": p.tolist(),
        }
    )
    pq_path = tmp_path / "shrinkage_synth.parquet"
    df_pl.write_parquet(str(pq_path))
    means_rows = (
        df_pl.group_by("naics")
        .agg(pl.col("is_written").mean().alias("mean_y"))
        .iter_rows(named=True)
    )
    means = {row["naics"]: float(row["mean_y"]) for row in means_rows}
    ptrue_rows = (
        df_pl.group_by("naics")
        .agg(pl.col("p_true").mean().alias("p_true_mean"))
        .iter_rows(named=True)
    )
    ptrue = {row["naics"]: float(row["p_true_mean"]) for row in ptrue_rows}
    naics_idx = build_hierarchical_indices(naics_list, cut_points=[2, 3, 4, 5, 6])
    zip_idx = build_hierarchical_indices(zip_list, cut_points=[1])
    model = build_conversion_model(
        y=y,
        naics_levels=np.asarray(naics_idx["code_levels"]),
        zip_levels=np.asarray(zip_idx["code_levels"]),
        naics_group_counts=list(naics_idx["group_counts"]),
        zip_group_counts=list(zip_idx["group_counts"]),
        target_accept=0.95,
    )
    with model:
        idata = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=42,
            target_accept=0.95,
        )
    l6_map = naics_idx["maps"][4]
    eff6_mean = _posterior_group_mean(idata, "naics_eff_4")

    def eff6(name: str) -> float:
        return float(eff6_mean[l6_map[name]])

    eta = idata.posterior["eta"].mean(dim=("chain", "draw")).values
    first_idx = {}
    for i, code in enumerate(naics_list):
        if code not in first_idx:
            first_idx[code] = i
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    return {
        "means": means,
        "ptrue": ptrue,
        "eff6": eff6,
        "p_hat": p_hat,
        "first_idx": first_idx,
    }


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
def test_empirical_mean_ordering_matches_intended(shrinkage_synth_data):
    means = shrinkage_synth_data["means"]
    assert means["07123"] > means["07122"] > means["07121"]


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
@pytest.mark.parametrize(
    "code,tol",
    [
        ("07123", 0.035),
        ("07122", 0.035),
        ("07121", 0.035),
        ("97123", 0.15),
    ],
)
def test_empirical_mean_close_to_true_probability(shrinkage_synth_data, code, tol):
    means = shrinkage_synth_data["means"]
    ptrue = shrinkage_synth_data["ptrue"]
    assert abs(means[code] - ptrue[code]) < tol


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
def test_well_populated_07123_distinct_from_siblings(shrinkage_synth_data):
    eff6 = shrinkage_synth_data["eff6"]
    assert eff6("07123") - eff6("07122") > 0.4


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
def test_well_populated_07122_distinct_from_07121(shrinkage_synth_data):
    eff6 = shrinkage_synth_data["eff6"]
    assert eff6("07122") - eff6("07121") > 0.2


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
def test_sparse_97123_shrinks_toward_parent_level(shrinkage_synth_data):
    eff6 = shrinkage_synth_data["eff6"]
    assert abs(eff6("97123")) < abs(eff6("07123")) - 0.2


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Skip by default; set RUN_RECOVERY=1 to run shrinkage behavior test.",
)
def test_posterior_predictive_mean_ordering_matches_expected(shrinkage_synth_data):
    p_hat = shrinkage_synth_data["p_hat"]
    first_idx = shrinkage_synth_data["first_idx"]
    assert (
        p_hat[first_idx["07123"]]
        > p_hat[first_idx["07122"]]
        > p_hat[first_idx["07121"]]
    )
