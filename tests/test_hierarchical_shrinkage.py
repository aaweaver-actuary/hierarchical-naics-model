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


def _posterior_group_mean(idata, rv_name: str) -> np.ndarray:
    """Posterior mean over chain, draw for a group-indexed RV."""
    da = idata.posterior[rv_name]  # type: ignore[index]
    return np.asarray(da.mean(dim=("chain", "draw")).values)


@pytest.mark.skipif(
    os.environ.get("RUN_SHRINK") != "1",
    reason="Skip by default; set RUN_SHRINK=1 to run shrinkage behavior test.",
)
def test_hierarchical_shrinkage_cases(tmp_path):
    rng = np.random.default_rng(42)

    # Case design: 6-digit codes under two 5-digit parents '0712*' and '9712*'
    codes = [
        ("07121", 1800),
        ("07122", 1800),
        ("07123", 1800),  # well-populated
        ("97121", 1800),
        ("97122", 1800),
        ("97123", 30),  # sparse
    ]

    # True per-6-digit effects (logit contributions) and a shared 5-digit parent effect
    beta0_true = -1.0
    eff_5digit: Dict[str, float] = {
        "0712": 0.25,
        "9712": 0.25,
    }
    eff_6digit: Dict[str, float] = {
        "07121": -0.8,
        "07122": -0.3,
        "07123": +1.0,
        "97121": -0.8,  # same as 07*** counterparts
        "97122": -0.3,
        "97123": +1.0,  # sparse
    }

    # Build dataset
    naics_list: list[str] = []
    zip_list: list[str] = []
    logits: list[float] = []
    for code, n in codes:
        for _ in range(n):
            naics_list.append(code)
            zip_list.append("00000")  # trivial ZIP
            l5 = eff_5digit[code[:4]]
            l6 = eff_6digit[code]
            logits.append(beta0_true + l5 + l6)
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y = rng.binomial(1, p).astype("int8")

    # Save to parquet for traceability and quick inspection
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

    # Pre-fit sanity: aggregate means by 6-digit (pure Polars to avoid pandas/pyarrow dep)
    means_rows = (
        df_pl.group_by("naics")
        .agg(pl.col("is_written").mean().alias("mean_y"))
        .iter_rows(named=True)
    )
    means = {row["naics"]: float(row["mean_y"]) for row in means_rows}
    assert means["07123"] > means["07122"] > means["07121"]  # matches intended ordering

    # Also check empirical means are close to p_true (tighter for large N)
    ptrue_rows = (
        df_pl.group_by("naics")
        .agg(pl.col("p_true").mean().alias("p_true_mean"))
        .iter_rows(named=True)
    )
    ptrue = {row["naics"]: float(row["p_true_mean"]) for row in ptrue_rows}
    assert abs(means["07123"] - ptrue["07123"]) < 0.035
    assert abs(means["07122"] - ptrue["07122"]) < 0.035
    assert abs(means["07121"] - ptrue["07121"]) < 0.035
    # For the sparse code allow wider tolerance
    assert abs(means["97123"] - ptrue["97123"]) < 0.15

    # Indices with full NAICS hierarchy (2..6) and trivial ZIP
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

    # Extract 6-digit effects (NAICS level index 4)
    l6_map = naics_idx["maps"][4]  # type: ignore[index]
    eff6_mean = _posterior_group_mean(idata, "naics_eff_4")

    def eff6(name: str) -> float:
        return float(eff6_mean[l6_map[name]])

    # Case 1: well-populated 07123 is distinct from its siblings
    assert eff6("07123") - eff6("07122") > 0.4
    assert eff6("07122") - eff6("07121") > 0.2

    # Case 2: sparse 97123 shrinks toward parent level (smaller magnitude than 07123)
    assert abs(eff6("97123")) < abs(eff6("07123")) - 0.2

    # Additionally confirm overall ordering in expected direction via posterior predictive means
    # Compute posterior mean of p per code via eta deterministic
    eta = idata.posterior["eta"].mean(dim=("chain", "draw")).values  # type: ignore[index]
    # Use the first occurrence index for each code to compare representative predictions
    first_idx: Dict[str, int] = {}
    for i, code in enumerate(naics_list):
        if code not in first_idx:
            first_idx[code] = i
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    assert (
        p_hat[first_idx["07123"]]
        > p_hat[first_idx["07122"]]
        > p_hat[first_idx["07121"]]
    )
