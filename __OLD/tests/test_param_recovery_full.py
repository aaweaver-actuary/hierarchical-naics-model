from __future__ import annotations

import os
from typing import List

import numpy as np
import pymc as pm
import pytest

from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model


def _posterior_mean(idata, name: str) -> float:
    # Helper to compute posterior mean for a scalar RV across chains/draws
    da = idata.posterior[name]  # type: ignore[index]
    return float(da.mean().values)


def _posterior_mean_array(idata, name: str) -> np.ndarray:
    da = idata.posterior[name]  # type: ignore[index]
    return np.asarray(da.mean(dim=("chain", "draw")).values)


@pytest.mark.skipif(
    os.environ.get("RUN_RECOVERY") != "1",
    reason="Parameter recovery test is skipped by default; set RUN_RECOVERY=1 to run.",
)
def test_full_parameter_recovery():
    rng = np.random.default_rng(1234)

    # Define groups
    naics_groups: List[str] = ["11", "12", "13", "14", "15"]
    zip_groups: List[str] = ["1", "2", "3", "4", "5"]

    # True parameters
    beta0_true = -0.8
    naics_mu_true, naics_sigma_true = 0.30, 0.40
    zip_mu_true, zip_sigma_true = -0.20, 0.30

    # Draw group effects from the hierarchical distribution
    a_naics_true = rng.normal(naics_mu_true, naics_sigma_true, size=len(naics_groups))
    b_zip_true = rng.normal(zip_mu_true, zip_sigma_true, size=len(zip_groups))

    # Build a balanced dataset: all combinations with repeats
    reps = 80  # total N = 5*5*reps = 2000
    naics_codes = []
    zip_codes = []
    for n in naics_groups:
        for z in zip_groups:
            naics_codes.extend([n] * reps)
            zip_codes.extend([z] * reps)

    # Indices (single level each)
    naics_idx = build_hierarchical_indices(naics_codes, cut_points=[2])
    zip_idx = build_hierarchical_indices(zip_codes, cut_points=[1])

    naics_levels = np.asarray(naics_idx["code_levels"])  # (N,1)
    zip_levels = np.asarray(zip_idx["code_levels"])  # (N,1)
    yN = naics_levels.shape[0]

    # Compose eta and simulate outcomes
    eta = beta0_true + a_naics_true[naics_levels[:, 0]] + b_zip_true[zip_levels[:, 0]]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=yN).astype("int8")

    # Build and fit model
    model = build_conversion_model(
        y=y,
        naics_levels=naics_levels,
        zip_levels=zip_levels,
        naics_group_counts=list(naics_idx["group_counts"]),
        zip_group_counts=list(zip_idx["group_counts"]),
        target_accept=0.95,
    )
    with model:
        idata = pm.sample(
            draws=800,
            tune=800,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=2025,
            target_accept=0.95,
        )

    # Recover parameters (posterior means close to true values)
    beta0_hat = _posterior_mean(idata, "beta0")
    naics_mu_hat = _posterior_mean(idata, "naics_mu_0")
    naics_sigma_hat = _posterior_mean(idata, "naics_sigma_0")
    zip_mu_hat = _posterior_mean(idata, "zip_mu_0")
    zip_sigma_hat = _posterior_mean(idata, "zip_sigma_0")

    # Note: beta0 and per-level mu terms are not separately identifiable;
    # the identifiable baseline is their sum.
    baseline_true = beta0_true + naics_mu_true + zip_mu_true
    baseline_hat = beta0_hat + naics_mu_hat + zip_mu_hat

    # With finite groups (5), sigma is identified via the realized group effects.
    # Compare to the empirical SD of the sampled group effects rather than the population sigma.
    sd_naics_emp = float(np.std(a_naics_true, ddof=1))
    sd_zip_emp = float(np.std(b_zip_true, ddof=1))

    # Tolerances acknowledge finite N, shrinkage, and MCMC error
    assert abs(baseline_hat - baseline_true) < 0.15
    assert abs(naics_sigma_hat - sd_naics_emp) < 0.20
    assert abs(zip_sigma_hat - sd_zip_emp) < 0.20
