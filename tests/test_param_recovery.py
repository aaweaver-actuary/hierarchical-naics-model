from __future__ import annotations

import numpy as np
import pymc as pm

from hierarchical_naics_model.build_conversion_model import build_conversion_model


def test_intercept_recovery_simple():
    rng = np.random.default_rng(123)
    N = 600
    beta0_true = -0.7
    # Create trivial single-group indices (no variation), so model relies on beta0
    naics_levels = np.zeros((N, 1), dtype=int)
    zip_levels = np.zeros((N, 1), dtype=int)
    naics_group_counts = [1]
    zip_group_counts = [1]
    p = 1 / (1 + np.exp(-(beta0_true)))
    y = rng.binomial(1, p, size=N).astype("int8")

    model = build_conversion_model(
        y=y,
        naics_levels=naics_levels,
        zip_levels=zip_levels,
        naics_group_counts=naics_group_counts,
        zip_group_counts=zip_group_counts,
        target_accept=0.9,
    )
    with model:
        idata = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=321,
            return_inferencedata=True,
        )

        beta0_mean = float(idata.posterior["beta0"].mean().values)  # type: ignore[attr-defined]
    # With N=600 we expect a decent recovery; allow a reasonable tolerance
    assert abs(beta0_mean - beta0_true) < 0.35
