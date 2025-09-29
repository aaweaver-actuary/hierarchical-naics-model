from __future__ import annotations

import pymc as pm
import numpy as np

from hierarchical_naics_model.build_conversion_model import build_conversion_model
from hierarchical_naics_model.diagnostics import (
    compute_rhat,
    posterior_predictive_checks,
)


def test_diagnostics_happy_path(model_inputs):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=60, tune=60, chains=2, cores=1, progressbar=False, random_seed=7
        )
    # compute_rhat via az.summary path
    rh = compute_rhat(idata, var_names=["beta0"])
    assert "beta0" in rh and np.isfinite(rh["beta0"])  # pragma: no branch
    # ppc path with observed extract present
    metrics = posterior_predictive_checks(model, idata)
    assert "mean_ppc" in metrics  # pragma: no branch
