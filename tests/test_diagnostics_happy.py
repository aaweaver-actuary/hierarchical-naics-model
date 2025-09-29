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


def test_student_t_branch_and_health(model_inputs):
    # Exercise use_student_t=True path and basic health checks
    model = build_conversion_model(**model_inputs, use_student_t=True)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=50,
            tune=50,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=9,
            target_accept=0.95,
        )
    # No (or very few) divergences expected for this small model
    if "sample_stats" in idata.groups():  # type: ignore[attr-defined]
        div = idata.sample_stats.get("diverging")  # type: ignore[attr-defined]
        if div is not None:
            assert int(div.sum()) == 0
    # PPC basic calibration
    metrics = posterior_predictive_checks(model, idata)
    assert metrics["mean_ppc"] >= 0.0 and metrics["mean_ppc"] <= 1.0
