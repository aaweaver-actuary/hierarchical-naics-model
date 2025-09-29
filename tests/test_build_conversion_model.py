from __future__ import annotations

import numpy as np
import pymc as pm

from hierarchical_naics_model.build_conversion_model import build_conversion_model


def test_model_builds_and_dims(model_inputs):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    assert isinstance(model, pm.Model)

    with model:
        # Ensure key RVs exist
        assert "beta0" in model.named_vars
        # Ensure per-level parameters created
        L_naics = model_inputs["naics_levels"].shape[1]
        L_zip = model_inputs["zip_levels"].shape[1]
        for j in range(L_naics):
            assert f"naics_mu_{j}" in model.named_vars
            assert f"naics_sigma_{j}" in model.named_vars
            assert f"naics_eff_{j}" in model.named_vars
        for j in range(L_zip):
            assert f"zip_mu_{j}" in model.named_vars
            assert f"zip_sigma_{j}" in model.named_vars
            assert f"zip_eff_{j}" in model.named_vars


def test_prior_predictive_runs(model_inputs):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        prior = pm.sample_prior_predictive(samples=100)
    # Check that prior predictive includes observed variable draws
    if hasattr(prior, "prior_predictive"):
        y_draws = prior.prior_predictive.get("is_written")  # type: ignore[attr-defined]
    else:
        y_draws = prior.get("is_written")  # type: ignore[call-arg]
    assert y_draws is not None
    # Shape: (chains?, draws, N) -> new PyMC returns (samples, N)
    arr = np.asarray(y_draws)
    assert arr.ndim >= 2
    assert arr.shape[-1] == model_inputs["y"].shape[0]


def test_posterior_sampling_smoke(model_inputs):
    # Subset to keep runtime low while still exercising NUTS
    y = model_inputs["y"]
    n = min(200, y.shape[0])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(y.shape[0], size=n, replace=False))

    sub_inputs = {
        "y": model_inputs["y"][take],
        "naics_levels": model_inputs["naics_levels"][take, :],
        "zip_levels": model_inputs["zip_levels"][take, :],
        # Group counts remain the same (random effects defined over full groups)
        "naics_group_counts": model_inputs["naics_group_counts"],
        "zip_group_counts": model_inputs["zip_group_counts"],
    }

    model = build_conversion_model(**sub_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=2025,
            progressbar=False,
        )

    # Basic sanity: posterior exists and has samples for key RVs
    assert "posterior" in idata.groups()
    for name in ["beta0", "eta", "p"]:
        assert name in idata.posterior or name in idata.posterior.coords  # type: ignore[attr-defined]
