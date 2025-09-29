from __future__ import annotations

import numpy as np
import pymc as pm

from hierarchical_naics_model.diagnostics import (
    compute_rhat,
    posterior_predictive_checks,
    sample_ppc,
    extract_observed,
    compute_ppc_metrics,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model


def test_compute_rhat_on_simple_model(fitted_model_idata):
    _model, idata = fitted_model_idata
    rhats = compute_rhat(idata, var_names=["beta0"])
    assert "beta0" in rhats
    assert np.isfinite(rhats["beta0"]) and rhats["beta0"] > 0


def test_ppc_metrics(fitted_model_idata):
    model, idata = fitted_model_idata
    metrics = posterior_predictive_checks(
        model, idata, observed_name="is_written", samples=50
    )
    # Basic sanity: means present and within [0,1]
    assert "mean_ppc" in metrics
    assert 0.0 <= metrics["mean_ppc"] <= 1.0
    if "mean_obs" in metrics:
        assert 0.0 <= metrics["mean_obs"] <= 1.0
        assert metrics["abs_err_mean"] >= 0


def test_compute_rhat_fallbacks(monkeypatch, model_inputs):
    # Force az.summary to raise so we go through rhat fallback
    import arviz as az
    import pandas as pd

    def raise_summary(*_a, **_k):
        raise RuntimeError("no summary")

    class FakeDA:
        def to_series(self):
            return pd.Series({"beta0": 1.03})

    monkeypatch.setattr(az, "summary", raise_summary)
    monkeypatch.setattr(az, "rhat", lambda *a, **k: FakeDA())

    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=30, tune=30, chains=2, cores=1, progressbar=False, random_seed=3
        )
    rh = compute_rhat(idata, var_names=["beta0"])  # should use fallback
    assert "beta0" in rh and rh["beta0"] > 1.0


def test_compute_rhat_default_fill(monkeypatch, model_inputs):
    # Both summary and rhat fail -> default fill with 1.0
    import arviz as az

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(az, "summary", boom)
    monkeypatch.setattr(az, "rhat", boom)

    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=10, tune=10, chains=2, cores=1, progressbar=False, random_seed=4
        )
    rh = compute_rhat(idata, var_names=["beta0"])  # filled to 1.0
    assert rh.get("beta0") == 1.0


def test_ppc_extract_observed_fallback(monkeypatch, model_inputs):
    # Make az.extract fail so we only return mean_ppc
    import arviz as az
    import numpy as np
    import pymc as pm

    def boom_extract(*_a, **_k):
        raise RuntimeError("no extract")

    monkeypatch.setattr(az, "extract", boom_extract)

    # Monkeypatch posterior predictive to return a small array
    def fake_ppc(*_a, **_k):
        return {"is_written": np.random.randint(0, 2, size=(2, 10))}

    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=20, tune=20, chains=2, cores=1, progressbar=False, random_seed=5
        )
    metrics = posterior_predictive_checks(model, idata)
    assert "mean_ppc" in metrics and "mean_obs" not in metrics


def test_sampling_helpers_and_1d_metrics(fitted_model_idata):
    # Exercise sample_ppc and extract_observed directly and cover 1D metrics path
    model, idata = fitted_model_idata
    ppc = sample_ppc(model, idata, observed_name="is_written", random_seed=123)
    assert "is_written" in ppc
    y_obs = extract_observed(idata, observed_name="is_written")
    # Create a 1D y_ppc to trigger the 1D branch in compute_ppc_metrics
    y_ppc_1d = np.asarray(ppc["is_written"]).reshape(-1)
    metrics = compute_ppc_metrics(y_ppc_1d, y_obs)
    assert "mean_ppc" in metrics
