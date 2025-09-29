from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from hierarchical_naics_model.diagnostics import (
    compute_rhat,
    posterior_predictive_checks,
    sample_ppc,
    extract_observed,
    compute_ppc_metrics,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model


@pytest.fixture
def simple_model_rhat(fitted_model_idata):
    _model, idata = fitted_model_idata
    return compute_rhat(idata, var_names=["beta0"])


def test_rhat_contains_beta0_key_in_simple_model(simple_model_rhat):
    assert "beta0" in simple_model_rhat


def test_rhat_value_is_finite_and_positive_for_beta0_in_simple_model(simple_model_rhat):
    assert np.isfinite(simple_model_rhat["beta0"]) and simple_model_rhat["beta0"] > 0


@pytest.fixture
def ppc_metrics_result(fitted_model_idata):
    model, idata = fitted_model_idata
    return posterior_predictive_checks(
        model, idata, observed_name="is_written", samples=50
    )


def test_ppc_metrics_contains_mean_ppc_key(ppc_metrics_result):
    assert "mean_ppc" in ppc_metrics_result


def test_ppc_metrics_mean_ppc_in_unit_interval(ppc_metrics_result):
    assert 0.0 <= ppc_metrics_result["mean_ppc"] <= 1.0


@pytest.mark.parametrize("key", ["mean_obs"])
def test_ppc_metrics_mean_obs_in_unit_interval_if_present(ppc_metrics_result, key):
    if key in ppc_metrics_result:
        assert 0.0 <= ppc_metrics_result[key] <= 1.0


def test_ppc_metrics_abs_err_mean_non_negative_if_present(ppc_metrics_result):
    if "abs_err_mean" in ppc_metrics_result:
        assert ppc_metrics_result["abs_err_mean"] >= 0


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


@pytest.fixture
def sampled_ppc_and_observed(fitted_model_idata):
    # Returns model, idata, ppc, y_obs, y_ppc_1d for parametrized tests
    model, idata = fitted_model_idata
    ppc = sample_ppc(model, idata, observed_name="is_written", random_seed=123)
    y_obs = extract_observed(idata, observed_name="is_written")
    y_ppc_1d = np.asarray(ppc["is_written"]).reshape(-1)
    return {
        "model": model,
        "idata": idata,
        "ppc": ppc,
        "y_obs": y_obs,
        "y_ppc_1d": y_ppc_1d,
    }


def test_sample_ppc_returns_is_written_key(sampled_ppc_and_observed):
    assert "is_written" in sampled_ppc_and_observed["ppc"]


def test_extract_observed_returns_non_empty_array(sampled_ppc_and_observed):
    y_obs = sampled_ppc_and_observed["y_obs"]
    assert y_obs is not None and np.size(y_obs) > 0


def test_compute_ppc_metrics_returns_mean_ppc_key_for_1d_ppc(sampled_ppc_and_observed):
    metrics = compute_ppc_metrics(
        sampled_ppc_and_observed["y_ppc_1d"], sampled_ppc_and_observed["y_obs"]
    )
    assert "mean_ppc" in metrics


@pytest.mark.parametrize(
    "ppc_shape",
    [
        (10,),  # 1D array
        (2, 10),  # 2D array
        (2, 5, 2),  # 3D array
    ],
)
def test_compute_ppc_metrics_handles_various_ppc_shapes(
    ppc_shape, sampled_ppc_and_observed
):
    # Edge case: test compute_ppc_metrics with different shapes
    y_ppc = np.random.randint(0, 2, size=ppc_shape)
    y_obs = sampled_ppc_and_observed["y_obs"]
    metrics = compute_ppc_metrics(y_ppc, y_obs)
    assert "mean_ppc" in metrics


@pytest.mark.parametrize(
    "y_ppc, y_obs, expected",
    [
        (np.zeros(10), np.zeros(10), 0.0),  # All zeros
        (np.ones(10), np.ones(10), 1.0),  # All ones
        (np.ones(10), np.zeros(10), 1.0),  # All ones vs all zeros
        (np.zeros(10), np.ones(10), 0.0),  # All zeros vs all ones
    ],
)
def test_compute_ppc_metrics_mean_ppc_matches_expected(y_ppc, y_obs, expected):
    metrics = compute_ppc_metrics(y_ppc, y_obs)
    assert metrics["mean_ppc"] == expected
