from __future__ import annotations

import pymc as pm
import numpy as np

from hierarchical_naics_model.build_conversion_model import build_conversion_model
import pytest
from hierarchical_naics_model.diagnostics import (
    compute_rhat,
    posterior_predictive_checks,
)


@pytest.fixture
def sampled_idata(model_inputs):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=60, tune=60, chains=2, cores=1, progressbar=False, random_seed=7
        )
    return model, idata


@pytest.fixture
def sampled_student_t_idata(model_inputs):
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
    return model, idata


@pytest.mark.parametrize("var_name", ["beta0", "beta1", "sigma"])
def test_compute_rhat_returns_finite_for_all_main_parameters(sampled_idata, var_name):
    _, idata = sampled_idata
    rh = compute_rhat(idata, var_names=[var_name])
    assert np.isfinite(rh[var_name])


def test_posterior_predictive_checks_returns_mean_ppc_key(sampled_idata):
    model, idata = sampled_idata
    metrics = posterior_predictive_checks(model, idata)
    assert "mean_ppc" in metrics


@pytest.mark.parametrize("ppc_key", ["mean_ppc", "std_ppc"])
def test_posterior_predictive_checks_metric_is_in_valid_range(sampled_idata, ppc_key):
    model, idata = sampled_idata
    metrics = posterior_predictive_checks(model, idata)
    value = metrics.get(ppc_key)
    assert value is None or (0.0 <= value <= 1.0)


def test_student_t_model_has_no_divergences_for_small_model(sampled_student_t_idata):
    _, idata = sampled_student_t_idata
    if "sample_stats" in idata.groups():  # type: ignore[attr-defined]
        div = idata.sample_stats.get("diverging")  # type: ignore[attr-defined]
        if div is not None:
            assert int(div.sum()) == 0


def test_student_t_model_ppc_mean_in_valid_range(sampled_student_t_idata):
    model, idata = sampled_student_t_idata
    metrics = posterior_predictive_checks(model, idata)
    assert 0.0 <= metrics["mean_ppc"] <= 1.0


@pytest.fixture
def sampled_student_t_idata_small(model_inputs):
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
    return model, idata


@pytest.mark.parametrize("ppc_key", ["mean_ppc", "std_ppc"])
def test_student_t_model_ppc_metric_in_valid_range_for_small_model(
    sampled_student_t_idata_small, ppc_key
):
    model, idata = sampled_student_t_idata_small
    metrics = posterior_predictive_checks(model, idata)
    value = metrics.get(ppc_key)
    assert value is None or (0.0 <= value <= 1.0)


def test_student_t_model_no_divergences_for_small_model(sampled_student_t_idata_small):
    _, idata = sampled_student_t_idata_small
    if "sample_stats" in idata.groups():  # type: ignore[attr-defined]
        div = idata.sample_stats.get("diverging")  # type: ignore[attr-defined]
        if div is not None:
            assert int(div.sum()) == 0


@pytest.mark.parametrize("var_name", ["beta0", "beta1", "sigma"])
def test_student_t_model_rhat_is_finite_for_main_parameters(
    sampled_student_t_idata_small, var_name
):
    _, idata = sampled_student_t_idata_small
    rh = compute_rhat(idata, var_names=[var_name])
    assert np.isfinite(rh[var_name])
