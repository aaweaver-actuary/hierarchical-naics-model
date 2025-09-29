from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from hierarchical_naics_model.build_conversion_model import build_conversion_model


@pytest.fixture
def model(model_inputs):
    return build_conversion_model(**model_inputs)  # type: ignore[missing-argument]


@pytest.fixture
def naics_zip_dims(model_inputs):
    return model_inputs["naics_levels"].shape[1], model_inputs["zip_levels"].shape[1]


def test_model_is_instance_of_pm_Model(model):
    assert isinstance(model, pm.Model)


@pytest.mark.parametrize("rv_name", ["beta0"])
def test_model_contains_key_random_variable(model, rv_name):
    assert rv_name in model.named_vars


@pytest.mark.parametrize("j", range(3))  # Assumes at least 3 NAICS levels in fixture
def test_model_contains_naics_mu_parameter(model_inputs, model, j):
    if j < model_inputs["naics_levels"].shape[1]:
        assert f"naics_mu_{j}" in model.named_vars


@pytest.mark.parametrize("j", range(3))
def test_model_contains_naics_sigma_parameter(model_inputs, model, j):
    if j < model_inputs["naics_levels"].shape[1]:
        assert f"naics_sigma_{j}" in model.named_vars


@pytest.mark.parametrize("j", range(3))
def test_model_contains_naics_eff_parameter(model_inputs, model, j):
    if j < model_inputs["naics_levels"].shape[1]:
        assert f"naics_eff_{j}" in model.named_vars


@pytest.mark.parametrize("j", range(3))  # Assumes at least 3 ZIP levels in fixture
def test_model_contains_zip_mu_parameter(model_inputs, model, j):
    if j < model_inputs["zip_levels"].shape[1]:
        assert f"zip_mu_{j}" in model.named_vars


@pytest.mark.parametrize("j", range(3))
def test_model_contains_zip_sigma_parameter(model_inputs, model, j):
    if j < model_inputs["zip_levels"].shape[1]:
        assert f"zip_sigma_{j}" in model.named_vars


@pytest.mark.parametrize("j", range(3))
def test_model_contains_zip_eff_parameter(model_inputs, model, j):
    if j < model_inputs["zip_levels"].shape[1]:
        assert f"zip_eff_{j}" in model.named_vars


@pytest.fixture
def prior_predictive(model_inputs):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        prior = pm.sample_prior_predictive(samples=100)
    return prior


def test_prior_predictive_includes_observed_variable_draws(prior_predictive):
    # Check that prior predictive includes observed variable draws
    if hasattr(prior_predictive, "prior_predictive"):
        y_draws = prior_predictive.prior_predictive.get("is_written")  # type: ignore[attr-defined]
    else:
        y_draws = prior_predictive.get("is_written")  # type: ignore[call-arg]
    assert y_draws is not None


def test_prior_predictive_draws_have_at_least_two_dimensions(
    prior_predictive, model_inputs
):
    if hasattr(prior_predictive, "prior_predictive"):
        y_draws = prior_predictive.prior_predictive.get("is_written")  # type: ignore[attr-defined]
    else:
        y_draws = prior_predictive.get("is_written")  # type: ignore[call-arg]
    arr = np.asarray(y_draws)
    assert arr.ndim >= 2


def test_prior_predictive_draws_last_dimension_matches_observed_length(
    prior_predictive, model_inputs
):
    if hasattr(prior_predictive, "prior_predictive"):
        y_draws = prior_predictive.prior_predictive.get("is_written")  # type: ignore[attr-defined]
    else:
        y_draws = prior_predictive.get("is_written")  # type: ignore[call-arg]
    arr = np.asarray(y_draws)
    assert arr.shape[-1] == model_inputs["y"].shape[0]


@pytest.mark.parametrize("samples", [1, 10, 100, 500])
def test_prior_predictive_runs_with_various_sample_sizes(model_inputs, samples):
    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        prior = pm.sample_prior_predictive(samples=samples)
    if hasattr(prior, "prior_predictive"):
        y_draws = prior.prior_predictive.get("is_written")  # type: ignore[attr-defined]
    else:
        y_draws = prior.get("is_written")  # type: ignore[call-arg]
    arr = np.asarray(y_draws)
    # In PyMC >=5 with InferenceData, dims are (chain, draw, obs)
    if hasattr(prior, "prior_predictive"):
        assert arr.shape[1] == samples
    else:
        assert arr.shape[0] == samples


@pytest.fixture
def posterior_idata(model_inputs):
    # Subset to keep runtime low while still exercising NUTS
    y = model_inputs["y"]
    n = min(200, y.shape[0])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(y.shape[0], size=n, replace=False))

    sub_inputs = {
        "y": model_inputs["y"][take],
        "naics_levels": model_inputs["naics_levels"][take, :],
        "zip_levels": model_inputs["zip_levels"][take, :],
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
    return idata


def test_posterior_inference_returns_inference_data_with_posterior_group(
    posterior_idata,
):
    assert "posterior" in posterior_idata.groups()


@pytest.mark.parametrize("rv_name", ["beta0", "eta", "p"])
def test_posterior_group_contains_expected_random_variable_or_coord(
    posterior_idata, rv_name
):
    assert (
        rv_name in posterior_idata.posterior
        or rv_name in posterior_idata.posterior.coords
    )  # type: ignore[attr-defined]


@pytest.mark.parametrize("draws", [10, 50, 100])
def test_posterior_sampling_returns_correct_number_of_draws(model_inputs, draws):
    y = model_inputs["y"]
    n = min(200, y.shape[0])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(y.shape[0], size=n, replace=False))

    sub_inputs = {
        "y": model_inputs["y"][take],
        "naics_levels": model_inputs["naics_levels"][take, :],
        "zip_levels": model_inputs["zip_levels"][take, :],
        "naics_group_counts": model_inputs["naics_group_counts"],
        "zip_group_counts": model_inputs["zip_group_counts"],
    }

    model = build_conversion_model(**sub_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=draws,
            tune=10,
            chains=1,
            cores=1,
            target_accept=0.9,
            random_seed=2025,
            progressbar=False,
        )
    assert idata.posterior.dims["draw"] == draws


@pytest.mark.parametrize("chain_count", [1, 2])
def test_posterior_sampling_returns_correct_number_of_chains(model_inputs, chain_count):
    y = model_inputs["y"]
    n = min(200, y.shape[0])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(y.shape[0], size=n, replace=False))

    sub_inputs = {
        "y": model_inputs["y"][take],
        "naics_levels": model_inputs["naics_levels"][take, :],
        "zip_levels": model_inputs["zip_levels"][take, :],
        "naics_group_counts": model_inputs["naics_group_counts"],
        "zip_group_counts": model_inputs["zip_group_counts"],
    }

    model = build_conversion_model(**sub_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=10,
            tune=10,
            chains=chain_count,
            cores=1,
            target_accept=0.9,
            random_seed=2025,
            progressbar=False,
        )
    assert idata.posterior.dims["chain"] == chain_count


@pytest.fixture
def bad_y_non_binary(model_inputs):
    bad_inputs = dict(**model_inputs)
    bad_inputs["y"] = np.array([0, 2, 1], dtype=int)
    return bad_inputs


@pytest.fixture
def bad_naics_levels_shape(model_inputs):
    bad_inputs = dict(**model_inputs)
    bad_inputs["naics_levels"] = bad_inputs["naics_levels"][:10, :]
    return bad_inputs


@pytest.fixture
def bad_naics_levels_out_of_range(model_inputs):
    bad_inputs = dict(**model_inputs)
    bad_inputs["naics_levels"] = bad_inputs["naics_levels"].copy()
    bad_inputs["naics_levels"][0, 0] = bad_inputs["naics_group_counts"][0]
    return bad_inputs


@pytest.fixture
def bad_naics_levels_non_2d(model_inputs):
    bad_inputs = dict(**model_inputs)
    bad_inputs["naics_levels"] = bad_inputs["naics_levels"].reshape(-1)
    return bad_inputs


@pytest.fixture
def bad_zip_group_counts_length(model_inputs):
    bad_inputs = dict(**model_inputs)
    bad_inputs["zip_group_counts"] = bad_inputs["zip_group_counts"][:-1]
    return bad_inputs


@pytest.fixture
def bad_zip_levels_negative_index(model_inputs):
    bad_inputs = dict(**model_inputs)
    z = bad_inputs["zip_levels"].copy()
    z[0, 0] = -1
    bad_inputs["zip_levels"] = z
    return bad_inputs


@pytest.mark.parametrize(
    "bad_fixture,expected_exception",
    [
        ("bad_y_non_binary", ValueError),
        ("bad_naics_levels_shape", ValueError),
        ("bad_naics_levels_out_of_range", ValueError),
        ("bad_naics_levels_non_2d", ValueError),
        ("bad_zip_group_counts_length", ValueError),
        ("bad_zip_levels_negative_index", ValueError),
    ],
)
def test_build_conversion_model_raises_value_error_for_invalid_inputs(
    bad_fixture, expected_exception, request
):
    bad_inputs = request.getfixturevalue(bad_fixture)
    with pytest.raises(expected_exception):
        build_conversion_model(**bad_inputs)  # type: ignore[arg-type]
