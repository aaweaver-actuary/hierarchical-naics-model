# tests/test_model_nested_deltas_shape_and_ppc.py
import numpy as np
import pymc as pm
import pytest
from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.build_conversion_model import (
    build_conversion_model,
)


def _simulate(n=400, seed=0):
    rng = np.random.default_rng(seed)
    naics_pool = np.array(["511110", "511120", "512130", "522110", "522120"])
    zip_pool = np.array(["02139", "30309", "45040", "30310", "45039"])

    naics = rng.choice(naics_pool, size=n, replace=True)
    zips = rng.choice(zip_pool, size=n, replace=True)

    # True DGP: base at level-0 and small deltas at deeper levels
    cut_naics = [2, 3, 6]
    cut_zip = [2, 3, 5]

    idx_n = build_hierarchical_indices(naics, cut_points=cut_naics, prefix_fill="0")
    idx_z = build_hierarchical_indices(zips, cut_points=cut_zip, prefix_fill="0")

    # Create latent effects
    rng2 = np.random.default_rng(seed + 1)
    beta0 = -1.2
    naics_base = rng2.normal(0.2, 0.35, size=idx_n["group_counts"][0])
    naics_delta1 = rng2.normal(0.0, 0.25, size=idx_n["group_counts"][1])
    naics_delta2 = rng2.normal(0.0, 0.20, size=idx_n["group_counts"][2])

    zip_base = rng2.normal(0.1, 0.30, size=idx_z["group_counts"][0])
    zip_delta1 = rng2.normal(0.0, 0.20, size=idx_z["group_counts"][1])
    zip_delta2 = rng2.normal(0.0, 0.15, size=idx_z["group_counts"][2])

    eta = (
        beta0
        + naics_base[idx_n["code_levels"][:, 0]]
        + naics_delta1[idx_n["code_levels"][:, 1]]
        + naics_delta2[idx_n["code_levels"][:, 2]]
        + zip_base[idx_z["code_levels"][:, 0]]
        + zip_delta1[idx_z["code_levels"][:, 1]]
        + zip_delta2[idx_z["code_levels"][:, 2]]
    )
    p = 1.0 / (1.0 + np.exp(-eta))
    y = np.random.binomial(1, p, size=n).astype("int8")

    return y, idx_n, idx_z


@pytest.fixture(scope="module")
def simulated_data():
    y, idx_n, idx_z = _simulate(n=300, seed=42)
    return y, idx_n, idx_z


@pytest.fixture(scope="module")
def built_model(simulated_data):
    y, idx_n, idx_z = simulated_data
    model = build_conversion_model(
        y=y,
        naics_levels=idx_n["code_levels"],
        zip_levels=idx_z["code_levels"],
        naics_group_counts=idx_n["group_counts"],
        zip_group_counts=idx_z["group_counts"],
        target_accept=0.9,
    )
    return model


@pytest.fixture(scope="module")
def sampled_idata(built_model):
    with built_model:
        idata = pm.sample(
            500, tune=500, chains=2, target_accept=0.9, cores=1, progressbar=False
        )
    return idata


def test_model_is_instance_of_pm_Model(built_model):
    assert isinstance(built_model, pm.Model)


def test_posterior_p_shape_matches_y_length(simulated_data, sampled_idata):
    y, _, _ = simulated_data
    p_post = sampled_idata.posterior["p"].mean(dim=("chain", "draw")).to_numpy()
    assert p_post.shape[0] == y.shape[0]


def test_posterior_p_is_finite(sampled_idata):
    p_post = sampled_idata.posterior["p"].mean(dim=("chain", "draw")).to_numpy()
    assert np.isfinite(p_post).all()


def test_posterior_p_within_zero_one(sampled_idata):
    p_post = sampled_idata.posterior["p"].mean(dim=("chain", "draw")).to_numpy()
    assert (p_post >= 0).all() and (p_post <= 1).all()


@pytest.mark.parametrize(
    "n,seed",
    [
        (1, 0),  # minimal case
        (10, 123),  # small sample
        (300, 42),  # typical case
        (1000, 999),  # large sample
    ],
)
def test_simulate_edge_cases(n, seed):
    y, idx_n, idx_z = _simulate(n=n, seed=seed)
    assert y.shape[0] == n
