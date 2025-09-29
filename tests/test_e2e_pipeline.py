from __future__ import annotations

import numpy as np
import pymc as pm

from hierarchical_naics_model.generate_synthetic_data import generate_synthetic_data
from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model
import pytest


@pytest.fixture(scope="module")
def synthetic_data():
    naics_pool = ["511110", "511120", "51213", "52", "52412", "52413"]
    zip_pool = ["30309", "94103", "10001", "02139", "73301"]
    df = generate_synthetic_data(
        n=500, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-1.2, seed=42
    )
    return df, naics_pool, zip_pool


@pytest.fixture(scope="module")
def hierarchical_indices(synthetic_data):
    df, naics_pool, zip_pool = synthetic_data
    naics_cuts = [2, 3, 6]
    zip_cuts = [2, 3, 5]
    naics_idx = build_hierarchical_indices(
        df["naics"].astype(str).tolist(), cut_points=naics_cuts
    )
    zip_idx = build_hierarchical_indices(
        df["zip"].astype(str).tolist(), cut_points=zip_cuts
    )
    return df, naics_idx, zip_idx, naics_cuts, zip_cuts


@pytest.fixture(scope="module")
def sampled_model(hierarchical_indices):
    df, naics_idx, zip_idx, naics_cuts, zip_cuts = hierarchical_indices
    y = df["is_written"].to_numpy()
    take = np.arange(min(200, len(df)))
    model = build_conversion_model(
        y=y[take],
        naics_levels=naics_idx["code_levels"][take, :],
        zip_levels=zip_idx["code_levels"][take, :],
        naics_group_counts=naics_idx["group_counts"],
        zip_group_counts=zip_idx["group_counts"],
        target_accept=0.9,
    )
    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=1,
            random_seed=123,
            progressbar=False,
        )
    return idata


@pytest.mark.parametrize("n", [0, 1, 500])
def test_generate_synthetic_data_row_count(n):
    df = generate_synthetic_data(
        n=n, naics_codes=["511110"], zip_codes=["30309"], base_logit=-1.2, seed=42
    )
    assert len(df) == n


def test_naics_code_levels_shape_matches_data_length(hierarchical_indices):
    df, naics_idx, _, _, _ = hierarchical_indices
    assert naics_idx["code_levels"].shape[0] == len(df)


def test_zip_code_levels_shape_matches_data_length(hierarchical_indices):
    df, _, zip_idx, _, _ = hierarchical_indices
    assert zip_idx["code_levels"].shape[0] == len(df)


def test_naics_group_counts_length_matches_cut_points(hierarchical_indices):
    _, naics_idx, _, naics_cuts, _ = hierarchical_indices
    assert len(naics_idx["group_counts"]) == len(naics_cuts)


def test_zip_group_counts_length_matches_cut_points(hierarchical_indices):
    _, _, zip_idx, _, zip_cuts = hierarchical_indices
    assert len(zip_idx["group_counts"]) == len(zip_cuts)


def test_posterior_group_exists_in_sampled_model(sampled_model):
    assert "posterior" in sampled_model.groups()


@pytest.mark.parametrize("key", ["beta0", "eta", "p"])
def test_posterior_contains_expected_random_variables(sampled_model, key):
    assert key in sampled_model.posterior or key in sampled_model.posterior.coords  # type: ignore[attr-defined]
