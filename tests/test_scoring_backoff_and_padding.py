# tests/test_scoring_backoff_and_padding.py
import numpy as np
import pandas as pd

from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.predict_proba import predict_proba
import pytest


@pytest.fixture
def train_data():
    train_naics = ["511110", "511120", "522110"]
    train_zip = ["02139", "30309", "45040"]
    idx_n = build_hierarchical_indices(
        train_naics, cut_points=[2, 3, 6], prefix_fill="0"
    )
    idx_z = build_hierarchical_indices(train_zip, cut_points=[2, 3, 5], prefix_fill="0")
    effects = {
        "beta0": -1.0,
        # Provide both possible indices for base effects to match backoff resolution
        "naics_base": pd.Series([0.2, 0.0], index=[0, 1]),
        "naics_deltas": [
            pd.Series([0.05, -0.05], index=[0, 1]),
            pd.Series([0.10, 0.00, -0.10], index=[0, 1, 2]),
        ],
        "zip_base": pd.Series([0.1, 0.0], index=[0, 1]),
        "zip_deltas": [
            pd.Series([0.02, -0.02, 0.0], index=[0, 1, 2]),
            pd.Series([0.04, -0.03, 0.01], index=[0, 1, 2]),
        ],
    }
    return idx_n, idx_z, effects


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "naics": ["511130", "522120", "521000"],
            "zip": ["02138", "30310", "02"],
        }
    )


@pytest.fixture
def scored(train_data, test_df):
    idx_n, idx_z, effects = train_data
    return predict_proba(
        df_new=test_df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[2, 3, 6],
        zip_cut_points=[2, 3, 5],
        naics_level_maps=idx_n["maps"],
        zip_level_maps=idx_z["maps"],
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )


@pytest.mark.parametrize("col", ["p", "eta"])
def test_scored_contains_probability_and_eta_columns(scored, col):
    assert col in scored


@pytest.mark.parametrize("j", range(3))
def test_backoff_naics_flags_are_boolean(scored, j):
    assert scored[f"backoff_naics_{j}"].dtype == bool


@pytest.mark.parametrize("j", range(3))
def test_backoff_zip_flags_are_boolean(scored, j):
    assert scored[f"backoff_zip_{j}"].dtype == bool


def test_probability_is_finite_for_unseen_leaf_codes(scored):
    assert np.isfinite(scored["p"]).all()


def test_probability_is_strictly_between_zero_and_one_for_all_rows(scored):
    assert ((scored["p"] > 0) & (scored["p"] < 1)).all()


@pytest.mark.parametrize(
    "naics,zip",
    [
        ("511130", "02138"),  # unseen leaf for both
        ("522120", "30310"),  # unseen leaf for both
        ("521000", "02"),  # unseen parent for naics, unseen parent for zip
    ],
)
def test_model_handles_unseen_codes_and_backoff_produces_valid_probability(
    train_data, naics, zip
):
    idx_n, idx_z, effects = train_data
    df = pd.DataFrame({"naics": [naics], "zip": [zip]})
    scored = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[2, 3, 6],
        zip_cut_points=[2, 3, 5],
        naics_level_maps=idx_n["maps"],
        zip_level_maps=idx_z["maps"],
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert ((scored["p"] > 0) & (scored["p"] < 1)).all()
