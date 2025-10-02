# tests/unit/test_scoring_extract_stub.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hierarchical_naics_model.scoring.extract import extract_effect_tables_nested

az = pytest.importorskip("arviz")


@pytest.fixture
def stub_idata():
    posterior = {
        "beta0": np.array([[0.1, 0.3]]),
        "naics_base": np.array([[[0.0, 0.2], [0.2, 0.4]]]),
        "zip_base": np.array([[[0.5, 0.7, 0.9], [0.7, 0.9, 1.1]]]),
        "naics_delta_1": np.array([[[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]]),
        "zip_delta_1": np.array([[[0.0, -0.1], [0.1, 0.0]]]),
    }
    coords = {
        "naics_g0": np.arange(2),
        "zip_g0": np.arange(3),
        "naics_g1": np.arange(3),
        "zip_g1": np.arange(2),
    }
    dims = {
        "naics_base": ["chain", "draw", "naics_g0"],
        "zip_base": ["chain", "draw", "zip_g0"],
        "naics_delta_1": ["chain", "draw", "naics_g1"],
        "zip_delta_1": ["chain", "draw", "zip_g1"],
        "beta0": ["chain", "draw"],
    }
    return az.from_dict(posterior=posterior, coords=coords, dims=dims)


@pytest.fixture
def extracted_effects(stub_idata):
    return extract_effect_tables_nested(stub_idata)


def test_extract_effect_tables_nested_returns_dict(extracted_effects):
    assert isinstance(extracted_effects, dict)


def test_extract_effect_tables_nested_keys_are_correct(extracted_effects):
    expected_keys = {"beta0", "naics_base", "naics_deltas", "zip_base", "zip_deltas"}
    assert set(extracted_effects.keys()) == expected_keys


def test_extract_effect_tables_nested_naics_base_is_series(extracted_effects):
    assert isinstance(extracted_effects["naics_base"], pd.Series)


def test_extract_effect_tables_nested_zip_base_is_series(extracted_effects):
    assert isinstance(extracted_effects["zip_base"], pd.Series)


def test_extract_effect_tables_nested_naics_deltas_is_list_of_length_1(
    extracted_effects,
):
    assert (
        isinstance(extracted_effects["naics_deltas"], list)
        and len(extracted_effects["naics_deltas"]) == 1
    )


def test_extract_effect_tables_nested_zip_deltas_is_list_of_length_1(extracted_effects):
    assert (
        isinstance(extracted_effects["zip_deltas"], list)
        and len(extracted_effects["zip_deltas"]) == 1
    )


def test_extract_effect_tables_nested_beta0_mean_is_correct(extracted_effects):
    assert abs(extracted_effects["beta0"] - 0.2) < 1e-12


def test_extract_effect_tables_nested_naics_base_mean_is_correct(extracted_effects):
    np.testing.assert_allclose(
        extracted_effects["naics_base"].values, np.array([0.1, 0.3])
    )


def test_extract_effect_tables_nested_zip_base_mean_is_correct(extracted_effects):
    np.testing.assert_allclose(
        extracted_effects["zip_base"].values, np.array([0.6, 0.8, 1.0])
    )


@pytest.mark.parametrize(
    "posterior,expected_beta0",
    [
        ({"beta0": np.array([[0.0, 0.0]])}, 0.0),
        ({"beta0": np.array([[1.0, 1.0]])}, 1.0),
        ({"beta0": np.array([[0.5, 1.5]])}, 1.0),
    ],
)
def test_extract_effect_tables_nested_beta0_parametrized(posterior, expected_beta0):
    coords = {}
    dims = {"beta0": ["chain", "draw"]}
    idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
    if "naics_base" not in posterior or "zip_base" not in posterior:
        with pytest.raises(ValueError, match="missing variable 'naics_base'"):
            extract_effect_tables_nested(idata)
    else:
        eff = extract_effect_tables_nested(idata)
        assert abs(eff["beta0"] - expected_beta0) < 1e-12


@pytest.mark.parametrize(
    "naics_base,expected",
    [
        (np.array([[[0.0, 0.0], [0.0, 0.0]]]), np.array([0.0, 0.0])),
        (np.array([[[1.0, 2.0], [3.0, 4.0]]]), np.array([2.0, 3.0])),
    ],
)
def test_extract_effect_tables_nested_naics_base_parametrized(naics_base, expected):
    posterior = {"naics_base": naics_base, "beta0": np.array([[0.0, 0.0]])}
    coords = {"naics_g0": np.arange(naics_base.shape[2])}
    dims = {"naics_base": ["chain", "draw", "naics_g0"], "beta0": ["chain", "draw"]}
    idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
    if "zip_base" not in posterior:
        with pytest.raises(ValueError, match="missing variable 'zip_base'"):
            extract_effect_tables_nested(idata)
    else:
        eff = extract_effect_tables_nested(idata)
        np.testing.assert_allclose(eff["naics_base"].values, expected)
