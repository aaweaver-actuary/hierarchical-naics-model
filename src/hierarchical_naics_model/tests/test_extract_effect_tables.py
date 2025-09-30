from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from typing import List, cast

from hierarchical_naics_model.extract_effect_tables import extract_effect_tables
import pytest


class _FakeIdata:
    def __init__(self, posterior: xr.Dataset) -> None:
        self.posterior = posterior


def _make_posterior(with_zip: bool = True) -> xr.Dataset:
    # dims
    chain, draw = 2, 3
    n_naics0, n_naics1 = 3, 2
    n_zip0 = 2
    coords = {
        "chain": np.arange(chain),
        "draw": np.arange(draw),
        "NAICS_0": np.arange(n_naics0),
        "NAICS_1": np.arange(n_naics1),
    }
    if with_zip:
        coords["ZIP_0"] = np.arange(n_zip0)

    # base vectors with known means
    beta0_val = 1.23
    eff_naics0 = np.array([0.1, -0.2, 0.4])
    eff_naics1 = np.array([-0.3, 0.7])
    eff_zip0 = np.array([0.05, -0.05])

    data_vars: dict = {
        "beta0": (("chain", "draw"), np.full((chain, draw), beta0_val)),
        # Intentionally provide level 1 before 0 to ensure numeric sort works
        "naics_eff_1": (
            ("chain", "draw", "NAICS_1"),
            np.tile(eff_naics1, (chain, draw, 1)),
        ),
        "naics_eff_0": (
            ("chain", "draw", "NAICS_0"),
            np.tile(eff_naics0, (chain, draw, 1)),
        ),
    }
    if with_zip:
        data_vars["zip_eff_0"] = (
            ("chain", "draw", "ZIP_0"),
            np.tile(eff_zip0, (chain, draw, 1)),
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture
def posterior_with_zip():
    return _make_posterior(with_zip=True)


@pytest.fixture
def posterior_without_zip():
    return _make_posterior(with_zip=False)


@pytest.fixture
def idata_with_zip(posterior_with_zip):
    return _FakeIdata(posterior_with_zip)


@pytest.fixture
def idata_without_zip(posterior_without_zip):
    return _FakeIdata(posterior_without_zip)


@pytest.fixture
def extracted_with_zip(idata_with_zip):
    return extract_effect_tables(idata_with_zip)


@pytest.fixture
def extracted_without_zip(idata_without_zip):
    return extract_effect_tables(idata_without_zip)


def test_extract_effect_tables_beta0_mean_is_correct(extracted_with_zip):
    assert np.isclose(extracted_with_zip["beta0"], 1.23)


@pytest.mark.parametrize(
    "level,expected_index,expected_values",
    [
        (0, [0, 1, 2], np.array([0.1, -0.2, 0.4])),
        (1, [0, 1], np.array([-0.3, 0.7])),
    ],
)
def test_extract_effect_tables_naics_table_indices_and_values_are_correct(
    level, expected_index, expected_values, extracted_with_zip
):
    naics_tables = cast(List[pd.Series], extracted_with_zip["naics_tables"])
    assert naics_tables[level].index.tolist() == expected_index
    np.testing.assert_allclose(naics_tables[level].values, expected_values)


def test_extract_effect_tables_zip_table_values_are_correct(extracted_with_zip):
    zip_tables = cast(List[pd.Series], extracted_with_zip["zip_tables"])
    np.testing.assert_allclose(zip_tables[0].values, np.array([0.05, -0.05]))


def test_extract_effect_tables_naics_level_names_are_default(extracted_with_zip):
    assert extracted_with_zip["naics_level_names"] == ["NAICS_L0", "NAICS_L1"]


def test_extract_effect_tables_zip_level_names_are_default(extracted_with_zip):
    assert extracted_with_zip["zip_level_names"] == ["ZIP_L0"]


def test_extract_effect_tables_naics_tables_length_with_zip_is_two(extracted_with_zip):
    naics_tables = cast(List[pd.Series], extracted_with_zip["naics_tables"])
    assert len(naics_tables) == 2


def test_extract_effect_tables_zip_tables_length_with_zip_is_one(extracted_with_zip):
    zip_tables = cast(List[pd.Series], extracted_with_zip["zip_tables"])
    assert len(zip_tables) == 1


def test_extract_effect_tables_naics_tables_length_without_zip_is_two(
    extracted_without_zip,
):
    naics_tables = cast(List[pd.Series], extracted_without_zip["naics_tables"])
    assert len(naics_tables) == 2


def test_extract_effect_tables_zip_tables_length_without_zip_is_zero(
    extracted_without_zip,
):
    zip_tables = cast(List[pd.Series], extracted_without_zip["zip_tables"])
    assert len(zip_tables) == 0


def test_extract_effect_tables_zip_level_names_without_zip_is_empty(
    extracted_without_zip,
):
    assert extracted_without_zip["zip_level_names"] == []


@pytest.fixture
def custom_level_names():
    return {
        "naics_level_names": ["N0", "N1"],
        "zip_level_names": ["Z0"],
    }


@pytest.fixture
def extracted_with_custom_names(custom_level_names):
    posterior = _make_posterior(with_zip=True)
    idata = _FakeIdata(posterior)
    out = extract_effect_tables(
        idata,
        naics_level_names=custom_level_names["naics_level_names"],
        zip_level_names=custom_level_names["zip_level_names"],
    )
    return out, custom_level_names


@pytest.mark.parametrize(
    "level,expected_name",
    [
        (0, "N0"),
        (1, "N1"),
    ],
)
def test_extract_effect_tables_naics_custom_level_names_are_correct(
    extracted_with_custom_names, level, expected_name
):
    out, custom_level_names = extracted_with_custom_names
    assert out["naics_level_names"][level] == expected_name


def test_extract_effect_tables_zip_custom_level_name_is_correct(
    extracted_with_custom_names,
):
    out, custom_level_names = extracted_with_custom_names
    assert out["zip_level_names"][0] == custom_level_names["zip_level_names"][0]


@pytest.fixture
def extracted_no_zip_level():
    posterior = _make_posterior(with_zip=False)
    idata = _FakeIdata(posterior)
    return extract_effect_tables(idata)


def test_extract_effect_tables_no_zip_level_naics_tables_length_is_two(
    extracted_no_zip_level,
):
    naics_tables = cast(List[pd.Series], extracted_no_zip_level["naics_tables"])
    assert len(naics_tables) == 2


def test_extract_effect_tables_no_zip_level_zip_tables_length_is_zero(
    extracted_no_zip_level,
):
    zip_tables = cast(List[pd.Series], extracted_no_zip_level["zip_tables"])
    assert len(zip_tables) == 0


def test_extract_effect_tables_no_zip_level_zip_level_names_is_empty(
    extracted_no_zip_level,
):
    assert extracted_no_zip_level["zip_level_names"] == []


@pytest.mark.parametrize(
    "with_zip,expected_naics_len,expected_zip_len,expected_zip_names",
    [
        (True, 2, 1, ["ZIP_L0"]),
        (False, 2, 0, []),
    ],
)
def test_extract_effect_tables_table_lengths_and_zip_level_names_are_correct(
    with_zip, expected_naics_len, expected_zip_len, expected_zip_names
):
    posterior = _make_posterior(with_zip=with_zip)
    idata = _FakeIdata(posterior)
    out = extract_effect_tables(idata)
    naics_tables = cast(List[pd.Series], out["naics_tables"])
    zip_tables = cast(List[pd.Series], out["zip_tables"])
    assert len(naics_tables) == expected_naics_len
    assert len(zip_tables) == expected_zip_len
    assert out["zip_level_names"] == expected_zip_names


def test_extract_effect_tables_empty_posterior():
    import xarray as xr

    idata = _FakeIdata(xr.Dataset())
    out = extract_effect_tables(idata)
    assert out["naics_tables"] == []
    assert out["zip_tables"] == []
    assert out["naics_level_names"] == []
    assert out["zip_level_names"] == []


def test_extract_effect_tables_missing_expected_vars():
    import xarray as xr

    # Only provide beta0, no naics/zip effects
    ds = xr.Dataset({"beta0": ("chain", np.ones(2))})
    idata = _FakeIdata(ds)
    out = extract_effect_tables(idata)
    assert out["beta0"] == 1.0
    assert out["naics_tables"] == []
    assert out["zip_tables"] == []


def test_extract_effect_tables_custom_level_names_with_missing_tables():
    import xarray as xr

    ds = xr.Dataset({"beta0": ("chain", np.ones(2))})
    idata = _FakeIdata(ds)
    custom_naics = ["CustomN0", "CustomN1"]
    custom_zip = ["CustomZ0"]
    out = extract_effect_tables(
        idata, naics_level_names=custom_naics, zip_level_names=custom_zip
    )
    assert out["naics_level_names"] == custom_naics
    assert out["zip_level_names"] == custom_zip
