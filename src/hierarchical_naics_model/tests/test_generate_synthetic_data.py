from __future__ import annotations
import pandas as pd

from hierarchical_naics_model.generate_synthetic_data import generate_synthetic_data
import pytest


@pytest.fixture
def synthetic_df(naics_pool, zip_pool):
    return generate_synthetic_data(
        200, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-1.5, seed=0
    )


def test_generated_dataframe_is_instance_of_pandas_dataframe(synthetic_df):
    assert isinstance(synthetic_df, pd.DataFrame)


def test_generated_dataframe_has_expected_columns(synthetic_df):
    assert set(synthetic_df.columns) == {"is_written", "naics", "zip"}


def test_generated_dataframe_has_expected_number_of_rows(synthetic_df):
    assert len(synthetic_df) == 200


@pytest.mark.parametrize("value", [0, 1])
def test_generated_dataframe_is_written_column_contains_only_0_or_1(
    synthetic_df, value
):
    assert value in synthetic_df["is_written"].unique()


@pytest.mark.parametrize("naics_code", ["111110", "112120"])
def test_generated_dataframe_naics_column_contains_naics_pool_values(
    synthetic_df, naics_code, naics_pool
):
    # Only test if naics_code is in the pool
    if naics_code in naics_pool:
        assert naics_code in synthetic_df["naics"].values


@pytest.mark.parametrize("zip_code", ["90210", "10001"])
def test_generated_dataframe_zip_column_contains_zip_pool_values(
    synthetic_df, zip_code, zip_pool
):
    # Only test if zip_code is in the pool
    if zip_code in zip_pool:
        assert zip_code in synthetic_df["zip"].values


@pytest.mark.parametrize("base_logit", [-10, 0, 10])
def test_generated_dataframe_handles_extreme_base_logit_values(
    naics_pool, zip_pool, base_logit
):
    df = generate_synthetic_data(
        200, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=base_logit, seed=0
    )
    assert isinstance(df, pd.DataFrame)


def test_reproducibility(naics_pool, zip_pool):
    df1 = generate_synthetic_data(
        300, naics_codes=naics_pool, zip_codes=zip_pool, seed=123
    )
    df2 = generate_synthetic_data(
        300, naics_codes=naics_pool, zip_codes=zip_pool, seed=123
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_base_logit_effect(naics_pool, zip_pool):
    df_low = generate_synthetic_data(
        5000, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-3.0, seed=1
    )
    df_high = generate_synthetic_data(
        5000, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=+3.0, seed=1
    )
    # With same seed and pools, higher base_logit should yield strictly higher mean
    assert df_high["is_written"].mean() > df_low["is_written"].mean()
