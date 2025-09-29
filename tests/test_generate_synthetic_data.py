from __future__ import annotations
import pandas as pd

from hierarchical_naics_model.generate_synthetic_data import generate_synthetic_data


def test_generate_basic_shape_and_columns(naics_pool, zip_pool):
    df = generate_synthetic_data(
        200, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-1.5, seed=0
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"is_written", "naics", "zip"}
    assert len(df) == 200
    # value ranges
    assert df["is_written"].isin([0, 1]).all()
    assert df["naics"].isin(naics_pool).all()
    assert df["zip"].isin(zip_pool).all()


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
