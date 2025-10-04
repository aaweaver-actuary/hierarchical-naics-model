from __future__ import annotations

import polars as pl

from hierarchical_naics_model.eval.temporal import temporal_split


def test_temporal_split_basic():
    df = pl.LazyFrame(
        {
            "quote_date": ["2024-01-01", "2024-01-10", "2024-02-01"],
            "value": [1, 2, 3],
        }
    )
    train, valid = temporal_split(
        df, date_col="quote_date", cutoff_inclusive="2024-01-10"
    )

    train_df = train.collect() if isinstance(train, pl.LazyFrame) else train
    valid_df = valid.collect() if isinstance(valid, pl.LazyFrame) else valid

    assert train_df.height == 2
    assert valid_df.height == 1
    assert train_df["value"].to_list() == [1, 2]
    assert valid_df["value"].to_list() == [3]


def test_temporal_split_with_dataframe():
    df = pl.DataFrame(
        {
            "quote_date": ["2024-01-01", "2024-01-10", "2024-02-01"],
            "value": [1, 2, 3],
        }
    )
    train_df, valid_df = temporal_split(
        df, date_col="quote_date", cutoff_inclusive="2024-01-05"
    )

    assert isinstance(train_df, pl.DataFrame)
    assert isinstance(valid_df, pl.DataFrame)
    assert train_df.height == 1
    assert valid_df.height == 2
