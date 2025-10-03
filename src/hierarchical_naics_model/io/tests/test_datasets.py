from __future__ import annotations

import pandas as pd
import pytest

from hierarchical_naics_model.io.datasets import load_parquet, save_parquet


def test_save_and_load_parquet(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    path = tmp_path / "nested" / "data.parquet"
    save_parquet(df, path)
    loaded = load_parquet(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_load_parquet_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "missing.parquet")
