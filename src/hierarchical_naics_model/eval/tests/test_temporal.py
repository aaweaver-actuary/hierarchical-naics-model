from __future__ import annotations

import pandas as pd

from hierarchical_naics_model.eval.temporal import temporal_split


def test_temporal_split_basic():
    df = pd.DataFrame(
        {
            "quote_date": ["2024-01-01", "2024-01-10", "2024-02-01"],
            "value": [1, 2, 3],
        }
    )
    train, valid = temporal_split(
        df, date_col="quote_date", cutoff_inclusive="2024-01-10"
    )

    assert len(train) == 2
    assert len(valid) == 1
    assert train["value"].tolist() == [1, 2]
    assert valid["value"].tolist() == [3]
