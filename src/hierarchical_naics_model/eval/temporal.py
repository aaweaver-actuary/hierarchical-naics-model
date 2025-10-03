from __future__ import annotations

from typing import Tuple
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    *,
    date_col: str,
    cutoff_inclusive: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train/validation by date.

    Parameters
    ----------
    df
        Input DataFrame.
    date_col
        Column with date or datetime (string parseable).
    cutoff_inclusive
        ISO date/time string. Train includes rows with date <= cutoff.

    Returns
    -------
    (train_df, valid_df)
        Two DataFrames preserving original columns.
    """
    cutoff = pd.to_datetime(cutoff_inclusive)
    dates = pd.to_datetime(df[date_col])
    mask = dates <= cutoff
    train_df = df.loc[mask].copy()
    valid_df = df.loc[~mask].copy()
    return train_df, valid_df
