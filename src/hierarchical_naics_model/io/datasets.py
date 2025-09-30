from __future__ import annotations

from typing import Sequence
import pandas as pd


def load_parquet(path: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.

    Parameters
    ----------
    path
        File path.
    columns
        Optional column subset.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    # TODO: implement (pd.read_parquet with columns).
    raise NotImplementedError


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to Parquet.

    Parameters
    ----------
    df
        DataFrame to persist.
    path
        Destination path.
    """
    # TODO: implement (df.to_parquet).
    raise NotImplementedError
