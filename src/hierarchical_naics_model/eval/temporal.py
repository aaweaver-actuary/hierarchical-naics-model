from __future__ import annotations

from datetime import datetime
from typing import Tuple, cast

import polars as pl


def _parse_cutoff(cutoff_inclusive: str) -> datetime:
    text = cutoff_inclusive.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Could not parse cutoff '{cutoff_inclusive}' as an ISO datetime."
        ) from exc


def _date_expression(date_col: str, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.Datetime:
        return pl.col(date_col)
    if dtype == pl.Date:
        return pl.col(date_col).cast(pl.Datetime)
    return pl.col(date_col).str.strptime(pl.Datetime, strict=False)


def temporal_split(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    date_col: str,
    cutoff_inclusive: str,
) -> Tuple[pl.LazyFrame | pl.DataFrame, pl.LazyFrame | pl.DataFrame]:
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
    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        schema = df.collect_schema()
        columns = list(schema.keys())
        lf = df
    else:
        schema = df.schema
        columns = df.columns
        lf = df.lazy()

    if date_col not in columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    dtype = schema[date_col]
    date_expr = _date_expression(date_col, dtype)
    cutoff_dt = _parse_cutoff(cutoff_inclusive)
    cutoff_lit = pl.lit(cutoff_dt)

    train_lf = lf.filter(date_expr <= cutoff_lit)
    valid_lf = lf.filter(date_expr > cutoff_lit)

    if is_lazy:
        return train_lf, valid_lf
    train_lazy = cast(pl.LazyFrame, train_lf)
    valid_lazy = cast(pl.LazyFrame, valid_lf)
    return cast(pl.DataFrame, train_lazy.collect()), cast(
        pl.DataFrame, valid_lazy.collect()
    )
