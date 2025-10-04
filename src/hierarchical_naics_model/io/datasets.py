from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_parquet(path: str, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """Load a Parquet file into a DataFrame.

    Parameters
    ----------
    path
        File path.
    columns
        Optional column subset.
    """

    file_path = Path(path)
    if not file_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    lf = pl.scan_parquet(file_path)
    if columns:
        return lf.select(list(columns))
    return lf


def save_parquet(df: pl.DataFrame | pl.LazyFrame, path: str) -> None:
    """Persist a DataFrame to Parquet, creating parent directories if needed."""

    file_path = Path(path)
    _ensure_parent(file_path)
    if isinstance(df, pl.LazyFrame):
        df.collect().write_parquet(file_path)
    else:
        df.write_parquet(file_path)
