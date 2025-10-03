from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_parquet(path: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
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
    return pd.read_parquet(file_path, columns=list(columns) if columns else None)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a DataFrame to Parquet, creating parent directories if needed."""

    file_path = Path(path)
    _ensure_parent(file_path)
    df.to_parquet(file_path, index=False)
