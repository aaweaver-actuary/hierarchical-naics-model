from __future__ import annotations

import math
from typing import Iterable, List, Tuple, TypedDict

import polars as pl


class RankingReport(TypedDict):
    summary: pl.DataFrame
    base_rate: float


__all__ = ["ranking_report"]


def ranking_report(
    y_true,
    p_hat,
    *,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> RankingReport:
    """
    Compute precision@k% and lift@k% and cumulative gains.

    Parameters
    ----------
    y_true
        Binary labels (N,).
    p_hat
        Probabilities (N,).
    ks
        Percent thresholds (0..100). Non-monotone input is allowed.

    Returns
    -------
    dict
        {
          "summary": pl.LazyFrame[k_pct, k_count, precision, lift, cum_gain],
          "base_rate": float,
        }
    """
    y = pl.Series("y", y_true)
    p = pl.Series("p", p_hat)

    if y.dtype.base_type() == pl.List or p.dtype.base_type() == pl.List:
        raise ValueError("y_true and p_hat must be 1-D sequences.")
    if y.len() != p.len():
        raise ValueError("y_true and p_hat must have the same length.")

    N = int(y.len())
    if N == 0:
        return {
            "summary": pl.DataFrame(
                {
                    "k_pct": [],
                    "k_count": [],
                    "precision": [],
                    "lift": [],
                    "cum_gain": [],
                }
            ),
            "base_rate": float("nan"),
        }

    df = pl.DataFrame({"y": y.cast(pl.Float64), "p": p.cast(pl.Float64)})
    df = df.sort("p", descending=True, maintain_order=True)
    df = df.with_columns(pl.col("y").cum_sum().alias("cum_hits"))

    total_hits = float(df.select(pl.col("y").sum()).item())
    base_rate = float(df.select(pl.col("y").mean()).item())

    rows: List[Tuple[int, int, float, float, float]] = []
    for k in ks:
        k_pct = int(k)
        if k_pct <= 0:
            k_count = 0
        else:
            k_count = max(1, min(N, math.ceil(N * (k_pct / 100.0))))

        if k_count == 0:
            precision = 0.0
            lift = 0.0
            cum_gain = 0.0
        else:
            hits = float(df["cum_hits"][k_count - 1])
            precision = hits / k_count
            if base_rate > 0:
                lift = precision / base_rate
            else:
                lift = math.inf if hits > 0 else 0.0
            denom = total_hits if total_hits > 0 else 1.0
            cum_gain = hits / denom
        rows.append((k_pct, k_count, precision, lift, cum_gain))

    summary = pl.DataFrame(
        rows,
        schema=[
            ("k_pct", pl.Int64),
            ("k_count", pl.Int64),
            ("precision", pl.Float64),
            ("lift", pl.Float64),
            ("cum_gain", pl.Float64),
        ],
        orient="row",
    )
    return {"summary": summary, "base_rate": float(base_rate)}
