from __future__ import annotations

from typing import Iterable, List, Tuple, TypedDict
import numpy as np
import pandas as pd


class RankingReport(TypedDict):
    summary: pd.DataFrame
    base_rate: float


__all__ = ["ranking_report"]


def ranking_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
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
          "summary": pd.DataFrame[k_pct, k_count, precision, lift, cum_gain],
          "base_rate": float,
        }
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_hat).astype(float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and p_hat must be 1-D arrays of the same shape.")
    N = y.size
    if N == 0:
        return {
            "summary": pd.DataFrame(
                columns=["k_pct", "k_count", "precision", "lift", "cum_gain"]
            ),
            "base_rate": np.nan,
        }

    order = np.argsort(-p, kind="mergesort")  # stable
    y_sorted = y[order]

    base_rate = y.mean() if N > 0 else np.nan

    rows: List[Tuple[int, int, float, float, float]] = []
    for k_pct in ks:
        k_pct = int(k_pct)
        k_count = max(1, min(N, int(np.ceil(N * (k_pct / 100.0))))) if k_pct > 0 else 0
        if k_count == 0:
            precision = 0.0
            lift = 0.0 if base_rate == 0 else 0.0
            cum_gain = 0.0
        else:
            hits = y_sorted[:k_count].sum()
            precision = float(hits / k_count)
            lift = (
                float(precision / base_rate)
                if base_rate > 0
                else np.inf
                if hits > 0
                else 0.0
            )
            cum_gain = float(hits / (y.sum() if y.sum() > 0 else 1.0))
        rows.append((k_pct, k_count, precision, lift, cum_gain))

    summary = pd.DataFrame(
        rows, columns=["k_pct", "k_count", "precision", "lift", "cum_gain"]
    )
    return {"summary": summary, "base_rate": float(base_rate)}
