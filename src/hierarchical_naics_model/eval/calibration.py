from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


__all__ = ["calibration_report"]


def calibration_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    bins: int = 10,
) -> Dict[str, object]:
    """
    Reliability curve + standard calibration metrics.

    Metrics
    -------
    - reliability: DataFrame with columns
      [bin_low, bin_high, n, mean_p, mean_y, gap]
    - ece: Expected Calibration Error (weighted |gap|)
    - brier: mean squared error of probabilities
    - log_loss: cross-entropy with clipping

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1), shape (N,).
    p_hat : array-like
        Predicted probabilities in [0,1], shape (N,).
    bins : int
        Number of equal-width probability bins.

    Returns
    -------
    dict
        {"reliability": DataFrame, "ece": float, "brier": float, "log_loss": float}
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(p_hat).astype(float)

    if y.ndim != 1 or p.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("y_true and p_hat must be 1-D arrays of the same length.")

    N = y.size
    if N == 0:
        empty = pd.DataFrame(
            columns=["bin_low", "bin_high", "n", "mean_p", "mean_y", "gap"]
        )
        return {
            "reliability": empty,
            "ece": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
        }

    # Stability for log-loss
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)

    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-inclusive last bin
    bin_idx = np.searchsorted(edges, p_clip, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    rows = []
    ece = 0.0
    for b in range(bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        low, high = edges[b], edges[b + 1]
        if n_b == 0:
            mean_p = mean_y = gap = 0.0
        else:
            mean_p = float(p_clip[mask].mean())
            mean_y = float(y[mask].mean())
            gap = mean_p - mean_y
            ece += (n_b / N) * abs(gap)
        rows.append((low, high, n_b, mean_p, mean_y, gap))

    reliability = pd.DataFrame(
        rows, columns=["bin_low", "bin_high", "n", "mean_p", "mean_y", "gap"]
    )
    brier = float(((p - y) ** 2).mean())
    log_loss = float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean())

    return {
        "reliability": reliability,
        "ece": float(ece),
        "brier": brier,
        "log_loss": log_loss,
    }
