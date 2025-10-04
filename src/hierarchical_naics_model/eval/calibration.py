from __future__ import annotations

from typing import Dict

import polars as pl


__all__ = ["calibration_report"]


def calibration_report(
    y_true,
    p_hat,
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
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    y = pl.Series("y", y_true)
    p = pl.Series("p", p_hat)

    if y.dtype.base_type() == pl.List or p.dtype.base_type() == pl.List:
        raise ValueError("y_true and p_hat must be 1-D sequences.")

    if y.len() != p.len():
        raise ValueError("y_true and p_hat must be the same length.")

    if y.len() == 0:
        empty = pl.DataFrame(
            {
                "bin_low": [],
                "bin_high": [],
                "n": [],
                "mean_p": [],
                "mean_y": [],
                "gap": [],
            }
        )
        return {
            "reliability": empty,
            "ece": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
        }

    df = pl.DataFrame({"y": y.cast(pl.Float64), "p": p.cast(pl.Float64)})
    eps = 1e-15
    bins_float = float(bins)
    df = df.with_columns(pl.col("p").clip(eps, 1.0 - eps).alias("p_clip"))
    df = df.with_columns(
        ((pl.col("p_clip") * bins_float).floor().clip(None, bins - 1))
        .cast(pl.Int64)
        .alias("bin")
    )

    agg = df.group_by("bin").agg(
        pl.len().alias("n"),
        pl.col("p_clip").mean().alias("mean_p"),
        pl.col("y").mean().alias("mean_y"),
    )

    base = pl.DataFrame(
        {
            "bin": list(range(bins)),
            "bin_low": [i / bins_float for i in range(bins)],
            "bin_high": [(i + 1) / bins_float for i in range(bins)],
        }
    )

    summary = (
        base.join(agg, on="bin", how="left")
        .with_columns(
            pl.col("n").fill_null(0).cast(pl.Int64),
            pl.col("mean_p").fill_null(0.0),
            pl.col("mean_y").fill_null(0.0),
        )
        .with_columns((pl.col("mean_p") - pl.col("mean_y")).alias("gap"))
    )

    N = float(df.height)
    ece = summary.select(
        ((pl.col("n").cast(pl.Float64) / N) * pl.col("gap").abs()).sum().alias("ece")
    ).item()

    brier = df.select(((pl.col("p") - pl.col("y")) ** 2).mean().alias("brier")).item()
    log_loss = df.select(
        (
            -(
                pl.col("y") * pl.col("p_clip").log()
                + (1.0 - pl.col("y")) * (1.0 - pl.col("p_clip")).log()
            )
        )
        .mean()
        .alias("log_loss")
    ).item()

    reliability = summary.select(
        ["bin_low", "bin_high", "n", "mean_p", "mean_y", "gap"]
    )

    return {
        "reliability": reliability,
        "ece": float(ece),
        "brier": float(brier),
        "log_loss": float(log_loss),
    }
