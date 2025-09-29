from __future__ import annotations
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd


def calibration_and_lift_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    bins: int = 10,
    ks: Iterable[int] = (1, 5, 10, 20, 30),  # percent thresholds
) -> Dict[str, object]:
    """
    Compute reliability (calibration), Brier/log-loss, and ranking lift metrics.

    Parameters
    ----------
    y_true
        Ground-truth outcomes in {0,1}, shape (N,).
    p_hat
        Predicted probabilities in [0,1], shape (N,).
    bins
        Number of equally-sized probability bins for reliability/ECE.
    ks
        Percent cutoffs for precision@k and lift@k.

    Returns
    -------
    out : dict
        - 'summary' : dict with:
            'n', 'base_rate', 'brier', 'log_loss', 'ece'
        - 'reliability' : DataFrame with columns:
            ['bin', 'count', 'p_avg', 'y_rate', 'abs_gap', 'squared_gap']
        - 'ranking' : DataFrame with columns:
            ['k_pct','k_count','pos_in_top','precision_at_k','lift_at_k','cum_gain_pct']
        - 'sorted_scores' : DataFrame with ['y_true','p_hat'] sorted desc by p_hat

    Notes
    -----
    - ECE (Expected Calibration Error) is computed as sum over bins of
      (count/N) * |y_rate - p_avg|.
    - 'lift_at_k' compares precision@k to the global base rate.
    - 'cum_gain_pct' is the cumulative capture of positives vs. all positives.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_hat, dtype=float)
    assert y.shape == p.shape

    n = y.size
    base_rate = float(y.mean())

    # Brier and log-loss (clip for stability)
    eps = 1e-12
    brier = float(np.mean((y - p) ** 2))
    log_loss = float(
        -np.mean(
            y * np.log(np.clip(p, eps, 1.0))
            + (1 - y) * np.log(np.clip(1 - p, eps, 1.0 - eps))
        )
    )

    # Reliability / ECE
    # Bin by predicted probability into equal-width bins in [0,1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)

    df_rel = (
        pd.DataFrame({"bin": bin_idx, "p": p, "y": y})
        .groupby("bin")
        .agg(count=("y", "size"), p_avg=("p", "mean"), y_rate=("y", "mean"))
        .reset_index()
        .sort_values("bin")
    )
    df_rel["abs_gap"] = (df_rel["y_rate"] - df_rel["p_avg"]).abs()
    df_rel["squared_gap"] = (df_rel["y_rate"] - df_rel["p_avg"]) ** 2

    ece = float((df_rel["count"] / n * df_rel["abs_gap"]).sum())

    # Ranking (lift/gain)
    order = np.argsort(-p)  # descending by score
    y_sorted = y[order]
    p_sorted = p[order]
    sorted_scores = pd.DataFrame({"y_true": y_sorted, "p_hat": p_sorted})

    pos_total = int(y.sum())
    ranking_rows: List[Dict[str, float]] = []
    for k in ks:
        k = int(k)
        k_count = max(1, int(np.ceil(k / 100.0 * n)))
        pos_in_top = int(y_sorted[:k_count].sum())
        precision_at_k = pos_in_top / k_count
        lift_at_k = precision_at_k / base_rate if base_rate > 0 else np.nan
        cum_gain_pct = pos_in_top / pos_total if pos_total > 0 else np.nan
        ranking_rows.append(
            dict(
                k_pct=k,
                k_count=k_count,
                pos_in_top=pos_in_top,
                precision_at_k=precision_at_k,
                lift_at_k=lift_at_k,
                cum_gain_pct=cum_gain_pct,
            )
        )
    df_rank = pd.DataFrame(ranking_rows)

    return {
        "summary": dict(
            n=int(n),
            base_rate=base_rate,
            brier=brier,
            log_loss=log_loss,
            ece=ece,
        ),
        "reliability": df_rel,
        "ranking": df_rank,
        "sorted_scores": sorted_scores,
    }
