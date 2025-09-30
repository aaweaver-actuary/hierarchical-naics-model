from __future__ import annotations

from typing import Dict, Iterable
import numpy as np


def ranking_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    ks: Iterable[int] = (5, 10, 20, 30),
) -> Dict[str, object]:
    """
    Compute precision@k%, lift@k%, and cumulative gains.

    Parameters
    ----------
    y_true
        Binary labels (N,).
    p_hat
        Predicted probabilities (N,).
    ks
        Percent thresholds for top-k (as whole percents 0-100).

    Returns
    -------
    dict
        {
          "summary": pd.DataFrame[k_pct, k_count, precision, lift, cum_gain],
          "base_rate": float,
        }
    """
    # TODO: implement (sort by p_hat desc; compute metrics at each k).
    raise NotImplementedError
