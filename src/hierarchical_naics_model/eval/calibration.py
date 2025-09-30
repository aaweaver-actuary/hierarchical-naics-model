from __future__ import annotations

from typing import Dict
import numpy as np


def calibration_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    bins: int = 10,
) -> Dict[str, object]:
    """
    Compute reliability curve and calibration metrics.

    Parameters
    ----------
    y_true
        Binary labels (N,).
    p_hat
        Predicted probabilities (N,).
    bins
        Number of probability bins for reliability curve.

    Returns
    -------
    dict
        {
          "reliability": pd.DataFrame[bin_low, bin_high, n, mean_p, mean_y, gap],
          "ece": float,
          "brier": float,
          "log_loss": float,
        }
    """
    # TODO: implement (bin by p_hat; compute mean_p, mean_y; ECE; Brier; log-loss).
    raise NotImplementedError
