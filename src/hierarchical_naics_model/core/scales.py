from __future__ import annotations

import numpy as np


def exp_decay_sigmas(J: int, sigma0: float, kappa: float) -> np.ndarray:
    """
    Deterministic monotone depth scales: sigma_j = sigma0 * exp(-kappa * j).

    Parameters
    ----------
    J
        Number of levels (>=1).
    sigma0
        Base scale (>0).
    kappa
        Positive decay rate.

    Returns
    -------
    np.ndarray
        Vector (J,) of strictly decreasing positive scales.

    Notes
    -----
    - Pure function; use in tests and quick what-if analysis.
    """
    # TODO: implement
    raise NotImplementedError
