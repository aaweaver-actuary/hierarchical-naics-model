from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import pymc as pm
except Exception:  # pragma: no cover - import guard for environments without PyMC
    pm = None  # type: ignore


def build_conversion_model_nested_deltas(
    *,
    y: np.ndarray,
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
    naics_group_counts: Sequence[int],
    zip_group_counts: Sequence[int],
    target_accept: float = 0.92,
    use_student_t_level0: bool = False,
):
    """
    Build a PyMC logistic model with base + nested deltas for NAICS and ZIP.

    Parameters
    ----------
    y
        Binary outcomes (N,).
    naics_levels
        Integer indices (N, J) across NAICS hierarchy levels.
    zip_levels
        Integer indices (N, M) across ZIP hierarchy levels.
    naics_group_counts
        Group counts per NAICS level (len J).
    zip_group_counts
        Group counts per ZIP level (len M).
    target_accept
        NUTS target_accept for sampling (model.default_sampling_kwargs).
    use_student_t_level0
        If True, Level-0 base means use StudentT prior; otherwise Normal.

    Returns
    -------
    pm.Model
        Constructed model with deterministics `eta`, `p` and likelihood `is_written`.

    Notes
    -----
    - Deeper deltas are zero-mean with depth-shrinking scales.
    - Non-centered parameterization is recommended for all random vectors.
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")

    # TODO: implement
    # - pm.Data for y and per-level indices
    # - Priors: beta0, base means (Normal or StudentT), half-normal scales per level
    # - Random vectors: naics_base (level 0), naics_delta_{j>=1}, similarly for ZIP
    # - eta = beta0 + base[idx0] + sum(delta[idxj])
    # - p = sigmoid(eta); Bernoulli likelihood
    # - model.default_sampling_kwargs set with target_accept
    raise NotImplementedError
