# nested_quotewrite/core/effects.py
from __future__ import annotations

from typing import List, Sequence
import numpy as np

from .validation import validate_level_indices


__all__ = ["eta_additive", "eta_nested"]


def _ensure_vector(name: str, arr: np.ndarray, length: int) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"`{name}` must be a numpy.ndarray; got {type(arr).__name__}")
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1-D; got shape {arr.shape}")
    if arr.shape[0] != length:
        raise ValueError(
            f"`{name}` length mismatch: expected {length}, got {arr.shape[0]}."
        )
    if not np.issubdtype(arr.dtype, np.floating):
        # Allow integers but cast can be done by caller; be strict here.
        raise ValueError(f"`{name}` dtype must be float; got {arr.dtype}")


def _check_effects_list(
    name: str, effects: Sequence[np.ndarray], expected_L: int
) -> None:
    if len(effects) != expected_L:
        raise ValueError(
            f"`{name}` length ({len(effects)}) must match number of levels ({expected_L})."
        )
    for j, vec in enumerate(effects):
        if not isinstance(vec, np.ndarray):
            raise ValueError(
                f"`{name}[{j}]` must be a numpy.ndarray; got {type(vec).__name__}"
            )
        if vec.ndim != 1:
            raise ValueError(f"`{name}[{j}]` must be 1-D; got shape {vec.shape}")
        if not np.issubdtype(vec.dtype, np.floating):
            raise ValueError(f"`{name}[{j}]` dtype must be float; got {vec.dtype}")


def eta_additive(
    beta0: float,
    naics_effects: List[np.ndarray],
    zip_effects: List[np.ndarray],
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
) -> np.ndarray:
    """
    Compute the linear predictor under **additive per-level effects**.

    The model is:
        eta_i = beta0
                + sum_j NAICS_level_j[ naics_levels[i, j] ]
                + sum_m ZIP_level_m  [ zip_levels[i, m]   ]

    Parameters
    ----------
    beta0
        Global intercept (float).
    naics_effects
        List of per-level NAICS effect vectors. Length must equal `naics_levels.shape[1]`.
        For level `j`, the vector length must equal the number of NAICS groups at level `j`.
    zip_effects
        List of per-level ZIP effect vectors. Length must equal `zip_levels.shape[1]`.
    naics_levels
        Integer index matrix of shape (N, L_naics).
    zip_levels
        Integer index matrix of shape (N, L_zip).

    Returns
    -------
    np.ndarray
        Vector of eta (float64) of length N.

    Raises
    ------
    ValueError
        If shapes or lengths are inconsistent, or indices are out of bounds.

    Examples
    --------
    >>> import numpy as np
    >>> beta0 = -1.0
    >>> naics_effects = [np.array([0.1, -0.2]), np.array([0.05, 0.0, -0.05])]
    >>> zip_effects   = [np.array([0.2]),      np.array([-0.1, 0.0])]
    >>> naics_levels = np.array([[0, 2], [1, 1]], dtype=int)
    >>> zip_levels   = np.array([[0, 1], [0, 0]], dtype=int)
    >>> eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)
    array([-0.85, -1.1 ])
    """
    # Validate index matrices vs declared effect list lengths
    if not isinstance(naics_levels, np.ndarray) or naics_levels.ndim != 2:
        raise ValueError("`naics_levels` must be a 2-D numpy.ndarray.")
    if not isinstance(zip_levels, np.ndarray) or zip_levels.ndim != 2:
        raise ValueError("`zip_levels` must be a 2-D numpy.ndarray.")

    N = naics_levels.shape[0]
    L_naics = naics_levels.shape[1]
    L_zip = zip_levels.shape[1]

    _check_effects_list("naics_effects", naics_effects, L_naics)
    _check_effects_list("zip_effects", zip_effects, L_zip)

    eta = np.full(N, float(beta0), dtype=np.float64)

    if L_naics > 0 and naics_effects and all(len(v) > 0 for v in naics_effects):
        validate_level_indices(naics_levels, [len(v) for v in naics_effects])
        for j in range(L_naics):
            vec = naics_effects[j]
            eta += vec[naics_levels[:, j]]

    if L_zip > 0 and zip_effects and all(len(v) > 0 for v in zip_effects):
        validate_level_indices(zip_levels, [len(v) for v in zip_effects])
        for m in range(L_zip):
            vec = zip_effects[m]
            eta += vec[zip_levels[:, m]]

    return eta


def eta_nested(
    beta0: float,
    naics_base: np.ndarray,
    naics_deltas: List[np.ndarray],
    zip_base: np.ndarray,
    zip_deltas: List[np.ndarray],
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
) -> np.ndarray:
    """
    Compute the linear predictor under **base + nested deltas**.

    The model is:
        eta_i = beta0
                + NAICS_base        [ naics_levels[i, 0] ]
                + sum_{j=1..J-1} NAICS_delta_j[ naics_levels[i, j] ]
                + ZIP_base          [ zip_levels[i, 0]   ]
                + sum_{m=1..M-1} ZIP_delta_m [ zip_levels[i, m] ]

    Parameters
    ----------
    beta0
        Global intercept (float).
    naics_base
        Vector of base effects for NAICS level-0 (length K0).
    naics_deltas
        List of delta vectors for NAICS levels 1..J-1.
    zip_base
        Vector of base effects for ZIP level-0.
    zip_deltas
        List of delta vectors for ZIP levels 1..M-1.
    naics_levels
        Integer index matrix of shape (N, J). Column 0 indexes `naics_base`, column j
        indexes `naics_deltas[j-1]`.
    zip_levels
        Integer index matrix of shape (N, M). Column 0 indexes `zip_base`, column m
        indexes `zip_deltas[m-1]`.

    Returns
    -------
    np.ndarray
        Vector of eta (float64) of length N.

    Raises
    ------
    ValueError
        If shapes or lengths are inconsistent, or any index is out of bounds.

    Examples
    --------
    >>> import numpy as np
    >>> beta0 = -1.0
    >>> naics_base   = np.array([0.1, -0.1], dtype=float)
    >>> naics_deltas = [np.array([0.0, 0.05, -0.05], dtype=float)]
    >>> zip_base     = np.array([0.2], dtype=float)
    >>> zip_deltas   = [np.array([-0.1, 0.0], dtype=float)]
    >>> naics_levels = np.array([[0, 2], [1, 1]], dtype=int)  # J=2
    >>> zip_levels   = np.array([[0, 1], [0, 0]], dtype=int)  # M=2
    >>> eta_nested(beta0, naics_base, naics_deltas, zip_base, zip_deltas, naics_levels, zip_levels)
    array([-0.8 , -1.05])
    """
    # Validate matrices
    if not isinstance(naics_levels, np.ndarray) or naics_levels.ndim != 2:
        raise ValueError("`naics_levels` must be a 2-D numpy.ndarray.")
    if not isinstance(zip_levels, np.ndarray) or zip_levels.ndim != 2:
        raise ValueError("`zip_levels` must be a 2-D numpy.ndarray.")

    N, J = naics_levels.shape
    _, M = zip_levels.shape

    # Validate base vectors and delta lists
    _ensure_vector("naics_base", naics_base, length=int(naics_base.shape[0]))
    _ensure_vector("zip_base", zip_base, length=int(zip_base.shape[0]))

    if J == 0 or M == 0:
        raise ValueError(
            "`naics_levels` and `zip_levels` must each have at least one level (column 0)."
        )

    # Bounds check for level-0 vs base vectors
    validate_level_indices(naics_levels[:, :1], [len(naics_base)])
    validate_level_indices(zip_levels[:, :1], [len(zip_base)])

    # Deltas lengths must match deeper levels count
    if len(naics_deltas) != max(J - 1, 0):
        raise ValueError(
            f"`naics_deltas` length ({len(naics_deltas)}) must equal number of deeper NAICS "
            f"levels (J-1 = {J - 1})."
        )
    if len(zip_deltas) != max(M - 1, 0):
        raise ValueError(
            f"`zip_deltas` length ({len(zip_deltas)}) must equal number of deeper ZIP "
            f"levels (M-1 = {M - 1})."
        )

    for j, vec in enumerate(naics_deltas, start=1):
        _ensure_vector(f"naics_deltas[{j - 1}]", vec, length=int(vec.shape[0]))
    for m, vec in enumerate(zip_deltas, start=1):
        _ensure_vector(f"zip_deltas[{m - 1}]", vec, length=int(vec.shape[0]))

    # Bounds checks for deeper levels
    if J > 1:
        validate_level_indices(naics_levels[:, 1:], [len(v) for v in naics_deltas])
    if M > 1:
        validate_level_indices(zip_levels[:, 1:], [len(v) for v in zip_deltas])

    # Compute eta
    eta = np.full(N, float(beta0), dtype=np.float64)

    # Base contributions (level-0)
    eta += naics_base[naics_levels[:, 0]]
    eta += zip_base[zip_levels[:, 0]]

    # NAICS deltas
    for j in range(1, J):
        eta += naics_deltas[j - 1][naics_levels[:, j]]

    # ZIP deltas
    for m in range(1, M):
        eta += zip_deltas[m - 1][zip_levels[:, m]]

    return eta
