from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


__all__ = ["validate_cut_points", "validate_level_indices"]


def validate_cut_points(cut_points: Sequence[int]) -> Tuple[int, ...]:
    """
    Validate that cut points are positive and strictly increasing.

    Parameters
    ----------
    cut_points
        Proposed per-level prefix lengths (e.g., [2, 3, 5]).

    Returns
    -------
    tuple of int
        Normalized, immutable cut points.

    Raises
    ------
    ValueError
        If `cut_points` is empty, contains non-positive values, or is not strictly
        increasing.

    Examples
    --------
    >>> validate_cut_points([2, 3, 6])
    (2, 3, 6)
    >>> validate_cut_points([2, 2, 3])
    Traceback (most recent call last):
        ...
    ValueError: cut_points must be strictly increasing (e.g., [2, 3, 5]); got [2, 2, 3]
    """
    try:
        cp = tuple(int(c) for c in cut_points)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cut_points must be a sequence of ints; got {cut_points!r}"
        ) from exc

    if len(cp) == 0:
        raise ValueError("cut_points cannot be empty.")
    if any(c <= 0 for c in cp):
        raise ValueError(f"cut_points must be positive integers; got {list(cp)}")
    if any(b <= a for a, b in zip(cp, cp[1:])):
        raise ValueError(
            f"cut_points must be strictly increasing (e.g., [2, 3, 5]); got {list(cp)}"
        )
    return cp


def validate_level_indices(levels: np.ndarray, group_counts: Sequence[int]) -> None:
    """
    Validate a matrix of per-level group indices against declared group counts.

    Parameters
    ----------
    levels
        Integer matrix of shape (N, L) where `levels[i, j]` is the group index for
        observation `i` at level `j`. May be empty if `L == 0`.
    group_counts
        Sequence of length `L` giving the number of groups at each level (K_j).

    Raises
    ------
    ValueError
        If `levels` is not 2-D, if the number of levels does not match
        `len(group_counts)`, or if any index is out of bounds [0, K_j-1].

    Notes
    -----
    - This function does not modify inputs and returns `None` on success.
    - Use this before building models or computing effects to fail fast on shape
      mistakes.
    """
    if not isinstance(levels, np.ndarray):
        raise ValueError(
            f"`levels` must be a numpy.ndarray; got {type(levels).__name__}"
        )
    if levels.ndim != 2:
        raise ValueError(f"`levels` must be 2-D (N, L); got shape {levels.shape}")

    N, L = levels.shape
    try:
        gc = [int(x) for x in group_counts]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"`group_counts` must be a sequence of ints; got {group_counts!r}"
        ) from exc

    if len(gc) != L:
        raise ValueError(
            f"`group_counts` length ({len(gc)}) must match number of levels L={L}."
        )
    if any(k <= 0 for k in gc):
        raise ValueError(f"all group_counts must be positive; got {gc}")

    # Empty matrix is acceptable (e.g., N==0); still verify dtype when N>0
    if N == 0:
        return

    # Ensure integer type
    if not np.issubdtype(levels.dtype, np.integer):
        raise ValueError(f"`levels` dtype must be an integer type; got {levels.dtype}")

    # Bounds check per level
    for j in range(L):
        col = levels[:, j]
        if col.size == 0:
            continue
        min_val = int(col.min())
        max_val = int(col.max())
        if min_val < 0 or max_val >= gc[j]:
            raise ValueError(
                f"indices out of bounds at level {j}: observed min={min_val}, max={max_val}, "
                f"allowed range=[0, {gc[j] - 1}]."
            )
