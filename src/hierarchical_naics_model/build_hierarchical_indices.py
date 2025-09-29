from __future__ import annotations

from typing import Dict, List, Mapping, Sequence
import numpy as np
import pandas as pd


def build_hierarchical_indices(
    codes: Sequence[str],
    *,
    cut_points: Sequence[int] | None = None,
    prefix_fill: str | None = None,
) -> Dict[str, object]:
    """
    Convert hierarchical categorical codes (e.g., NAICS or ZIP) into per-level
    integer indices suitable for hierarchical partial pooling.

    Parameters
    ----------
    codes
        Sequence of raw codes (strings). Example NAICS: "511110"; ZIP: "30309".
    cut_points
        Monotone increasing character-lengths indicating hierarchy cuts.
        If None, sensible defaults are inferred from the code lengths:
        - If max length == 6 (NAICS-like), defaults to [2, 3, 4, 5, 6].
        - If max length <= 5 (ZIP-like), defaults to [1, 2, ..., max_len].
        - Otherwise, defaults to [1, 2, ..., max_len].
    prefix_fill
        Optional character used to right-pad shorter codes to the maximum length.
        If `None`, codes are used as-is. For numeric-like codes that might be
        missing trailing digits, pass `'0'` to ensure stable prefix extraction.

    Returns
    -------
    out : dict
        - 'levels': list[str]
            Names "L{cut}" for each cut (e.g., "L2","L3","L5").
        - 'code_levels': np.ndarray, shape (N, L)
            At row i and level ℓ, the integer group index for codes[i] at that level.
        - 'unique_per_level': list[np.ndarray]
            Unique *string* labels at each level in index order.
        - 'maps': list[Mapping[str, int]]
            List of dicts mapping level label -> integer index.
        - 'group_counts': list[int]
            Number of unique groups at each level.
        - 'max_len': int
            The maximum length used when slicing prefixes.

    Notes
    -----
    - This function does not assume NAICS or ZIP specifically. It only slices
      prefixes at the `cut_points` you pass and ranks unique labels to integers.
    - You can call this separately for NAICS and ZIP and then pass the outputs
      into the PyMC model builder.

    Examples
    --------
    >>> build = build_hierarchical_indices(["511110","511120","512130"], cut_points=[2,3,6])
    >>> build["levels"]
    ['L2', 'L3', 'L6']
    >>> build["group_counts"]  # 2-digit groups, 3-digit groups, 6-digit groups
    [1, 2, 3]
    >>> build["code_levels"].shape
    (3, 3)
    """
    if len(codes) == 0:
        raise ValueError("`codes` cannot be empty.")

    codes = pd.Series(codes, dtype="string")
    max_len = int(codes.str.len().max())

    # Infer default cuts if not provided
    if cut_points is None:
        if max_len == 6:
            cut_points = [2, 3, 4, 5, 6]
        elif max_len <= 5:
            cut_points = list(range(1, max_len + 1))
        else:
            cut_points = list(range(1, max_len + 1))

    full_len = max(max_len, max(cut_points))

    if prefix_fill is not None:
        codes = codes.str.pad(width=full_len, side="right", fillchar=prefix_fill)

    levels: List[str] = [f"L{c}" for c in cut_points]
    unique_per_level: List[np.ndarray] = []
    maps: List[Mapping[str, int]] = []
    group_counts: List[int] = []
    code_levels = np.empty((len(codes), len(cut_points)), dtype=np.int64)

    for j, c in enumerate(cut_points):
        labels = codes.str.slice(0, c)
        uniq = np.asarray(pd.Index(labels).unique(), dtype="object")
        # stable order via pandas categorical codes
        cat = pd.Categorical(labels, categories=uniq, ordered=False)
        # cat.codes may already be an ndarray; ensure ndarray and int dtype
        idx = np.asarray(cat.codes, dtype=np.int64)  # 0..K-1
        code_levels[:, j] = idx
        unique_per_level.append(uniq)
        maps.append({u: i for i, u in enumerate(uniq)})
        group_counts.append(len(uniq))

    return {
        "levels": levels,
        "code_levels": code_levels,
        "unique_per_level": unique_per_level,
        "maps": maps,
        "group_counts": group_counts,
        "max_len": full_len,
    }
