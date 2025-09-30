from __future__ import annotations

from typing import List, Mapping, Optional, TypedDict
import numpy as np
import pandas as pd
from .types import Strings, Integers


class HierIndex(TypedDict):
    levels: List[str]
    code_levels: np.ndarray  # (N, L) obs-level integer indices per level
    unique_per_level: List[np.ndarray]  # string labels per level in index order
    maps: List[Mapping[str, int]]  # level label -> int index
    group_counts: List[int]  # # groups per level
    parent_index_per_level: List[np.ndarray | None]  # len L; array for j>0 else None
    max_len: int
    cut_points: List[int]


def build_hierarchical_indices(
    codes: Strings,
    *,
    cut_points: Optional[Integers] = None,
    prefix_fill: Optional[str] = None,
) -> HierIndex:
    """
    Convert hierarchical codes (e.g., NAICS, ZIP) into per-level integer indices,
    **with prefix padding** and **child->parent pointers**.

    Parameters
    ----------
    codes
        Raw code strings.
    cut_points
        Monotone increasing lengths that define the hierarchy (e.g., [2,3,4,5,6]).
    prefix_fill
        Character to RIGHT-pad codes so slicing at deeper levels is safe.
        If None, defaults to '0'. To disable, pass empty string ('').

    Returns
    -------
    HierIndex
        - levels: names ["L{c}"] for each cut
        - code_levels: (N, L) int indices per observation per level
        - unique_per_level: list of np.ndarray of labels in index order
        - maps: list of dict[label->index] per level
        - group_counts: # unique labels per level
        - parent_index_per_level: for j>0, np.ndarray[Kj] that maps each group
          at level j to its parent group index at level j-1. For level 0: None.
        - max_len, cut_points
    """
    if len(codes) == 0:
        raise ValueError("`codes` cannot be empty.")
    codes = pd.Series(codes, dtype="string")
    if codes.isna().any():
        raise ValueError("`codes` contains null/NaN values; please clean input.")

    max_code_len = int(codes.str.len().max())
    # Infer default cuts if not provided
    if cut_points is None:
        if max_code_len == 6:
            cut_points = [2, 3, 4, 5, 6]
        elif max_code_len <= 5:
            cut_points = list(range(1, max_code_len + 1))
        else:
            cut_points = list(range(1, max_code_len + 1))

    # Validate cuts
    if len(cut_points) == 0:
        raise ValueError("`cut_points` cannot be empty when provided.")
    if any(int(c) <= 0 for c in cut_points):
        raise ValueError("cut_points must be positive.")
    if any(c2 <= c1 for c1, c2 in zip(cut_points, cut_points[1:])):
        raise ValueError("`cut_points` must be strictly increasing.")

    full_len = max(max_code_len, max(cut_points))

    # Enforce prefix_fill='0' by default unless explicitly disabled
    if prefix_fill is None:
        prefix_fill = "0"
    if prefix_fill:
        codes = codes.str.pad(width=full_len, side="right", fillchar=prefix_fill)

    levels: List[str] = [f"L{c}" for c in cut_points]
    unique_per_level: List[np.ndarray] = []
    maps: List[Mapping[str, int]] = []
    group_counts: List[int] = []
    code_levels = np.empty((len(codes), len(cut_points)), dtype=np.int64)

    for j, c in enumerate(cut_points):
        labels = codes.str.slice(0, c)
        uniq = pd.Index(labels).unique()
        cat = pd.Categorical(labels, categories=uniq, ordered=False)
        idx = cat.codes
        code_levels[:, j] = idx
        lab_arr = np.asarray(uniq, dtype="object")
        unique_per_level.append(lab_arr)
        maps.append({lab: i for i, lab in enumerate(lab_arr)})
        group_counts.append(len(uniq))

    # Build child->parent pointers between consecutive levels
    parent_index_per_level: List[np.ndarray | None] = [None]
    for j in range(1, len(cut_points)):
        child_labels = unique_per_level[j]
        parent_map = maps[j - 1]
        parent_cut = cut_points[j - 1]
        parent_idx_vec = np.empty(len(child_labels), dtype=np.int64)
        for g, child_lab in enumerate(child_labels):
            parent_lab = str(child_lab)[:parent_cut]
            try:
                parent_idx_vec[g] = parent_map[parent_lab]
            except KeyError as e:
                # Should not happen with padding + consistent slicing, but guard anyway.
                raise RuntimeError(
                    f"Parent label '{parent_lab}' not found at level {j - 1} "
                    f"for child '{child_lab}' at level {j}."
                ) from e
        parent_index_per_level.append(parent_idx_vec)

    return {
        "levels": levels,
        "code_levels": code_levels,
        "unique_per_level": unique_per_level,
        "maps": maps,
        "group_counts": group_counts,
        "parent_index_per_level": parent_index_per_level,
        "max_len": full_len,
        "cut_points": list(cut_points),
    }
