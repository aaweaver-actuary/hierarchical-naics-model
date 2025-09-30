from __future__ import annotations

from typing import List, Optional, TypedDict
import numpy as np
import pandas as pd
from .types import Strings, Integers, Mappings, Arrays


class HierIndex(TypedDict):
    levels: Strings
    code_levels: np.ndarray  # (N, L) obs-level integer indices per level
    unique_per_level: Arrays  # string labels per level in index order
    maps: Mappings  # level label -> int index
    group_counts: Integers  # # groups per level
    parent_index_per_level: List[np.ndarray | None]  # len L; array for j>0 else None
    max_len: int
    cut_points: Integers


def build_hierarchical_indices(
    codes: Strings,
    *,
    cut_points: Optional[Integers] = None,
    prefix_fill: Optional[str] = "0",
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

    # Early error check
    if not codes:
        raise ValueError("`codes` cannot be empty.")

    codes_series = pd.Series(codes, dtype="string")
    max_code_len = int(codes_series.str.len().max())
    cut_points = _get_cut_points(max_code_len, cut_points)
    _validate_cut_points(cut_points)
    full_len = max(max_code_len, max(cut_points))
    codes = _validate_codes(codes, prefix_fill=prefix_fill, full_len=full_len)

    def _build_parent_index_per_level(unique_per_level, maps, cut_points):
        parent_index_per_level: List[np.ndarray | None] = [None]
        for j in range(1, len(cut_points)):
            child_labels = unique_per_level[j]
            parent_map = maps[j - 1]
            parent_cut = cut_points[j - 1]
            parent_idx_vec = np.empty(len(child_labels), dtype=np.int64)
            for g, child_lab in enumerate(child_labels):
                parent_lab = str(child_lab)[:parent_cut]
                if parent_lab not in parent_map:
                    raise RuntimeError(
                        f"Parent label '{parent_lab}' not found at level {j - 1} "
                        f"for child '{child_lab}' at level {j}."
                    )
                parent_idx_vec[g] = parent_map[parent_lab]
            parent_index_per_level.append(parent_idx_vec)
        return parent_index_per_level

    levels: Strings = [f"L{c}" for c in cut_points]
    code_levels, unique_per_level, maps, group_counts = _build_code_levels(
        codes, cut_points
    )
    parent_index_per_level = _build_parent_index_per_level(
        unique_per_level, maps, cut_points
    )

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


def _get_cut_points(
    max_code_len: int, cut_points: Optional[Integers] = None
) -> Integers:
    if cut_points is not None:
        return cut_points
    if max_code_len == 6:
        return [2, 3, 4, 5, 6]
    return list(range(1, max_code_len + 1))


def _build_code_levels(codes, cut_points):
    unique_per_level: List[np.ndarray] = []
    maps: Optional[Mappings] = []
    group_counts: List[int] = []
    code_levels = np.empty((len(codes), len(cut_points)), dtype=np.int64)
    for j, c in enumerate(cut_points):
        labels = codes.str.slice(0, c)
        uniq = pd.Index(labels).unique()
        cat = pd.Categorical(labels, categories=uniq, ordered=False)
        code_levels[:, j] = cat.codes
        lab_arr = np.asarray(uniq, dtype="object")
        unique_per_level.append(lab_arr)
        maps.append({lab: i for i, lab in enumerate(lab_arr)})
        group_counts.append(len(uniq))
    return code_levels, unique_per_level, maps, group_counts


def _validate_cut_points(cut_points):
    _validate_cut_points_nonempty(cut_points)
    _validate_cut_points_positive(cut_points)
    _validate_cut_points_strictly_increasing(cut_points)


def _validate_cut_points_strictly_increasing(cut_points):
    is_cut_points_non_increasing = any(
        c2 <= c1 for c1, c2 in zip(cut_points, cut_points[1:])
    )
    if is_cut_points_non_increasing:
        raise ValueError("`cut_points` must be strictly increasing.")


def _validate_cut_points_positive(cut_points):
    is_any_cut_point_nonpositive = any(int(c) <= 0 for c in cut_points)
    if is_any_cut_point_nonpositive:
        raise ValueError("cut_points must be positive.")


def _validate_cut_points_nonempty(cut_points):
    is_cut_points_empty = len(cut_points) == 0
    if is_cut_points_empty:
        raise ValueError("`cut_points` cannot be empty when provided.")


def _validate_codes(
    codes: Strings, prefix_fill: Optional[str] = "0", full_len: Optional[int] = None
) -> pd.Series:
    if not codes:
        raise ValueError("`codes` cannot be empty.")

    codes_series = pd.Series(codes, dtype="string")
    if codes_series.isna().any():
        raise ValueError("`codes` contains null/NaN values; please clean input.")

    fill = prefix_fill if prefix_fill is not None else "0"
    if fill:
        length = full_len if full_len is not None else int(codes_series.str.len().max())
        return codes_series.str.pad(width=length, side="right", fillchar=fill)
    return codes_series
