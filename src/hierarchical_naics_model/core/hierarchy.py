# nested_quotewrite/core/hierarchy.py
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd


__all__ = ["HierIndex", "build_hierarchical_indices", "make_backoff_resolver"]


class HierIndex(TypedDict):
    levels: list[str]
    code_levels: np.ndarray
    unique_per_level: list[np.ndarray]
    maps: list[Mapping[str, int]]
    group_counts: list[int]
    parent_index_per_level: list[np.ndarray | None]
    max_len: int
    cut_points: list[int]


# ------------------------------ #
# Internal helpers (module-local) #
# ------------------------------ #


def _validate_cut_points(cut_points: Sequence[int]) -> List[int]:
    cp = [int(c) for c in cut_points]
    if not cp:
        raise ValueError("cut_points cannot be empty.")
    if any(c <= 0 for c in cp):
        raise ValueError("cut_points must be positive integers.")
    if any(b <= a for a, b in zip(cp, cp[1:])):
        raise ValueError("cut_points must be strictly increasing (e.g., [2,3,5]).")
    return cp


def _to_str_series(codes: Sequence[str]) -> pd.Series:
    s = pd.Series(codes, dtype="string")
    s = s.fillna("").str.strip()
    # Ensure only digits remain? We don't hard-enforce here; padding/slicing is robust.
    return s


def _right_pad(s: pd.Series, width: int, fill: str) -> pd.Series:
    if not isinstance(fill, str) or len(fill) != 1:
        raise ValueError("prefix_fill must be a single character string, e.g., '0'.")
    if width <= 0:
        raise ValueError("Padding width must be positive.")
    # right pad to width using fill; if longer than width, keep as-is (do not truncate)
    return s.where(
        s.str.len() >= width, s.str.pad(width=width, side="right", fillchar=fill)
    )


def _slice_levels(s: pd.Series, cut_points: Sequence[int]) -> List[pd.Series]:
    return [s.str.slice(0, c) for c in cut_points]


def _factorize_stable(
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Factorize preserving first-seen order of unique labels.

    Returns
    -------
    codes : np.ndarray[int64]
        Per-row indices 0..K-1.
    uniques : np.ndarray[object]
        Unique labels in index order.
    mapping : dict[str, int]
        Label -> index.
    """
    uniq = pd.Index(labels).unique()
    cat = pd.Categorical(labels, categories=uniq, ordered=False)
    codes = cat.codes.astype(np.int64, copy=False)
    uniques = np.asarray(uniq.to_numpy(), dtype="object")
    mapping: Dict[str, int] = {str(uniques[i]): int(i) for i in range(len(uniques))}
    return codes, uniques, mapping


def _parent_index_for_level(
    child_uniques: np.ndarray,
    parent_map: Mapping[str, int],
    parent_cut: int,
) -> np.ndarray:
    """
    Map each unique label at level j to its parent index at level j-1.

    If a parent label is not present in `parent_map`, returns -1 for that child.
    """
    out = np.empty(len(child_uniques), dtype=np.int64)
    for i, lab in enumerate(child_uniques):
        par_label = str(lab)[:parent_cut]
        out[i] = parent_map.get(par_label, -1)
    return out


# ------------------------------ #
# Public API                     #
# ------------------------------ #


def build_hierarchical_indices(
    codes: Sequence[str],
    *,
    cut_points: Sequence[int],
    prefix_fill: str = "0",
) -> HierIndex:
    """
    Build per-level integer indices for hierarchical string codes (e.g., NAICS, ZIP).

    Parameters
    ----------
    codes
        Raw code strings (may be short or contain whitespace). Values are coerced to
        pandas "string" dtype; missing values become empty strings.
    cut_points
        Strictly increasing prefix lengths defining levels (e.g., [2, 3, 4, 5, 6]).
        The first element defines the most general level (L0), and the last typically
        represents the leaf level.
    prefix_fill
        Character used to right-pad codes to `max(cut_points)`. Default is '0'. Padding
        protects hierarchy slicing (e.g., NAICS "52" -> "520000" so that L3 is "520").

    Returns
    -------
    HierIndex
        Mapping with:
          - levels (list[str]): human-readable names (["L2","L3",...]).
          - code_levels (np.ndarray, shape (N,L)): per-row, per-level group indices.
          - unique_per_level (list[np.ndarray]): per-level unique label arrays.
          - maps (list[Mapping[str,int]]): per-level label->index dicts.
          - group_counts (list[int]): number of groups at each level.
          - parent_index_per_level (list[np.ndarray|None]): for j>0, child->parent index.
          - max_len (int): padding length (max(cut_points)).
          - cut_points (list[int]): echo of validated cut points.

    Notes
    -----
    - We do not truncate codes longer than `max(cut_points)`; slicing uses prefixes of
      the padded/unaltered code, so extra suffix characters do not affect grouping.
    - Parent pointers are derived from per-level uniques, not row-level indices.
    - For empty/invalid codes, after padding they become a string of the fill char
      (e.g., '000000'); they will be grouped consistently by prefix just like others.

    Examples
    --------
    >>> idx = build_hierarchical_indices(["52", "521", "522110"], cut_points=[2,3,6])
    >>> idx["levels"]
    ['L2', 'L3', 'L6']
    >>> idx["group_counts"][0]
    1
    """
    cp = _validate_cut_points(cut_points)
    max_len = max(cp)

    s = _to_str_series(codes)
    s = _right_pad(s, width=max_len, fill=prefix_fill)

    # Slice to level labels
    label_cols = _slice_levels(s, cp)  # list of Series, length L
    levels_names = [f"L{c}" for c in cp]

    # Factorize each level independently
    code_cols: List[np.ndarray] = []
    unique_per_level: List[np.ndarray] = []
    maps: List[Mapping[str, int]] = []
    group_counts: List[int] = []

    for lab_s in label_cols:
        codes_j, uniq_j, map_j = _factorize_stable(lab_s)
        code_cols.append(codes_j)
        unique_per_level.append(uniq_j)
        maps.append(map_j)
        group_counts.append(int(len(uniq_j)))

    # Stack per-level codes into (N, L)
    code_levels = (
        np.column_stack(code_cols)
        if code_cols
        else np.empty((len(s), 0), dtype=np.int64)
    )

    # Build parent pointers for group uniques (not row-level)
    parent_index_per_level: List[np.ndarray | None] = [None]
    for j in range(1, len(cp)):
        parent_map = maps[j - 1]
        parent_cut = cp[j - 1]
        child_uniques = unique_per_level[j]
        parent_idx_vec = _parent_index_for_level(child_uniques, parent_map, parent_cut)
        parent_index_per_level.append(parent_idx_vec)

    result: HierIndex = {
        "levels": levels_names,
        "code_levels": code_levels,
        "unique_per_level": unique_per_level,
        "maps": maps,
        "group_counts": group_counts,
        "parent_index_per_level": parent_index_per_level,
        "max_len": max_len,
        "cut_points": cp,
    }
    return result


def make_backoff_resolver(
    *,
    cut_points: Sequence[int],
    level_maps: Sequence[Mapping[str, int]],
    prefix_fill: str = "0",
) -> Callable[[str], List[int | None]]:
    """
    Create a resolver that maps a raw code to per-level indices with parent backoff.

    At each requested level j, the resolver attempts to look up the label formed by
    the first `cut_points[j]` characters of the **right-padded** code. If that label
    is unknown at level j, it backs off to parent levels j-1, j-2, ..., 0 until it
    finds a known ancestor. If no ancestor is known, `None` is returned for that level.

    Parameters
    ----------
    cut_points
        Strictly increasing prefix lengths (same sequence used in training).
    level_maps
        Per-level dicts of label → index learned from training data.
    prefix_fill
        Right-padding character used before slicing (e.g., '0').

    Returns
    -------
    callable
        Function `resolve(code: str) -> list[int | None]`.

    Examples
    --------
    >>> # Suppose training saw NAICS L2: {"52":0}, L3: {"521":0}, L6: {"522110":1}
    >>> cp = [2,3,6]
    >>> maps = [{"52":0}, {"521":0}, {"522110":1}]
    >>> resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="0")
    >>> resolve("521110")  # exact L3 present, L6 unknown -> backoff to L3/L2
    [0, 0, 0]
    >>> resolve("522130")  # L6 unknown; parent L3 "522" unknown; parent L2 "52" known
    [0, None, 0]
    """
    cp = _validate_cut_points(cut_points)
    if len(level_maps) != len(cp):
        raise ValueError("level_maps length must equal number of cut_points.")

    # Normalize maps to plain dicts (faster .get) and guard against non-string keys
    norm_maps: List[Dict[str, int]] = []
    for m in level_maps:
        dm: Dict[str, int] = {}
        for k, v in m.items():
            dm[str(k)] = int(v)
        norm_maps.append(dm)

    max_len = max(cp)

    def resolve(code: str) -> List[int | None]:
        raw = "" if code is None else str(code).strip()
        padded = raw if len(raw) >= max_len else raw.ljust(max_len, prefix_fill)

        out: List[int | None] = []
        for j, cut in enumerate(cp):
            # Try level j; if not found, walk up to parent levels.
            idx: int | None = None
            k = j
            while k >= 0 and idx is None:
                lab = padded[: cp[k]]
                cand = norm_maps[k].get(lab)
                if cand is not None:
                    idx = cand
                else:
                    k -= 1
            out.append(idx)
        return out

    return resolve
