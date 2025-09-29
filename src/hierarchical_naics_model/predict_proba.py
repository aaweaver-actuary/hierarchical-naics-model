from __future__ import annotations
from typing import Dict, List, Mapping, Sequence
import numpy as np
import pandas as pd

from .make_backoff_resolver import make_backoff_resolver
from .types import Integers


def predict_proba(
    df_new: pd.DataFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: Integers,
    zip_cut_points: Integers,
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Dict[str, object],
    prefix_fill: str = "0",
    return_components: bool = True,
) -> pd.DataFrame:
    """
    Score with nested-delta effects:
      eta = beta0 + naics_base[idx0] + Σ naics_delta_j[idxj] + zip_base[idx0] + Σ zip_delta_m[idxm]
    Backoff: if a level index is None → contribution 0 for that level.
    """
    required = {"beta0", "naics_base", "naics_deltas", "zip_base", "zip_deltas"}
    if not required.issubset(effects):
        raise ValueError(f"`effects` missing keys: {required - set(effects)}")

    beta0 = float(effects["beta0"])  # type: ignore
    naics_base: pd.Series = effects["naics_base"]  # type: ignore
    naics_deltas: List[pd.Series] = effects["naics_deltas"]  # type: ignore
    zip_base: pd.Series = effects["zip_base"]  # type: ignore
    zip_deltas: List[pd.Series] = effects["zip_deltas"]  # type: ignore

    res_naics = make_backoff_resolver(
        cut_points=naics_cut_points,
        level_maps=naics_level_maps,
        prefix_fill=prefix_fill,
    )
    res_zip = make_backoff_resolver(
        cut_points=zip_cut_points, level_maps=zip_level_maps, prefix_fill=prefix_fill
    )

    naics_idx = df_new[naics_col].astype(str).map(res_naics).to_list()
    zip_idx = df_new[zip_col].astype(str).map(res_zip).to_list()

    n = len(df_new)
    eta = np.full(n, beta0, dtype=float)

    # NAICS base at level 0 (most general)
    for i in range(n):
        i0 = naics_idx[i][0]
        if i0 is not None:
            eta[i] += float(naics_base.loc[i0])
    # NAICS deltas
    for j, delta_tbl in enumerate(naics_deltas, start=1):
        for i in range(n):
            ij = naics_idx[i][j]
            if ij is not None:
                eta[i] += float(delta_tbl.loc[ij])

    # ZIP base
    for i in range(n):
        i0 = zip_idx[i][0]
        if i0 is not None:
            eta[i] += float(zip_base.loc[i0])
    # ZIP deltas
    for m, delta_tbl in enumerate(zip_deltas, start=1):
        for i in range(n):
            im = zip_idx[i][m]
            if im is not None:
                eta[i] += float(delta_tbl.loc[im])

    p = 1.0 / (1.0 + np.exp(-eta))

    out = df_new.copy()
    out["eta"] = eta
    out["p"] = p

    if return_components:
        # Optional: emit whether a given level was missing and required backoff.
        # Here we simply flag None at that exact level (we don't differentiate which parent supplied).
        for j in range(len(naics_cut_points)):
            out[f"backoff_naics_{j}"] = [idxs[j] is None for idxs in naics_idx]
        for m in range(len(zip_cut_points)):
            out[f"backoff_zip_{m}"] = [idxs[m] is None for idxs in zip_idx]

    return out
