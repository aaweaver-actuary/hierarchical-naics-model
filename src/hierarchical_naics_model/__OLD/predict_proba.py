from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from .make_backoff_resolver import make_backoff_resolver
from .types import Integers, Mappings

from .logger import get_logger

log = get_logger(__name__)


def predict_proba(
    df_new: pd.DataFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: Integers,
    zip_cut_points: Integers,
    naics_level_maps: Mappings,
    zip_level_maps: Mappings,
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

    def _apply_effects(idx_list, base, deltas, cut_points):
        n = len(idx_list)
        eta = np.zeros(n, dtype=float)
        backoff = [[False] * len(cut_points) for _ in range(n)]
        for i, idxs in enumerate(idx_list):
            for j in range(len(cut_points)):
                idx = idxs[j] if len(idxs) > j else None
                if idx is not None and not pd.isna(idx):
                    if j == 0:
                        eta[i] += float(base.loc[idx])
                    else:
                        eta[i] += float(deltas[j - 1].loc[idx])
                else:
                    backoff[i][j] = True
        return eta, backoff

    n = len(df_new)
    eta = np.full(n, beta0, dtype=float)
    naics_eta, backoff_naics = _apply_effects(
        naics_idx, naics_base, naics_deltas, naics_cut_points
    )
    zip_eta, backoff_zip = _apply_effects(zip_idx, zip_base, zip_deltas, zip_cut_points)
    eta += naics_eta + zip_eta
    p = 1.0 / (1.0 + np.exp(-eta))

    out = df_new.copy()
    out["eta"] = eta
    out["p"] = p
    for j in range(len(naics_cut_points)):
        out[f"backoff_naics_{j}"] = [flags[j] for flags in backoff_naics]
    for m in range(len(zip_cut_points)):
        out[f"backoff_zip_{m}"] = [flags[m] for flags in backoff_zip]
    return out


# Helper to serialize trained maps for scoring
def serialize_level_maps(level_maps):
    """Convert per-level dicts to a serializable format (e.g., for JSON)."""
    return [dict(m) for m in level_maps]
