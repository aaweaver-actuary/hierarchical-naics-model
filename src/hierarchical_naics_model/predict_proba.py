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
    log.debug(f"NAICS backoff resolver: {res_naics}")
    res_zip = make_backoff_resolver(
        cut_points=zip_cut_points, level_maps=zip_level_maps, prefix_fill=prefix_fill
    )
    log.debug(f"ZIP backoff resolver: {res_zip}")

    naics_idx = df_new[naics_col].astype(str).map(res_naics).to_list()
    log.debug(f"NAICS indices for new data: {naics_idx}")
    zip_idx = df_new[zip_col].astype(str).map(res_zip).to_list()
    log.debug(f"ZIP indices for new data: {zip_idx}")
    log.debug(f"Effects: {effects}")
    log.debug(f"NAICS level maps: {naics_level_maps}")
    log.debug(f"ZIP level maps: {zip_level_maps}")

    n = len(df_new)
    eta = np.full(n, beta0, dtype=float)
    # Track backoff flags per level
    backoff_naics = [[False] * len(naics_cut_points) for _ in range(n)]
    backoff_zip = [[False] * len(zip_cut_points) for _ in range(n)]

    # NAICS base at level 0 (most general)
    for i in range(n):
        i0 = naics_idx[i][0] if len(naics_idx[i]) > 0 else None
        log.debug(f"Row {i} NAICS base idx0: {i0}")
        if i0 is not None:
            eta[i] += float(naics_base.loc[i0])
            log.debug(f"Row {i} NAICS base contribution: {float(naics_base.loc[i0])}")
        else:
            backoff_naics[i][0] = True
            log.debug(f"Row {i} NAICS base backoff: True")
    # NAICS deltas
    for j, delta_tbl in enumerate(naics_deltas, start=1):
        for i in range(n):
            ij = naics_idx[i][j] if len(naics_idx[i]) > j else None
            log.debug(f"Row {i} NAICS delta level {j} idx: {ij}")
            if ij is not None:
                eta[i] += float(delta_tbl.loc[ij])
                log.debug(
                    f"Row {i} NAICS delta level {j} contribution: {float(delta_tbl.loc[ij])}"
                )
            else:
                print(f"Setting backoff_naics[{i}][{j}] = True (ij={ij})")
                backoff_naics[i][j] = True
                log.debug(f"Row {i} NAICS delta level {j} backoff: True")
    # ZIP base
    for i in range(n):
        i0 = zip_idx[i][0] if len(zip_idx[i]) > 0 else None
        log.debug(f"Row {i} ZIP base idx0: {i0}")
        if i0 is not None:
            eta[i] += float(zip_base.loc[i0])
            log.debug(f"Row {i} ZIP base contribution: {float(zip_base.loc[i0])}")
        else:
            backoff_zip[i][0] = True
            log.debug(f"Row {i} ZIP base backoff: True")
    # ZIP deltas
    for m, delta_tbl in enumerate(zip_deltas, start=1):
        for i in range(n):
            im = zip_idx[i][m] if len(zip_idx[i]) > m else None
            log.debug(f"Row {i} ZIP delta level {m} idx: {im}")
            if im is not None:
                eta[i] += float(delta_tbl.loc[im])
                log.debug(
                    f"Row {i} ZIP delta level {m} contribution: {float(delta_tbl.loc[im])}"
                )
            else:
                print(f"Setting backoff_zip[{i}][{m}] = True (im={im})")
                backoff_zip[i][m] = True
                log.debug(f"Row {i} ZIP delta level {m} backoff: True")

    p = 1.0 / (1.0 + np.exp(-eta))
    out = df_new.copy()
    out["eta"] = eta
    out["p"] = p
    # Force all flags to Python bools after assignment
    backoff_naics = [[bool(x) for x in row] for row in backoff_naics]
    backoff_zip = [[bool(x) for x in row] for row in backoff_zip]
    for j in range(len(naics_cut_points)):
        out[f"backoff_naics_{j}"] = [flags[j] for flags in backoff_naics]
        log.debug(f"backoff_naics_{j}: {[flags[j] for flags in backoff_naics]}")
    for m in range(len(zip_cut_points)):
        out[f"backoff_zip_{m}"] = [flags[m] for flags in backoff_zip]
        log.debug(f"backoff_zip_{m}: {[flags[m] for flags in backoff_zip]}")
    log.debug(f"Final output DataFrame:\n{out}")
    return out


# Helper to serialize trained maps for scoring
def serialize_level_maps(level_maps):
    """Convert per-level dicts to a serializable format (e.g., for JSON)."""
    return [dict(m) for m in level_maps]
