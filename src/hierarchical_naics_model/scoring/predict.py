from __future__ import annotations

from typing import Mapping, Sequence, Dict
import pandas as pd


def predict_proba_nested(
    df_new: pd.DataFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Dict[str, object],
    prefix_fill: str = "0",
    return_components: bool = True,
) -> pd.DataFrame:
    """
    Score rows using base + nested deltas with parent backoff at each level.

    Parameters
    ----------
    df_new
        DataFrame containing at least `naics_col` and `zip_col`.
    naics_col, zip_col
        Column names with raw codes (strings).
    naics_cut_points, zip_cut_points
        Level definitions used in training.
    naics_level_maps, zip_level_maps
        Per-level label→index dicts learned from training.
    effects
        Output of `extract_effect_tables_nested`.
    prefix_fill
        Right-padding character for incoming codes.
    return_components
        If True, include per-level backoff flags in the output.

    Returns
    -------
    pd.DataFrame
        Input with added columns: `eta`, `p`, and optional backoff flags.
    """
    # TODO: implement
    # - resolve indices with parent fallback
    # - sum base + deltas when available
    # - compute sigma(eta) and append columns
    raise NotImplementedError
