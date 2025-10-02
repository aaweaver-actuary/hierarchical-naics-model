from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, List, cast

import numpy as np
import pandas as pd


__all__ = ["predict_proba_nested"]


def _pad(code: str | None, width: int, fill: str) -> str:
    raw = "" if code is None else str(code).strip()
    return raw if len(raw) >= width else raw.ljust(width, fill)


def _labels_for_levels(
    code: str | None, cut_points: Sequence[int], max_len: int, fill: str
) -> List[str]:
    padded = _pad(code, max_len, fill)
    return [padded[:c] for c in cut_points]


def _logistic(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def predict_proba_nested(
    df_new: pd.DataFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Mapping[str, Any],
    prefix_fill: str = "0",
    return_components: bool = True,
) -> pd.DataFrame:
    """
    Score rows with **base + nested deltas** using simple **per-level availability**.

    Rule
    ----
    - For each hierarchy:
      - Base (level 0): add base effect if label exists, otherwise add 0.
      - Level j>=1 delta: add delta only if that exact level label exists; do **not**
        reuse the parent’s index (i.e., no “leaking” parent effect into a deeper delta).
    - Emit `*_known` flags per level; `any_backoff` is True if any level is unknown.

    Parameters
    ----------
    df_new : pd.DataFrame
        Must contain `naics_col`, `zip_col`.
    naics_col, zip_col : str
        Column names for codes.
    naics_cut_points, zip_cut_points : Sequence[int]
        Prefix lengths per level (training-time definition).
    naics_level_maps, zip_level_maps : Sequence[Mapping[str,int]]
        Per-level label → index dicts learned at training.
    effects : dict
        Output of `extract_effect_tables_nested`.
    prefix_fill : str
        Right-padding character.
    return_components : bool
        If True, include `*_known` booleans and `any_backoff`.

    Returns
    -------
    pd.DataFrame
        Copy of `df_new` with new `eta` and `p` columns (and flags if requested).
    """
    beta0 = float(effects["beta0"])
    naics_base = cast(pd.Series, effects["naics_base"])
    zip_base = cast(pd.Series, effects["zip_base"])

    naics_deltas_raw = effects.get("naics_deltas", [])
    zip_deltas_raw = effects.get("zip_deltas", [])
    naics_deltas = list(cast(Iterable[pd.Series], naics_deltas_raw))
    zip_deltas = list(cast(Iterable[pd.Series], zip_deltas_raw))

    L_n = len(naics_cut_points)
    L_z = len(zip_cut_points)
    max_n = max(naics_cut_points) if L_n > 0 else 0
    max_z = max(zip_cut_points) if L_z > 0 else 0

    eta = np.full(len(df_new), beta0, dtype=float)
    naics_known = (
        np.zeros((len(df_new), L_n), dtype=bool)
        if L_n
        else np.ones((len(df_new), 0), dtype=bool)
    )
    zip_known = (
        np.zeros((len(df_new), L_z), dtype=bool)
        if L_z
        else np.ones((len(df_new), 0), dtype=bool)
    )

    for i, (n_code, z_code) in enumerate(
        df_new[[naics_col, zip_col]].itertuples(index=False, name=None)
    ):
        n_labels = (
            _labels_for_levels(n_code, naics_cut_points, max_n, prefix_fill)
            if L_n
            else []
        )
        z_labels = (
            _labels_for_levels(z_code, zip_cut_points, max_z, prefix_fill)
            if L_z
            else []
        )

        # NAICS base
        if L_n:
            if n_labels[0] in naics_level_maps[0]:
                idx0 = naics_level_maps[0][n_labels[0]]
                eta[i] += float(naics_base.iloc[idx0])
                naics_known[i, 0] = True

            # Deltas
            for j in range(1, L_n):
                if j - 1 < len(naics_deltas) and n_labels[j] in naics_level_maps[j]:
                    idx = naics_level_maps[j][n_labels[j]]
                    eta[i] += float(naics_deltas[j - 1].iloc[idx])
                    naics_known[i, j] = True

        # ZIP base
        if L_z:
            if z_labels[0] in zip_level_maps[0]:
                idx0 = zip_level_maps[0][z_labels[0]]
                eta[i] += float(zip_base.iloc[idx0])
                zip_known[i, 0] = True

            for m in range(1, L_z):
                if m - 1 < len(zip_deltas) and z_labels[m] in zip_level_maps[m]:
                    idx = zip_level_maps[m][z_labels[m]]
                    eta[i] += float(zip_deltas[m - 1].iloc[idx])
                    zip_known[i, m] = True

    p = _logistic(eta)

    out = df_new.copy()
    out["eta"] = eta
    out["p"] = p

    if return_components:
        for j, cut in enumerate(naics_cut_points):
            out[f"naics_L{cut}_known"] = naics_known[:, j]
        for m, cut in enumerate(zip_cut_points):
            out[f"zip_L{cut}_known"] = zip_known[:, m]
        out["any_backoff"] = ~(naics_known.all(axis=1) & zip_known.all(axis=1))

    return out
