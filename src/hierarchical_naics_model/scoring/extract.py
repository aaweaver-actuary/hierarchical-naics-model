from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr


__all__ = ["extract_effect_tables_nested"]


def _posterior_mean_1d(da: xr.DataArray, dim=("chain", "draw")) -> np.ndarray:
    """Return posterior mean as 1D numpy array, preserving coord order if present."""
    mean_da = da.mean(dim=dim)
    # Return values in declared coord order if any; otherwise as-is.
    return np.asarray(mean_da.values, dtype=float)


def _collect_sorted_deltas(posterior: xr.Dataset, prefix: str) -> List[str]:
    """
    Find variables named like f"{prefix}_delta_<k>" and return their names sorted by <k> (int).
    Example: prefix="naics" -> ["naics_delta_1", "naics_delta_2", ...]
    """
    cand = [v for v in posterior.data_vars.keys() if v.startswith(f"{prefix}_delta_")]

    def take_k(v: str) -> int:
        try:
            return int(v.split("_")[-1])
        except Exception:
            return 10**9

    return [v for v in sorted(cand, key=take_k)]


def extract_effect_tables_nested(
    idata,
) -> Dict[str, Dict[str, Union[float, pd.Series, List[pd.Series]]]]:
    """
    Reduce posterior samples to posterior means for nested-deltas scoring.

    Parameters
    ----------
    idata
        ArviZ InferenceData with groups/variables:
          - posterior["beta0"] (scalar)
          - posterior["naics_base"] (len K0)
          - posterior["zip_base"]   (len Z0)
          - optional posterior["naics_delta_j"] for j>=1
          - optional posterior["zip_delta_m"]   for m>=1

    Returns
    -------
    dict
        {
          "beta0": float,
          "naics_base": pd.Series,        # index 0..K0-1
          "naics_deltas": list[pd.Series],
          "zip_base": pd.Series,
          "zip_deltas": list[pd.Series],
        }
    """
    if xr is None:
        raise RuntimeError("xarray/arviz is required to extract posterior tables.")

    if "posterior" not in idata.groups():
        raise ValueError("InferenceData does not contain a 'posterior' group.")

    post = idata.posterior

    # Scalars
    if "beta0" not in post:
        raise ValueError("posterior is missing variable 'beta0'.")
    beta0 = float(post["beta0"].mean(("chain", "draw")).values)

    # Bases
    for req in ("naics_base", "zip_base"):
        if req not in post:
            raise ValueError(f"posterior is missing variable '{req}'.")

    naics_base = _posterior_mean_1d(post["naics_base"])
    zip_base = _posterior_mean_1d(post["zip_base"])

    # Deltas
    naics_delta_names = _collect_sorted_deltas(post, "naics")
    zip_delta_names = _collect_sorted_deltas(post, "zip")

    naics_deltas = [
        pd.Series(
            _posterior_mean_1d(post[name]),
            index=pd.RangeIndex(len(_posterior_mean_1d(post[name]))),
        )
        for name in naics_delta_names
    ]
    zip_deltas = [
        pd.Series(
            _posterior_mean_1d(post[name]),
            index=pd.RangeIndex(len(_posterior_mean_1d(post[name]))),
        )
        for name in zip_delta_names
    ]

    # Wrap bases as Series with integer index
    effects = {
        "beta0": beta0,
        "naics_base": pd.Series(naics_base, index=pd.RangeIndex(len(naics_base))),
        "naics_deltas": naics_deltas,
        "zip_base": pd.Series(zip_base, index=pd.RangeIndex(len(zip_base))),
        "zip_deltas": zip_deltas,
    }
    return effects
