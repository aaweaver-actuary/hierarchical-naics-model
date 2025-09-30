from __future__ import annotations

from typing import Dict

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None  # type: ignore


def extract_effect_tables_nested(idata) -> Dict[str, object]:
    """
    Reduce posterior samples to posterior means for nested-deltas scoring.

    Parameters
    ----------
    idata
        ArviZ InferenceData with variables:
        - beta0
        - naics_base, naics_delta_{j>=1}
        - zip_base, zip_delta_{m>=1}

    Returns
    -------
    dict
        {
          "beta0": float,
          "naics_base": pd.Series,
          "naics_deltas": list[pd.Series],
          "zip_base": pd.Series,
          "zip_deltas": list[pd.Series],
        }
    """
    if xr is None:
        raise RuntimeError("xarray/arviz not available.")
    # TODO: implement (mean over chain,draw; construct Series with int index).
    raise NotImplementedError
