from __future__ import annotations

from typing import Dict, List

import numpy as np
import xarray as xr


__all__ = ["extract_effect_tables_nested"]


def _posterior_mean_1d(da: xr.DataArray, dim=("chain", "draw")) -> np.ndarray:
    """Return posterior mean as a 1-D numpy array preserving coordinate order."""

    mean_da = da.mean(dim=dim)
    return np.asarray(mean_da.values, dtype=float)


def _collect_sorted_deltas(posterior: xr.Dataset, prefix: str) -> List[str]:
    """Collect delta variable names like ``{prefix}_delta_<k>`` sorted by ``k``."""

    candidate = [
        name for name in posterior.data_vars if str(name).startswith(f"{prefix}_delta_")
    ]

    def take_index(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:  # noqa: BLE001 - defensive, fall back to stable ordering
            return 10**9

    return [name for name in sorted(candidate, key=take_index)]


def extract_effect_tables_nested(idata) -> Dict[str, object]:
    """Extract posterior means for nested delta scoring without pandas dependencies."""

    if xr is None:  # pragma: no cover - sanity guard for optional dependency
        raise RuntimeError("xarray/arviz is required to extract posterior tables.")

    if "posterior" not in idata.groups():
        raise ValueError("InferenceData does not contain a 'posterior' group.")

    posterior = idata.posterior

    if "beta0" not in posterior:
        raise ValueError("posterior is missing variable 'beta0'.")
    if "naics_base" not in posterior:
        raise ValueError("posterior is missing variable 'naics_base'.")
    if "zip_base" not in posterior:
        raise ValueError("posterior is missing variable 'zip_base'.")

    beta0 = float(posterior["beta0"].mean(("chain", "draw")).values)
    naics_base = _posterior_mean_1d(posterior["naics_base"])
    zip_base = _posterior_mean_1d(posterior["zip_base"])

    naics_delta_names = _collect_sorted_deltas(posterior, "naics")
    zip_delta_names = _collect_sorted_deltas(posterior, "zip")

    naics_deltas = [_posterior_mean_1d(posterior[name]) for name in naics_delta_names]
    zip_deltas = [_posterior_mean_1d(posterior[name]) for name in zip_delta_names]

    return {
        "beta0": beta0,
        "naics_base": naics_base,
        "naics_deltas": naics_deltas,
        "zip_base": zip_base,
        "zip_deltas": zip_deltas,
    }
