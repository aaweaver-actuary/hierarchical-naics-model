# hierarchical_conversion_model/scoring.py
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def extract_effect_tables(
    idata,
    *,
    naics_level_names: Optional[List[str]] = None,
    zip_level_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Reduce posterior samples to posterior means for nested-delta scoring.

        Expected vars in `idata.posterior` (supports both naming schemes):
            - Modern nested-delta names: naics_base, naics_delta_{j}, zip_base, zip_delta_{m}
            - Or flat names: naics_eff_{j}, zip_eff_{m}
    """

    post = idata.posterior

    def _extract_base_and_deltas(post, base_name, delta_prefix, eff_prefix):
        tables = []
        if hasattr(post, "data_vars") and base_name in post.data_vars:
            tables.append(_series(base_name))
            delta_vars = sorted(
                [v for v in post.data_vars if str(v).startswith(delta_prefix)],
                key=lambda s: int(str(s).split("_")[-1]),
            )
            tables.extend([_series(str(v)) for v in delta_vars])
        elif hasattr(post, "data_vars"):
            eff_vars = sorted(
                [v for v in post.data_vars if str(v).startswith(eff_prefix)],
                key=lambda s: int(str(s).split("_")[-1]),
            )
            tables.extend([_series(str(v)) for v in eff_vars])
        return tables

    def _series(name: str) -> pd.Series:
        try:
            vec = post[name].mean(dim=("chain", "draw")).values
            return pd.Series(vec, index=np.arange(vec.shape[0]), name=name)
        except Exception:
            return pd.Series([], name=name)

    # Defensive: handle missing beta0 and empty posterior
    try:
        arr = post["beta0"] if "beta0" in getattr(post, "data_vars", {}) else None
        if arr is not None:
            dims = set(getattr(arr, "dims", []))
            if "chain" in dims and "draw" in dims:
                beta0 = float(arr.mean(dim=("chain", "draw")).values)
            else:
                beta0 = float(np.mean(arr.values))
        else:
            beta0 = 0.0
    except Exception:
        beta0 = 0.0

    naics_tables = _extract_base_and_deltas(
        post, "naics_base", "naics_delta_", "naics_eff_"
    )
    zip_tables = _extract_base_and_deltas(post, "zip_base", "zip_delta_", "zip_eff_")

    if naics_level_names is None:
        naics_level_names = [f"NAICS_L{j}" for j in range(len(naics_tables))]
    if zip_level_names is None:
        zip_level_names = [f"ZIP_L{m}" for m in range(len(zip_tables))]

    return {
        "beta0": beta0,
        "naics_tables": naics_tables,
        "zip_tables": zip_tables,
        "naics_level_names": naics_level_names,
        "zip_level_names": zip_level_names,
    }
