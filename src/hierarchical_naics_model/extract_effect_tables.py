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

    # Defensive: handle missing beta0 and empty posterior
    try:
        if "beta0" in getattr(post, "data_vars", {}):
            arr = post["beta0"]
            # If chain/draw dims exist, take mean; else, just take value
            dims = set(getattr(arr, "dims", []))
            if "chain" in dims and "draw" in dims:
                beta0 = float(arr.mean(dim=("chain", "draw")).values)
            else:
                beta0 = float(np.mean(arr.values))
        else:
            beta0 = 0.0
    except Exception:
        beta0 = 0.0

    def _series(name: str) -> pd.Series:
        try:
            vec = post[name].mean(dim=("chain", "draw")).values
            return pd.Series(vec, index=np.arange(vec.shape[0]), name=name)
        except Exception:
            return pd.Series([], name=name)

    naics_tables: List[pd.Series] = []
    zip_tables: List[pd.Series] = []

    if hasattr(post, "data_vars") and "naics_base" in post.data_vars:
        naics_tables.append(_series("naics_base"))
        for v in sorted(
            [v for v in post.data_vars if str(v).startswith("naics_delta_")],
            key=lambda s: int(str(s).split("_")[-1]),
        ):
            naics_tables.append(_series(str(v)))
    elif hasattr(post, "data_vars"):
        # Fallback to flat names naics_eff_{j}
        eff_vars = sorted(
            [v for v in post.data_vars if str(v).startswith("naics_eff_")],
            key=lambda s: int(str(s).split("_")[-1]),
        )
        for v in eff_vars:
            naics_tables.append(_series(str(v)))

    if hasattr(post, "data_vars") and "zip_base" in post.data_vars:
        zip_tables.append(_series("zip_base"))
        for v in sorted(
            [v for v in post.data_vars if str(v).startswith("zip_delta_")],
            key=lambda s: int(str(s).split("_")[-1]),
        ):
            zip_tables.append(_series(str(v)))
    elif hasattr(post, "data_vars"):
        eff_vars = sorted(
            [v for v in post.data_vars if str(v).startswith("zip_eff_")],
            key=lambda s: int(str(s).split("_")[-1]),
        )
        for v in eff_vars:
            zip_tables.append(_series(str(v)))

    # Default names if not provided
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
