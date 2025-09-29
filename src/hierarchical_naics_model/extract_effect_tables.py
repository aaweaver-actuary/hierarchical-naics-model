# hierarchical_conversion_model/scoring.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def extract_effect_tables(
    idata,
) -> Dict[str, object]:
    """
    Reduce posterior samples to posterior means for nested-delta scoring.

    Expected vars in `idata.posterior`:
      - beta0
      - naics_base (vector at level 0)
      - naics_delta_{j} for j>=1 (vectors)
      - zip_base
      - zip_delta_{m} for m>=1
    """
    post = idata.posterior

    beta0 = float(post["beta0"].mean(dim=("chain", "draw")).values)

    def _series(name: str) -> pd.Series:
        vec = post[name].mean(dim=("chain", "draw")).values
        return pd.Series(vec, index=np.arange(vec.shape[0]), name=name)

    naics_base = _series("naics_base")
    naics_deltas: List[pd.Series] = []
    for v in sorted(
        [v for v in post.data_vars if str(v).startswith("naics_delta_")],
        key=lambda s: int(str(s).split("_")[-1]),
    ):
        naics_deltas.append(_series(str(v)))

    zip_base = _series("zip_base")
    zip_deltas: List[pd.Series] = []
    for v in sorted(
        [v for v in post.data_vars if str(v).startswith("zip_delta_")],
        key=lambda s: int(str(s).split("_")[-1]),
    ):
        zip_deltas.append(_series(str(v)))

    return {
        "beta0": beta0,
        "naics_base": naics_base,
        "naics_deltas": naics_deltas,
        "zip_base": zip_base,
        "zip_deltas": zip_deltas,
        "naics_level_names": [f"N{j}" for j in range(len(naics_deltas) + 1)],
        "zip_level_names": [f"Z{m}" for m in range(len(zip_deltas) + 1)],
    }
