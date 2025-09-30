import pandas as pd
import numpy as np
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)
from hierarchical_naics_model.predict_proba import predict_proba


@log_test_performance
def test_predict_proba_basic(test_run_id):
    df = pd.DataFrame(
        {
            "naics": ["100", "200", "300"],
            "zip": ["111", "222", "333"],
        }
    )
    effects = {
        "beta0": 0.0,
        "naics_base": pd.Series([0.1, -0.1, 0.0], index=[0, 1, 2]),
        "naics_deltas": [
            pd.Series([0.05, -0.05, 0.0], index=[0, 1, 2]),
            pd.Series([0.02, -0.02, 0.0], index=[0, 1, 2]),
        ],
        "zip_base": pd.Series([0.2, -0.2, 0.0], index=[0, 1, 2]),
        "zip_deltas": [
            pd.Series([0.01, -0.01, 0.0], index=[0, 1, 2]),
            pd.Series([0.03, -0.03, 0.0], index=[0, 1, 2]),
        ],
    }
    # Simple level maps: all codes map to index 0, 1, or 2
    naics_level_maps = [
        {"1": 0, "2": 1, "3": 2},  # level 1
        {"10": 0, "20": 1, "30": 2},  # level 2
        {"100": 0, "200": 1, "300": 2},  # level 3
    ]
    zip_level_maps = [
        {"1": 0, "2": 1, "3": 2},  # level 1
        {"11": 0, "22": 1, "33": 2},  # level 2
        {"111": 0, "222": 1, "333": 2},  # level 3
    ]
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert "p" in out
    assert "eta" in out
    assert np.isfinite(out["p"]).all()
    assert np.isfinite(out["eta"]).all()
