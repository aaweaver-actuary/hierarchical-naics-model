import numpy as np
import pandas as pd
import pytest
from hierarchical_naics_model.calibration_and_lift_report import (
    calibration_and_lift_report,
)


def test_calibration_and_lift_report_basic():
    # Simple binary outcome and predicted probabilities
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0, 1, 0, 1, 0],
            "p_pred": [0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.4],
        }
    )
    out = calibration_and_lift_report(df["y_true"].values, df["p_pred"].values, bins=4)
    rel = out["reliability"]
    rank = out["ranking"]
    assert "bin" in rel.columns
    assert "lift_at_k" in rank.columns
    assert len(rel) == 4
    # Check monotonicity of mean predicted probability by bin
    assert rel["p_avg"].is_monotonic_increasing or rel["p_avg"].is_monotonic_decreasing
    # Check lift values are finite
    assert np.isfinite(rank["lift_at_k"]).all()
