import numpy as np
import pandas as pd
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)
from hierarchical_naics_model.calibration_and_lift_report import (
    calibration_and_lift_report,
)


@log_test_performance
def test_calibration_and_lift_report_basic(test_run_id):
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


def test_calibration_and_lift_report_all_zeros():
    y_true = np.zeros(10, dtype=int)
    p_pred = np.linspace(0, 1, 10)
    out = calibration_and_lift_report(y_true, p_pred, bins=5)
    assert out["summary"]["base_rate"] == 0.0
    assert np.isnan(out["ranking"]["lift_at_k"]).all()  # lift is nan if base_rate is 0


def test_calibration_and_lift_report_all_ones():
    y_true = np.ones(10, dtype=int)
    p_pred = np.linspace(0, 1, 10)
    out = calibration_and_lift_report(y_true, p_pred, bins=5)
    assert out["summary"]["base_rate"] == 1.0
    assert np.isfinite(out["ranking"]["lift_at_k"]).all()


def test_calibration_and_lift_report_empty():
    import pytest

    with pytest.raises(ValueError):
        calibration_and_lift_report(np.array([]), np.array([]), bins=3)
