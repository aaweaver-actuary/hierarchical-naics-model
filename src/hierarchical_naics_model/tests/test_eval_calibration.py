# tests/unit/test_eval_calibration.py
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hierarchical_naics_model.eval.calibration import calibration_report


@pytest.fixture
def basic_calibration_data():
    y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    p = np.array([0.1, 0.8, 0.4, 0.7, 0.9, 0.2], dtype=float)
    rep = calibration_report(y, p, bins=5)
    return rep


@pytest.mark.parametrize("col", ["bin_low", "bin_high", "n", "mean_p", "mean_y", "gap"])
def test_calibration_report_reliability_dataframe_has_expected_columns(
    basic_calibration_data, col
):
    df = basic_calibration_data["reliability"]
    assert col in df.columns


def test_calibration_report_expected_calibration_error_in_valid_range(
    basic_calibration_data,
):
    assert 0 <= basic_calibration_data["ece"] <= 1


def test_calibration_report_brier_score_non_negative(basic_calibration_data):
    assert basic_calibration_data["brier"] >= 0


def test_calibration_report_log_loss_non_negative(basic_calibration_data):
    assert basic_calibration_data["log_loss"] >= 0


def test_calibration_report_reliability_dataframe_has_correct_number_of_bins(
    basic_calibration_data,
):
    df = basic_calibration_data["reliability"]
    assert len(df) == 5


@pytest.mark.parametrize(
    "y, p, expected_brier",
    [
        (np.zeros(10, int), np.zeros(10, float), 0.0),
        (np.ones(10, int), np.ones(10, float), 0.0),
    ],
)
def test_calibration_report_degenerate_cases_brier_score(y, p, expected_brier):
    rep = calibration_report(y, p, bins=4)
    assert rep["brier"] == expected_brier


@pytest.mark.parametrize(
    "y, p",
    [
        (np.zeros(10, int), np.zeros(10, float)),
        (np.ones(10, int), np.ones(10, float)),
    ],
)
def test_calibration_report_degenerate_cases_log_loss_finite(y, p):
    rep = calibration_report(y, p, bins=4)
    assert np.isfinite(rep["log_loss"])


def test_calibration_report_all_zero_or_one_degenerate():
    y = np.zeros(10, int)
    p = np.zeros(10, float)
    rep0 = calibration_report(y, p, bins=4)
    assert rep0["brier"] == 0.0
    y1 = np.ones(10, int)
    p1 = np.ones(10, float)
    rep1 = calibration_report(y1, p1, bins=4)
    assert rep1["brier"] == 0.0
    # log_loss finite due to clipping
    assert np.isfinite(rep0["log_loss"])
    assert np.isfinite(rep1["log_loss"])


def test_calibration_report_validates_shapes():
    y = np.array([0, 1, 1])
    p = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        calibration_report(y, p, bins=3)


def test_calibration_report_handles_empty_input():
    rep = calibration_report(np.array([], dtype=int), np.array([], dtype=float), bins=3)
    reliability = rep["reliability"]
    assert isinstance(reliability, pl.DataFrame)
    assert reliability.is_empty()
    assert np.isnan(rep["ece"]) and np.isnan(rep["brier"])
