# tests/unit/test_eval_ranking.py
from __future__ import annotations

import numpy as np


from hierarchical_naics_model.eval.ranking import ranking_report
import pytest


@pytest.fixture
def toy_data():
    # Toy data where top probabilities align with positives
    y = np.array([0, 1, 0, 1, 1, 0, 0, 0], int)
    p = np.array([0.1, 0.9, 0.2, 0.85, 0.8, 0.3, 0.05, 0.4], float)
    ks = (5, 25, 50, 100)
    rep = ranking_report(y, p, ks=ks)
    df = rep["summary"].sort_values("k_pct")
    base = rep["base_rate"]
    return {"y": y, "p": p, "ks": ks, "rep": rep, "df": df, "base": base}


@pytest.mark.parametrize("k_pct,expected_k_count", [(25, 2), (100, 8)])
def test_ranking_report_k_count_matches_expected(toy_data, k_pct, expected_k_count):
    row = toy_data["df"][toy_data["df"].k_pct == k_pct].iloc[0]
    assert row.k_count == expected_k_count, (
        f"Expected {expected_k_count} at {k_pct}, got {row.k_count}"
    )


def test_ranking_report_base_rate_between_zero_and_one(toy_data):
    base = toy_data["base"]
    assert 0 < base < 1, f"Expected base rate between 0 and 1, got {base}"


def test_ranking_report_precision_at_25pct_is_one(toy_data):
    row25 = toy_data["df"][toy_data["df"].k_pct == 25].iloc[0]
    assert abs(row25.precision - 1.0) < 1e-12, (
        f"Expected precision at 25% to be 1.0, got {row25.precision}"
    )


def test_ranking_report_lift_at_25pct_greater_than_one(toy_data):
    row25 = toy_data["df"][toy_data["df"].k_pct == 25].iloc[0]
    assert row25.lift > 1.0, (
        f"Expected lift at 25% to be greater than 1.0, got {row25.lift}"
    )


def test_ranking_report_precision_at_100pct_equals_base_rate(toy_data):
    row100 = toy_data["df"][toy_data["df"].k_pct == 100].iloc[0]
    assert abs(row100.precision - toy_data["base"]) < 1e-12, (
        f"Expected precision at 100% to be equal to base rate, got {row100.precision}"
    )


@pytest.mark.parametrize(
    "y,p,ks,expected_precision",
    [
        (np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]), (50,), [0.0]),
        (np.array([1, 1, 1, 1]), np.array([0.9, 0.8, 0.7, 0.6]), (50,), [1.0]),
        (np.array([0, 1]), np.array([0.5, 0.5]), (100,), [0.5]),
    ],
)
def test_ranking_report_edge_case_precisions(y, p, ks, expected_precision):
    rep = ranking_report(y, p, ks=ks)
    df = rep["summary"].sort_values("k_pct")
    for idx, exp in enumerate(expected_precision):
        assert abs(df.iloc[idx].precision - exp) < 1e-12, (
            f"Expected precision {exp}, got {df.iloc[idx].precision}"
        )


def test_ranking_report_empty_dataset():
    rep = ranking_report(np.array([], int), np.array([], float), ks=(10,))
    assert rep["summary"].empty
    assert np.isnan(rep["base_rate"])


def test_ranking_report_validates_shapes():
    y = np.array([0, 1, 0])
    p = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        ranking_report(y, p, ks=(10,))


def test_ranking_report_zero_percent_threshold():
    rep = ranking_report(np.array([1, 0], int), np.array([0.9, 0.1], float), ks=(0,))
    df = rep["summary"]
    assert df.iloc[0].k_count == 0
    assert df.iloc[0].precision == 0.0


def test_ranking_report_zero_base_rate_branch():
    y = np.zeros(5, int)
    p = np.linspace(0, 1, 5)
    rep = ranking_report(y, p, ks=(10,))
    df = rep["summary"]
    assert df.iloc[0].lift == 0.0
