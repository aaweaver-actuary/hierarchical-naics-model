# tests/unit/test_scoring_predict.py
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hierarchical_naics_model.scoring.predict import predict_proba_nested


def _tiny_maps_and_effects():
    naics_cut = [2, 3]
    naics_maps = [
        {"52": 0, "31": 1},
        {"520": 0, "521": 1, "311": 2},
    ]
    zip_cut = [2]
    zip_maps = [{"45": 0, "46": 1}]
    effects = {
        "beta0": -1.0,
        "naics_base": np.array([0.2, -0.1], dtype=float),
        "naics_deltas": [np.array([0.05, 0.00, -0.05], dtype=float)],
        "zip_base": np.array([0.1, -0.1], dtype=float),
        "zip_deltas": [],
    }
    return naics_cut, zip_cut, naics_maps, zip_maps, effects


@pytest.fixture
def tiny_maps_and_effects():
    return _tiny_maps_and_effects()


@pytest.mark.parametrize(
    "naics,zip,expected_eta,expected_p_range,expected_naics_L2_known,expected_naics_L3_known,expected_zip_L2_known,expected_any_backoff",
    [
        # All codes known, no backoff
        ("521", "46", -1.0 + 0.2 + 0.0 - 0.1, (0.0, 1.0), True, True, True, False),
        # Unknown leaf delta, parent not used
        ("522130", "45", -1.0 + 0.2 + 0.0 + 0.1, (0.0, 1.0), True, False, True, True),
        # All codes unknown, only beta0
        ("99", "00", -1.0, (0.0, 1.0), False, False, False, True),
    ],
)
def test_predict_proba_nested_eta(
    tiny_maps_and_effects,
    naics,
    zip,
    expected_eta,
    expected_p_range,
    expected_naics_L2_known,
    expected_naics_L3_known,
    expected_zip_L2_known,
    expected_any_backoff,
):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert abs(out["eta"][0] - expected_eta) < 1e-12


@pytest.mark.parametrize(
    "naics,zip,expected_p_range",
    [
        ("521", "46", (0.0, 1.0)),
        ("522130", "45", (0.0, 1.0)),
        ("99", "00", (0.0, 1.0)),
    ],
)
def test_predict_proba_nested_probability_in_range(
    tiny_maps_and_effects, naics, zip, expected_p_range
):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert expected_p_range[0] < out["p"][0] < expected_p_range[1]


@pytest.mark.parametrize(
    "naics,zip,expected",
    [
        ("521", "46", True),
        ("522130", "45", True),
        ("99", "00", False),
    ],
)
def test_predict_proba_nested_naics_L2_known(
    tiny_maps_and_effects, naics, zip, expected
):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert out["naics_L2_known"][0] == expected


@pytest.mark.parametrize(
    "naics,zip,expected",
    [
        ("521", "46", True),
        ("522130", "45", False),
        ("99", "00", False),
    ],
)
def test_predict_proba_nested_naics_L3_known(
    tiny_maps_and_effects, naics, zip, expected
):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert out["naics_L3_known"][0] == expected


@pytest.mark.parametrize(
    "naics,zip,expected",
    [
        ("521", "46", True),
        ("522130", "45", True),
        ("99", "00", False),
    ],
)
def test_predict_proba_nested_zip_L2_known(tiny_maps_and_effects, naics, zip, expected):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert out["zip_L2_known"][0] == expected


@pytest.mark.parametrize(
    "naics,zip,expected",
    [
        ("521", "46", False),
        ("522130", "45", True),
        ("99", "00", True),
    ],
)
def test_predict_proba_nested_any_backoff(tiny_maps_and_effects, naics, zip, expected):
    naics_cut, zip_cut, naics_maps, zip_maps, effects = tiny_maps_and_effects
    df = pl.LazyFrame({"naics": [naics], "zip": [zip]})
    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )
    assert out["any_backoff"][0] == expected


def test_predict_proba_nested_multiple_zip_levels():
    df = pl.LazyFrame({"naics": ["52"], "zip": ["123"]})
    naics_cut = [2]
    zip_cut = [2, 3]
    naics_maps = [{"52": 0}]
    zip_maps = [{"12": 0}, {"123": 0}]
    effects = {
        "beta0": 0.0,
        "naics_base": np.array([0.0], dtype=float),
        "naics_deltas": [],
        "zip_base": np.array([0.1], dtype=float),
        "zip_deltas": [np.array([0.05], dtype=float)],
    }

    out = predict_proba_nested(
        df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=naics_cut,
        zip_cut_points=zip_cut,
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
        prefix_fill="0",
        return_components=True,
    )

    assert abs(out["eta"][0] - 0.15) < 1e-12
    assert bool(out["zip_L3_known"][0]) is True
