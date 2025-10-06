from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import xarray as xr

import arviz as az

from hierarchical_naics_model.reporting.dashboard import (
    _as_float,
    _build_decision_flow,
    _build_inference_suggestions,
    _build_variable_payload,
    _extract_loo_stats,
    _format_float,
)


def test_as_float_handles_iterables_and_errors():
    assert _as_float([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert math.isnan(_as_float(np.array([], dtype=float)))

    class NonNumeric:
        def __float__(self) -> float:  # pragma: no cover - exercised via _as_float
            raise ValueError("not convertible")

    assert math.isnan(_as_float(NonNumeric()))


def test_format_float_handles_value_error():
    assert _format_float("not-a-number") == "NA"


def test_extract_loo_stats_with_missing_and_bad_pareto():
    class BadPareto:
        def __array__(self, *_args, **_kwargs):
            raise ValueError("cannot convert")

    class DummyLoo:
        elpd_loo = 12.3
        loo_se = 1.1
        pareto_k = BadPareto()

    stats = _extract_loo_stats(DummyLoo())
    assert stats["loo"] == pytest.approx(12.3)
    assert math.isnan(stats["pareto_k_max"])


def test_build_decision_flow_indicator_when_no_values():
    fig, flow = _build_decision_flow([("Component", 0.5, 0.0)])
    assert flow["values"] == []
    assert fig.data
    assert fig.data[0].type == "indicator"


def test_build_inference_suggestions_edge_cases():
    empty = pl.DataFrame({})
    fallback = _build_inference_suggestions(empty)
    assert fallback == {"naics": [], "zip": []}

    df = pl.DataFrame(
        {
            "p": [0.6, 0.7, 0.5],
            "NAICS": ["52", None, None],
            "ZIP": [None, None, None],
            "eta": [0.4, 0.5, 0.1],
            "any_backoff": [True, False, True],
        }
    )
    suggestions = _build_inference_suggestions(df, top_k=2)
    assert len(suggestions["naics"]) == 1
    assert suggestions["naics"][0]["code"] == "52"
    assert "avg_eta" in suggestions["naics"][0]
    assert "backoff_rate" in suggestions["naics"][0]
    # ZIP column has no non-null entries, so it should be empty
    assert suggestions["zip"] == []


def test_build_variable_payload_handles_nan_prior():
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float),
            )
        }
    )
    prior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array([[np.nan, np.nan, np.nan]], dtype=float),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior, prior=prior)

    payload = _build_variable_payload(idata, max_variables=1, max_samples=5)
    assert payload["variables"], "Expected theta variable entry"
    entry = payload["variables"][0]
    assert entry["prior"]["sd"] == pytest.approx(1.0)
    assert entry["trace"]["total_draws"] == 3
    assert len(entry["trace"]["chains"]) == 2
    assert entry["trace"]["draw_indices"] == [0, 1, 2]


def test_build_variable_payload_autocorrelation_edge_cases():
    posterior_const = xr.Dataset(
        {
            "beta0": (
                ("chain", "draw"),
                np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
            )
        }
    )
    idata_const = az.InferenceData(posterior=posterior_const)

    payload_const = _build_variable_payload(idata_const, max_variables=1, max_samples=5)
    beta_entry = payload_const["variables"][0]
    beta_autocorr = beta_entry["autocorrelation"]
    assert beta_autocorr["lags"] == [0, 1]
    assert beta_autocorr["per_chain"][0]["values"] == [1.0, 0.0]
    assert isinstance(beta_autocorr["interpretation"], str)

    posterior_single = xr.Dataset(
        {
            "gamma": (
                ("chain", "draw"),
                np.array([[1.0], [np.nan]], dtype=float),
            )
        }
    )
    idata_single = az.InferenceData(posterior=posterior_single)

    payload_single = _build_variable_payload(
        idata_single, max_variables=1, max_samples=5
    )
    gamma_entry = payload_single["variables"][0]
    gamma_autocorr = gamma_entry["autocorrelation"]
    assert gamma_autocorr["lags"] == [0]
    assert gamma_autocorr["per_chain"]
    for series in gamma_autocorr["per_chain"]:
        assert series["values"][0] == 1.0


def test_build_variable_payload_ppc_without_observed_data():
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float),
            )
        }
    )
    predictive = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]], dtype=float),
            )
        }
    )
    idata = az.InferenceData(posterior=posterior, posterior_predictive=predictive)

    payload = _build_variable_payload(idata, max_variables=1, max_samples=5)
    entry = payload["variables"][0]
    ppc = entry["ppc"]
    assert ppc is not None
    assert ppc["observed"]["samples"] == []
    assert ppc["x_range"][0] < ppc["x_range"][1]


def test_build_variable_payload_ppc_with_indexed_variable():
    posterior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "idx"),
                np.array(
                    [
                        [[0.1, 0.2], [0.11, 0.21]],
                        [[0.09, 0.19], [0.12, 0.22]],
                    ],
                    dtype=float,
                ),
            )
        }
    )
    predictive = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "idx"),
                np.array(
                    [
                        [[0.08, 0.18], [0.1, 0.2]],
                        [[0.085, 0.185], [0.105, 0.205]],
                    ],
                    dtype=float,
                ),
            )
        }
    )
    observed = xr.Dataset(
        {
            "weights": (
                ("idx",),
                np.array([0.082, 0.196], dtype=float),
            )
        }
    )
    idata = az.InferenceData(
        posterior=posterior,
        posterior_predictive=predictive,
        observed_data=observed,
    )

    payload = _build_variable_payload(idata, max_variables=4, max_samples=5)
    indexed_entry = next(
        item for item in payload["variables"] if item["id"].endswith("[1]")
    )
    ppc = indexed_entry["ppc"]
    assert ppc is not None
    assert ppc["observed"]["samples"][0] == pytest.approx(0.196)
    assert len(ppc["kde"]["x"]) == len(ppc["kde"]["y"])


def test_build_variable_payload_ppc_shape_mismatch_recovers():
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw", "idx"),
                np.ones((1, 3, 2), dtype=float),
            )
        }
    )
    predictive = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.ones((1, 3), dtype=float),
            )
        }
    )
    observed = xr.Dataset({"theta": ((), np.array(1.0, dtype=float))})

    idata = az.InferenceData(
        posterior=posterior,
        posterior_predictive=predictive,
        observed_data=observed,
    )

    payload = _build_variable_payload(idata, max_variables=2, max_samples=5)
    entry = next(item for item in payload["variables"] if item["id"].endswith("[0]"))
    ppc = entry["ppc"]

    assert ppc is not None
    assert ppc["sample_count"] == 3
    assert ppc["observed"]["samples"]
    assert ppc["observed"]["samples"][0] == pytest.approx(1.0)
    lower, upper = ppc["x_range"]
    assert lower < 1.0 < upper
