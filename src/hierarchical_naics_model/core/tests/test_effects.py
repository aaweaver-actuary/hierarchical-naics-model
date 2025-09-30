# tests/test_core_effects.py
from __future__ import annotations

import numpy as np
import pytest

from hierarchical_naics_model.core.effects import eta_additive, eta_nested


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def small_indices():
    # N=3 obs
    naics_levels = np.array(
        [
            [0, 1],  # uses NAICS L0 idx 0, L1 idx 1
            [1, 2],
            [0, 0],
        ],
        dtype=int,
    )
    zip_levels = np.array(
        [
            [0, 1],  # ZIP L0 idx 0, L1 idx 1
            [0, 0],
            [0, 1],
        ],
        dtype=int,
    )
    return naics_levels, zip_levels


@pytest.fixture
def additive_effects():
    # L_naics=2: K0=2, K1=3; L_zip=2: K0=1, K1=2
    naics_effects = [
        np.array([0.10, -0.20], dtype=float),  # L0
        np.array([0.05, 0.00, -0.05], dtype=float),  # L1
    ]
    zip_effects = [
        np.array([0.20], dtype=float),  # L0
        np.array([-0.10, 0.00], dtype=float),  # L1
    ]
    return naics_effects, zip_effects


@pytest.fixture
def nested_effects():
    # NAICS J=2, ZIP M=2
    naics_base = np.array([0.10, -0.10], dtype=float)  # K0=2
    naics_deltas = [np.array([0.00, 0.05, -0.05], dtype=float)]  # K1=3
    zip_base = np.array([0.20], dtype=float)  # K0=1
    zip_deltas = [np.array([-0.10, 0.00], dtype=float)]  # K1=2
    return naics_base, naics_deltas, zip_base, zip_deltas


# -----------------------
# eta_additive: happy path
# -----------------------


def test_eta_additive_happy(small_indices, additive_effects):
    beta0 = -1.0
    naics_levels, zip_levels = small_indices
    naics_effects, zip_effects = additive_effects

    eta = eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)
    assert eta.shape == (naics_levels.shape[0],)
    assert eta.dtype == np.float64

    # Manual check for obs 0: beta0 + NAICS(L0:0.10) + NAICS(L1:0.00) + ZIP(L0:0.20) + ZIP(L1:0.00)
    expected0 = -1.0 + 0.10 + 0.00 + 0.20 + 0.00
    # obs 1: beta0 + NAICS(L0:-0.20) + NAICS(L1:-0.05) + ZIP(L0:0.20) + ZIP(L1:-0.10)
    expected1 = -1.0 + (-0.20) + (-0.05) + 0.20 + (-0.10)
    # obs 2: beta0 + NAICS(L0:0.10) + NAICS(L1:0.05) + ZIP(L0:0.20) + ZIP(L1:0.00)
    expected2 = -1.0 + 0.10 + 0.05 + 0.20 + 0.00

    np.testing.assert_allclose(
        eta,
        np.array([expected0, expected1, expected2], dtype=float),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("N,L_naics,L_zip", [(5, 0, 2), (5, 2, 0), (3, 0, 0)])
def test_eta_additive_handles_zero_levels(N, L_naics, L_zip):
    beta0 = -0.3
    naics_levels = (
        np.empty((N, L_naics), dtype=int)
        if L_naics > 0
        else np.empty((N, 0), dtype=int)
    )
    zip_levels = (
        np.empty((N, L_zip), dtype=int) if L_zip > 0 else np.empty((N, 0), dtype=int)
    )
    naics_effects = (
        [np.array([], dtype=float) for _ in range(L_naics)] if L_naics > 0 else []
    )
    zip_effects = [np.array([], dtype=float) for _ in range(L_zip)] if L_zip > 0 else []
    out = eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)
    np.testing.assert_allclose(out, np.full(N, beta0, dtype=float))


# -----------------------
# eta_additive: errors
# -----------------------


def test_eta_additive_bad_levels_shapes():
    beta0 = 0.0
    naics_effects = [np.array([0.0], dtype=float)]
    zip_effects = [np.array([0.0], dtype=float)]
    with pytest.raises(ValueError):
        eta_additive(
            beta0,
            naics_effects,
            zip_effects,
            np.array([0, 1], dtype=int),
            np.zeros((1, 1), int),
        )
    with pytest.raises(ValueError):
        eta_additive(
            beta0,
            naics_effects,
            zip_effects,
            np.zeros((1, 1), int),
            np.array([0, 1], dtype=int),
        )


def test_eta_additive_effects_length_mismatch():
    beta0 = 0.0
    naics_levels = np.zeros((2, 2), dtype=int)
    zip_levels = np.zeros((2, 1), dtype=int)
    naics_effects = [np.array([0.0], dtype=float)]  # length 1 but L_naics=2
    zip_effects = [np.array([0.0], dtype=float)]
    with pytest.raises(ValueError):
        eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)


def test_eta_additive_non_float_effects_raise():
    beta0 = 0.0
    naics_levels = np.zeros((2, 1), dtype=int)
    zip_levels = np.zeros((2, 1), dtype=int)
    naics_effects = [np.array([0], dtype=int)]  # not float
    zip_effects = [np.array([0.0], dtype=float)]
    with pytest.raises(ValueError):
        eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)


def test_eta_additive_out_of_bounds_indices_raise():
    beta0 = 0.0
    naics_levels = np.array([[1]], dtype=int)  # index 1 but only length 1 -> OOB
    zip_levels = np.array([[0]], dtype=int)
    naics_effects = [np.array([0.1], dtype=float)]
    zip_effects = [np.array([0.2], dtype=float)]
    with pytest.raises(ValueError):
        eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)


# -----------------------
# eta_nested: happy path
# -----------------------


def test_eta_nested_happy(small_indices, nested_effects):
    beta0 = -1.0
    naics_levels, zip_levels = small_indices
    naics_base, naics_deltas, zip_base, zip_deltas = nested_effects

    eta = eta_nested(
        beta0,
        naics_base,
        naics_deltas,
        zip_base,
        zip_deltas,
        naics_levels,
        zip_levels,
    )
    assert eta.shape == (naics_levels.shape[0],)
    assert eta.dtype == np.float64

    # Manual check obs 0:
    # beta0 + NAICS_base[0]=0.10 + NAICS_delta1[1]=0.05 + ZIP_base[0]=0.20 + ZIP_delta1[1]=0.00
    expected0 = -1.0 + 0.10 + 0.05 + 0.20 + 0.00
    # obs 1: NAICS_base[1]=-0.10 + delta1[2]=-0.05 + ZIP_base[0]=0.20 + delta1[0]=-0.10
    expected1 = -1.0 + (-0.10) + (-0.05) + 0.20 + (-0.10)
    # obs 2: NAICS_base[0]=0.10 + delta1[0]=0.00 + ZIP_base[0]=0.20 + delta1[1]=0.00
    expected2 = -1.0 + 0.10 + 0.00 + 0.20 + 0.00

    np.testing.assert_allclose(
        eta,
        np.array([expected0, expected1, expected2], dtype=float),
        rtol=0,
        atol=1e-12,
    )


# -----------------------
# eta_nested: errors
# -----------------------


def test_eta_nested_requires_at_least_one_level_each():
    beta0 = 0.0
    naics_base = np.array([0.1], dtype=float)
    zip_base = np.array([0.2], dtype=float)
    # J=0 or M=0
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            naics_base,
            [],
            zip_base,
            [],
            np.empty((2, 0), int),
            np.zeros((2, 1), int),
        )
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            naics_base,
            [],
            zip_base,
            [],
            np.zeros((2, 1), int),
            np.empty((2, 0), int),
        )


def test_eta_nested_base_must_be_float_and_1d():
    beta0 = 0.0
    naics_levels = np.zeros((2, 1), int)
    zip_levels = np.zeros((2, 1), int)
    # non-float dtype
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            np.array([0], int),
            [],
            np.array([0.0], float),
            [],
            naics_levels,
            zip_levels,
        )
    # not 1-D
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            np.array([[0.0]], float),
            [],
            np.array([0.0], float),
            [],
            naics_levels,
            zip_levels,
        )


def test_eta_nested_deltas_length_mismatch():
    beta0 = 0.0
    naics_levels = np.zeros((2, 2), int)  # J=2 => need len(naics_deltas)=1
    zip_levels = np.zeros((2, 1), int)  # M=1 => need len(zip_deltas)=0
    naics_base = np.array([0.1], float)
    zip_base = np.array([0.2], float)
    naics_deltas = []  # wrong (should be 1)
    zip_deltas = []  # correct
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )


def test_eta_nested_non_float_deltas_raise():
    beta0 = 0.0
    naics_levels = np.zeros((2, 2), int)
    zip_levels = np.zeros((2, 2), int)
    naics_base = np.array([0.1], float)
    zip_base = np.array([0.2], float)
    naics_deltas = [np.array([0, 1], int)]  # not float
    zip_deltas = [np.array([0.0, 1.0], float)]
    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )


def test_eta_nested_out_of_bounds_indices_raise():
    beta0 = 0.0
    naics_levels = np.array([[1, 0]], int)  # L0 index 1 but base length 1 -> OOB
    zip_levels = np.array([[0, 0]], int)
    naics_base = np.array([0.1], float)
    zip_base = np.array([0.2], float)
    naics_deltas = [np.array([0.0], float)]
    zip_deltas = [np.array([0.0], float)]

    with pytest.raises(ValueError):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )


def test_eta_nested_handles_zero_deltas_when_JM1_zero():
    # J=1 (no NAICS deltas), M=1 (no ZIP deltas)
    beta0 = 0.5
    naics_levels = np.array([[0], [0], [0]], int)
    zip_levels = np.array([[0], [0], [0]], int)
    naics_base = np.array([0.1], float)
    zip_base = np.array([0.2], float)

    out = eta_nested(beta0, naics_base, [], zip_base, [], naics_levels, zip_levels)
    np.testing.assert_allclose(out, np.array([0.5 + 0.1 + 0.2] * 3, dtype=float))
