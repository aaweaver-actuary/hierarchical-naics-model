import numpy as np
import pytest
from hierarchical_naics_model.core.effects import eta_additive, eta_nested


def test_eta_additive_basic():
    beta0 = 1.0
    naics_effects = [np.array([0.1, -0.2]), np.array([0.05, 0.0, -0.05])]
    zip_effects = [np.array([0.2]), np.array([-0.1, 0.0])]
    naics_levels = np.array([[0, 2], [1, 1]], dtype=int)
    zip_levels = np.array([[0, 1], [0, 0]], dtype=int)
    out = eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "bad_shape",
    [
        ((2, 2), (2, 1)),
        ((2, 1), (2, 2)),
    ],
)
def test_eta_additive_bad_shapes(bad_shape):
    naics_levels = np.zeros(bad_shape[0], dtype=int)
    zip_levels = np.zeros(bad_shape[1], dtype=int)
    naics_effects = [np.array([0.1, -0.2]), np.array([0.05, 0.0, -0.05])]
    zip_effects = [np.array([0.2]), np.array([-0.1, 0.0])]
    beta0 = 1.0
    with pytest.raises(Exception):
        eta_additive(beta0, naics_effects, zip_effects, naics_levels, zip_levels)


def test_eta_nested_basic():
    beta0 = 1.0
    naics_base = np.array([0.1, -0.1], dtype=float)
    naics_deltas = [np.array([0.0, 0.05, -0.05], dtype=float)]
    zip_base = np.array([0.2], dtype=float)
    zip_deltas = [np.array([-0.1, 0.0], dtype=float)]
    naics_levels = np.array([[0, 2], [1, 1]], dtype=int)
    zip_levels = np.array([[0, 1], [0, 0]], dtype=int)
    out = eta_nested(
        beta0, naics_base, naics_deltas, zip_base, zip_deltas, naics_levels, zip_levels
    )
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "bad_shape",
    [
        ((2, 2), (2, 1)),
        ((2, 1), (2, 2)),
    ],
)
def test_eta_nested_bad_shapes(bad_shape):
    naics_levels = np.zeros(bad_shape[0], dtype=int)
    zip_levels = np.zeros(bad_shape[1], dtype=int)
    naics_base = np.array([0.1, -0.1], dtype=float)
    naics_deltas = [np.array([0.0, 0.05, -0.05], dtype=float)]
    zip_base = np.array([0.2], dtype=float)
    zip_deltas = [np.array([-0.1, 0.0], dtype=float)]
    beta0 = 1.0
    with pytest.raises(Exception):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )
