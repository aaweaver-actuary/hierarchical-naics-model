import pytest
import numpy as np
from hierarchical_naics_model.core.effects import _ensure_vector, _check_effects_list


def test_ensure_vector_type():
    arr = np.array([1.0, 2.0], dtype=float)
    _ensure_vector("arr", arr, 2)
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        _ensure_vector("arr", [1.0, 2.0], 2)


def test_ensure_vector_ndim():
    arr = np.array([[1.0, 2.0]], dtype=float)
    with pytest.raises(ValueError, match="must be 1-D"):
        _ensure_vector("arr", arr, 2)


def test_ensure_vector_length():
    arr = np.array([1.0, 2.0], dtype=float)
    with pytest.raises(ValueError, match="length mismatch"):
        _ensure_vector("arr", arr, 3)


def test_ensure_vector_dtype():
    arr = np.array([1, 2], dtype=int)
    with pytest.raises(ValueError, match="dtype must be float"):
        _ensure_vector("arr", arr, 2)


def test_check_effects_list_type():
    effects = [np.array([1.0, 2.0], dtype=float), [1.0, 2.0]]
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        _check_effects_list("effects", effects, 2)


def test_check_effects_list_ndim():
    effects = [np.array([[1.0, 2.0]], dtype=float), np.array([1.0, 2.0], dtype=float)]
    with pytest.raises(ValueError, match="must be 1-D"):
        _check_effects_list("effects", effects, 2)


def test_check_effects_list_dtype():
    effects = [np.array([1, 2], dtype=int), np.array([1.0, 2.0], dtype=float)]
    with pytest.raises(ValueError, match="dtype must be float"):
        _check_effects_list("effects", effects, 2)


def test_check_effects_list_length():
    effects = [np.array([1.0, 2.0], dtype=float)]
    with pytest.raises(ValueError, match="length.*must match number of levels"):
        _check_effects_list("effects", effects, 2)
