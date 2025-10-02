import numpy as np
import pytest
from hierarchical_naics_model.utils import _validation


def test_validate_target_shape_is_1d_numpy_array():
    arr = np.array([0, 1, 1])
    _validation._validate_target_shape_is_1d_numpy_array(arr)
    arr2 = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        _validation._validate_target_shape_is_1d_numpy_array(arr2)


def test_validate_target_is_integer_subtype():
    arr = np.array([0, 1, 1], dtype=np.int32)
    _validation._validate_target_is_integer_subtype(arr)
    arr2 = np.array([0.1, 1.2, 1.0])
    with pytest.raises(ValueError):
        _validation._validate_target_is_integer_subtype(arr2)


def test_validate_binary_response_with_only_0s_and_1s():
    arr = np.array([0, 1, 1])
    _validation._validate_binary_response_with_only_0s_and_1s(arr)
    arr2 = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        _validation._validate_binary_response_with_only_0s_and_1s(arr2)
