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


@pytest.mark.parametrize(
    "validator",
    (
        _validation._validate_naics_levels_is_2d_array,
        _validation._validate_zip_levels_is_2d_array,
    ),
)
def test_validate_levels_requires_2d_arrays(validator):
    good = np.zeros((3, 2), dtype=int)
    validator(good)  # should not raise

    with pytest.raises(ValueError):
        validator(np.zeros(3, dtype=int))


def test_validate_input_shape_mismatched_rows():
    y = np.zeros(4, dtype=int)
    naics = np.zeros((3, 2), dtype=int)
    zips = np.zeros((4, 2), dtype=int)

    with pytest.raises(ValueError):
        _validation._validate_input_shape(y, naics, zips)
