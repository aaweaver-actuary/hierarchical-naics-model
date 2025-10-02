from ..core.validation import validate_level_indices
from ..types import Integers
import numpy as np


def check_inputs(
    y: np.ndarray,
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
    naics_group_counts: Integers,
    zip_group_counts: Integers,
) -> None:
    _validate_target_shape_is_1d_numpy_array(y)
    _validate_target_is_integer_subtype(y)
    _validate_binary_response_with_only_0s_and_1s(y)

    _validate_naics_levels_is_2d_array(naics_levels)
    _validate_zip_levels_is_2d_array(zip_levels)

    _validate_input_shape(y, naics_levels, zip_levels)

    validate_level_indices(naics_levels, naics_group_counts)
    validate_level_indices(zip_levels, zip_group_counts)


def _validate_input_shape(y, naics_levels, zip_levels):
    """Input shapes should be:
    y:               (N,) where N is number of observations
    naics_levels:   (N, J) where J is number of NAICS levels
    zip_levels:     (N, M) where M is number of ZIP levels"""
    N = y.shape[0]
    if naics_levels.shape[0] != N or zip_levels.shape[0] != N:
        raise ValueError(
            "`y`, `naics_levels`, and `zip_levels` must have the same number of rows."
        )


def _validate_zip_levels_is_2d_array(zip_levels):
    if not isinstance(zip_levels, np.ndarray) or zip_levels.ndim != 2:
        raise ValueError("`zip_levels` must be a 2-D numpy array (N, M).")


def _validate_naics_levels_is_2d_array(naics_levels):
    if not isinstance(naics_levels, np.ndarray) or naics_levels.ndim != 2:
        raise ValueError("`naics_levels` must be a 2-D numpy array (N, J).")


def _validate_binary_response_with_only_0s_and_1s(y):
    if np.any((y != 0) & (y != 1)):
        raise ValueError("`y` must contain only 0 or 1.")


def _validate_target_is_integer_subtype(y):
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("`y` must be integer dtype (0/1).")


def _validate_target_shape_is_1d_numpy_array(y):
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("`y` must be a 1-D numpy array.")
