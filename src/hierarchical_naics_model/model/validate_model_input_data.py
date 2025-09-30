import pymc as pm
import numpy as np


def _validate_model_input_data(
    y, naics_levels, zip_levels, naics_group_counts, zip_group_counts
):
    _validate_pymc_available(pm)

    y = np.asarray(y, dtype="int8")
    _validate_y_has_the_correct_shape(y)
    _validate_y_is_binary_response(y)
    _validate_naics_levels_are_2d_array(naics_levels)
    _validate_zip_levels_are_2d_array(zip_levels)
    _validate_everything_has_same_number_rows(y, naics_levels, zip_levels)
    L_naics = _validate_naics_levels_have_correct_length(
        naics_levels, naics_group_counts
    )
    L_zip = _validate_zip_group_counts(zip_levels, zip_group_counts)
    _validate_naics_levels_are_never_negative(naics_levels)
    _validate_zip_levels_are_never_negative(zip_levels)
    # Per-level upper bound checks
    _validate_naics_levels_indices(naics_levels, naics_group_counts, L_naics)
    _validate_zip_levels_indices(zip_levels, zip_group_counts, L_zip)
    return y, L_naics, L_zip


def _validate_zip_levels_indices(zip_levels, zip_group_counts, L_zip):
    for m in range(L_zip):
        if int(zip_levels[:, m].max(initial=-1)) >= int(zip_group_counts[m]):
            raise ValueError("`zip_levels` index out of range for level {m}.")


def _validate_naics_levels_indices(naics_levels, naics_group_counts, L_naics):
    for j in range(L_naics):
        if int(naics_levels[:, j].max(initial=-1)) >= int(naics_group_counts[j]):
            raise ValueError("`naics_levels` index out of range for level {j}.")


def _validate_zip_levels_are_never_negative(zip_levels):
    if (zip_levels < 0).any():
        raise ValueError("`zip_levels` contains negative indices.")


def _validate_naics_levels_are_never_negative(naics_levels):
    if (naics_levels < 0).any():
        raise ValueError("`naics_levels` contains negative indices.")


def _validate_zip_group_counts(zip_levels, zip_group_counts):
    L_zip = int(zip_levels.shape[1])
    if len(zip_group_counts) != L_zip:
        raise ValueError("`zip_group_counts` length must equal number of ZIP levels.")
    return L_zip


def _validate_naics_levels_have_correct_length(naics_levels, naics_group_counts):
    L_naics = int(naics_levels.shape[1])
    if len(naics_group_counts) != L_naics:
        raise ValueError(
            "`naics_group_counts` length must equal number of NAICS levels."
        )
    return L_naics


def _validate_everything_has_same_number_rows(y, naics_levels, zip_levels):
    N = y.shape[0]
    if naics_levels.shape[0] != N or zip_levels.shape[0] != N:
        raise ValueError(
            "`naics_levels` and `zip_levels` must have the same number of rows as `y`."
        )


def _validate_zip_levels_are_2d_array(zip_levels):
    if zip_levels.ndim != 2:
        raise ValueError("`zip_levels` must be a 2D integer array.")


def _validate_naics_levels_are_2d_array(naics_levels):
    if naics_levels.ndim != 2:
        raise ValueError("`naics_levels` must be a 2D integer array.")


def _validate_pymc_available(pm=pm):
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")


def _validate_y_has_the_correct_shape(y):
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array of 0/1.")


def _validate_y_is_binary_response(y):
    if not np.isin(y, [0, 1]).all():
        raise ValueError("`y` must be binary in {0,1}.")
