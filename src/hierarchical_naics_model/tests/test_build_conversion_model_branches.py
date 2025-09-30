import numpy as np
import pytest
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model


@log_test_performance
def test_invalid_y_shape(test_run_id):
    y = np.zeros((10, 1), dtype="int8")
    naics_levels = np.zeros((10, 1), dtype=int)
    zip_levels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


@log_test_performance
def test_invalid_y_nonbinary(test_run_id):
    y = np.arange(10)
    naics_levels = np.zeros((10, 1), dtype=int)
    zip_levels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


@log_test_performance
def test_invalid_naics_levels_shape(test_run_id):
    y = np.zeros(10, dtype="int8")
    naics_levels = np.zeros(10, dtype=int)
    zip_levels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


@log_test_performance
def test_invalid_zip_levels_shape(test_run_id):
    y = np.zeros(10, dtype="int8")
    naics_levels = np.zeros((10, 1), dtype=int)
    zip_levels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


def test_invalid_group_counts_length():
    y = np.zeros(10, dtype="int8")
    naics_levels = np.zeros((10, 2), dtype=int)
    zip_levels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


def test_invalid_index_bounds():
    y = np.zeros(10, dtype="int8")
    naics_levels = np.full((10, 1), -1, dtype=int)
    zip_levels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError):
        build_conversion_model(
            y=y,
            naics_levels=naics_levels,
            zip_levels=zip_levels,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )
