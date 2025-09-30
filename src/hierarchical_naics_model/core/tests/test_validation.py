# tests/test_core_validation.py
from __future__ import annotations

import numpy as np
import pytest

from hierarchical_naics_model.core.validation import (
    validate_cut_points,
    validate_level_indices,
)


# -----------------------
# validate_cut_points
# -----------------------


@pytest.mark.parametrize(
    "cut_points,expected",
    [
        ([2, 3, 5], (2, 3, 5)),
        ((1, 4, 6, 7), (1, 4, 6, 7)),
        (np.array([2, 3, 6]), (2, 3, 6)),
    ],
)
def test_validate_cut_points_happy(cut_points, expected):
    out = validate_cut_points(cut_points)
    assert out == expected
    assert isinstance(out, tuple)


@pytest.mark.parametrize(
    "bad",
    [
        [],  # empty
        [0, 1, 2],  # non-positive
        [-1, 2, 3],  # negative
        [2, 2, 3],  # not strictly increasing
        [3, 2],  # decreasing
        ["2", "x", 3],  # non-int convertible
    ],
)
def test_validate_cut_points_raises(bad):
    with pytest.raises(ValueError):
        validate_cut_points(bad)


# -----------------------
# validate_level_indices
# -----------------------


def test_validate_level_indices_happy_small():
    # N=4, L=2, group_counts=[2,3]
    levels = np.array([[0, 0], [1, 1], [0, 2], [1, 0]], dtype=int)
    validate_level_indices(levels, [2, 3])  # no raise


def test_validate_level_indices_empty_N_is_okay():
    levels = np.empty((0, 3), dtype=int)
    validate_level_indices(levels, [1, 1, 1])  # accepted (returns None)


def test_validate_level_indices_zero_levels_is_okay():
    levels = np.empty((5, 0), dtype=int)
    validate_level_indices(levels, [])  # accepted (returns None)


@pytest.mark.parametrize(
    "levels,group_counts,err_msg",
    [
        (None, [1], "`levels` must be a numpy.ndarray"),
        (np.array([0, 1, 2], dtype=int), [3], "`levels` must be 2-D"),
        (np.array([[0, 1]], dtype=float), [1, 2], "`levels` dtype must be an integer"),
        (np.array([[0, 1]], dtype=int), [2], "`group_counts` length"),
        (np.array([[0, 1]], dtype=int), [0, 2], "group_counts must be positive"),
    ],
)
def test_validate_level_indices_structure_raises(levels, group_counts, err_msg):
    with pytest.raises(ValueError) as e:
        validate_level_indices(levels, group_counts)
    assert err_msg in str(e.value)


def test_validate_level_indices_bounds_low_high_raises():
    levels = np.array([[0, 1, 2], [0, -1, 1]], dtype=int)  # -1 invalid at level 1
    with pytest.raises(ValueError) as e:
        validate_level_indices(levels, [1, 2, 3])
    assert "out of bounds at level 1" in str(e.value)

    levels2 = np.array([[0, 1], [0, 2]], dtype=int)  # 2 invalid when K2=2 (max index=1)
    with pytest.raises(ValueError) as e2:
        validate_level_indices(levels2, [1, 2])
    assert "allowed range=[0, 1]" in str(e2.value)


def test_validate_level_indices_group_counts_type_error():
    levels = np.array([[0, 0]], dtype=int)
    with pytest.raises(TypeError) as e:
        validate_level_indices(
            levels,
            ("a", 1),  # type: ignore
        )  # not ints, tuple to match Sequence[int]
    assert "must be a sequence of ints" in str(e.value)
