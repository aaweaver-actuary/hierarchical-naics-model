import pytest
from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)


def test_empty_codes_raises():
    with pytest.raises(ValueError):
        build_hierarchical_indices([], cut_points=[2, 3, 6], prefix_fill="0")


def test_null_codes_raises():
    with pytest.raises(ValueError):
        build_hierarchical_indices(
            [None, "511110"], cut_points=[2, 3, 6], prefix_fill="0"
        )


def test_invalid_cut_points_raises():
    codes = ["511110", "511120"]
    # Not strictly increasing
    with pytest.raises(ValueError):
        build_hierarchical_indices(codes, cut_points=[2, 2, 6], prefix_fill="0")
    # Negative cut
    with pytest.raises(ValueError):
        build_hierarchical_indices(codes, cut_points=[-1, 3, 6], prefix_fill="0")
