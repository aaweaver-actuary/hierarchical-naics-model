from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)


def test_empty_codes_raises():
    with pytest.raises(ValueError):
        build_hierarchical_indices([], cut_points=[2, 3])


def test_shapes_and_group_counts(naics_pool, naics_cut_points):
    # Duplicate and shuffle to create many rows
    codes = (naics_pool * 10)[:47]
    out = build_hierarchical_indices(codes, cut_points=naics_cut_points)

    assert out["levels"] == [f"L{c}" for c in naics_cut_points]
    code_levels = out["code_levels"]
    assert isinstance(code_levels, np.ndarray)
    assert code_levels.shape == (len(codes), len(naics_cut_points))

    # Each level's group count should match the number of unique prefixes
    codes_s = pd.Series(codes, dtype="string")
    expected_counts = [codes_s.str.slice(0, c).nunique() for c in naics_cut_points]
    assert out["group_counts"] == expected_counts

    # Maps and unique_per_level lengths align
    for uniq, mp in zip(out["unique_per_level"], out["maps"]):
        assert len(uniq) == len(mp)
        # every unique label maps to an integer in-range
        for i, u in enumerate(uniq):
            assert mp[u] == i


def test_prefix_fill_right_padding_behavior():
    codes = ["52", "51213", "511110"]  # varying lengths 2,5,6
    cut_points = [2, 3, 6]
    out_no_pad = build_hierarchical_indices(codes, cut_points=cut_points)
    out_pad = build_hierarchical_indices(codes, cut_points=cut_points, prefix_fill="0")

    # Without padding, 6-digit slice of "52" is just "52" (shorter), so uniqueness differs
    uniq_no_pad_L6 = set(out_no_pad["unique_per_level"][2])
    uniq_pad_L6 = set(out_pad["unique_per_level"][2])

    assert "52" in uniq_no_pad_L6
    assert "520000" in uniq_pad_L6  # right-padded to width 6 using '0'
