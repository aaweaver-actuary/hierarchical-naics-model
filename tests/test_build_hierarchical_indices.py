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


@pytest.fixture
def codes_and_cut_points(naics_pool, naics_cut_points):
    codes = (naics_pool * 10)[:47]
    out = build_hierarchical_indices(codes, cut_points=naics_cut_points)
    codes_s = pd.Series(codes, dtype="string")
    expected_counts = [codes_s.str.slice(0, c).nunique() for c in naics_cut_points]
    return {
        "codes": codes,
        "cut_points": naics_cut_points,
        "output": out,
        "expected_counts": expected_counts,
    }


def test_hierarchical_indices_levels_are_named_as_expected(codes_and_cut_points):
    out = codes_and_cut_points["output"]
    cut_points = codes_and_cut_points["cut_points"]
    assert out["levels"] == [f"L{c}" for c in cut_points]


def test_hierarchical_indices_code_levels_is_numpy_array(codes_and_cut_points):
    code_levels = codes_and_cut_points["output"]["code_levels"]
    assert isinstance(code_levels, np.ndarray)


def test_hierarchical_indices_code_levels_shape_is_correct(codes_and_cut_points):
    code_levels = codes_and_cut_points["output"]["code_levels"]
    codes = codes_and_cut_points["codes"]
    cut_points = codes_and_cut_points["cut_points"]
    assert code_levels.shape == (len(codes), len(cut_points))


def test_hierarchical_indices_group_counts_match_unique_prefixes(codes_and_cut_points):
    out = codes_and_cut_points["output"]
    expected_counts = codes_and_cut_points["expected_counts"]
    assert out["group_counts"] == expected_counts


@pytest.mark.parametrize("level_idx", [0, 1, 2])
def test_hierarchical_indices_unique_per_level_and_maps_length_align(
    codes_and_cut_points, level_idx
):
    out = codes_and_cut_points["output"]
    uniq = out["unique_per_level"][level_idx]
    mp = out["maps"][level_idx]
    assert len(uniq) == len(mp)


@pytest.mark.parametrize("level_idx", [0, 1, 2])
def test_hierarchical_indices_unique_label_maps_to_correct_integer(
    codes_and_cut_points, level_idx
):
    out = codes_and_cut_points["output"]
    uniq = out["unique_per_level"][level_idx]
    mp = out["maps"][level_idx]
    for i, u in enumerate(uniq):
        assert mp[u] == i


@pytest.mark.parametrize(
    "codes,cut_points,expected_shape",
    [
        (["12345"], [2, 5], (1, 2)),
        (["1", "12", "123"], [1, 2, 3], (3, 3)),
    ],
)
def test_hierarchical_indices_code_levels_shape_various_inputs(
    codes, cut_points, expected_shape
):
    out = build_hierarchical_indices(codes, cut_points=cut_points)
    assert out["code_levels"].shape == expected_shape


@pytest.fixture
def codes_and_cut_points_for_padding():
    codes = ["52", "51213", "511110"]  # varying lengths 2,5,6
    cut_points = [2, 3, 6]
    out_no_pad = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=None
    )
    out_pad = build_hierarchical_indices(codes, cut_points=cut_points, prefix_fill="0")
    return {
        "codes": codes,
        "cut_points": cut_points,
        "out_no_pad": out_no_pad,
        "out_pad": out_pad,
    }


@pytest.mark.parametrize(
    "level_idx,expected_no_pad,expected_pad",
    [
        (2, "52", "520000"),  # L6: "52" (no pad), "520000" (pad)
        (1, "52", "520"),  # L3: "52" (no pad), right-pad → "520"
        (0, "52", "52"),  # L2: "52" (no pad), "52" (pad, no effect)
    ],
)
def test_unique_per_level_contains_expected_label_without_padding(
    codes_and_cut_points_for_padding, level_idx, expected_no_pad, expected_pad
):
    uniq_no_pad = set(
        codes_and_cut_points_for_padding["out_no_pad"]["unique_per_level"][level_idx]
    )
    assert expected_no_pad in uniq_no_pad


@pytest.mark.parametrize(
    "level_idx,expected_no_pad,expected_pad",
    [
        (2, "52", "520000"),  # L6: "52" (no pad), "520000" (pad)
        (1, "52", "520"),  # L3: "52" (no pad), right-pad → "520"
        (0, "52", "52"),  # L2: "52" (no pad), "52" (pad, no effect)
    ],
)
def test_unique_per_level_contains_expected_label_with_right_padding(
    codes_and_cut_points_for_padding, level_idx, expected_no_pad, expected_pad
):
    uniq_pad = set(
        codes_and_cut_points_for_padding["out_pad"]["unique_per_level"][level_idx]
    )
    assert expected_pad in uniq_pad


@pytest.mark.parametrize(
    "codes,cut_points,prefix_fill,expected",
    [
        ([""], [2], None, ""),  # empty code, no pad
        ([""], [2], "0", "00"),  # empty code, pad to length 2
        (["1"], [2], None, "1"),  # short code, no pad
        (["1"], [2], "0", "10"),  # short code, pad to length 2
        (["123"], [2], None, "12"),  # longer code, no pad
        (["123"], [2], "0", "12"),  # longer code, pad, but no effect
    ],
)
def test_prefix_fill_edge_cases_for_single_code(
    codes, cut_points, prefix_fill, expected
):
    out = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=prefix_fill
    )
    assert out["unique_per_level"][0][0] == expected
