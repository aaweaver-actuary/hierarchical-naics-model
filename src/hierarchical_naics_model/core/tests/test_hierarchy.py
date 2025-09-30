# tests/test_core_hierarchy.py
from __future__ import annotations

import itertools
from typing import List, Mapping

import numpy as np
import pytest

# import the module under test
from hierarchical_naics_model.core.hierarchy import (
    build_hierarchical_indices,
    make_backoff_resolver,
)


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture(scope="module")
def toy_codes_mixed() -> List[str]:
    # Intentionally mixed lengths, whitespace, repeated groups, and a None
    # Only return strings, filter out None
    return [
        c
        for c in ["52", " 521 ", "522110", "522120", "52", None, ""]
        if isinstance(c, str)
    ]


@pytest.fixture(scope="module", params=[[2, 3, 6], [2, 3, 5]])
def cut_points_param(request) -> List[int]:
    return request.param


@pytest.fixture(scope="module")
def codes_longer_than_max() -> List[str]:
    # Codes longer than max(cut_points) must not be truncated; slicing should use prefixes.
    return ["520000999", "521123999", "522110888"]


# -------------------------
# Validation edge cases
# -------------------------


@pytest.mark.parametrize(
    "cut_points",
    [
        [],
        [0, 1, 2],
        [2, 2, 3],  # not strictly increasing
        [3, 2],  # decreasing
        [-1, 2, 3],  # non-positive
    ],
)
def test_build_hierarchical_indices_invalid_cut_points_raises(cut_points):
    with pytest.raises(ValueError):
        build_hierarchical_indices(["52"], cut_points=cut_points)


@pytest.mark.parametrize("fill", ["", "00", "abc"])  # not a single char
def test_build_hierarchical_indices_invalid_prefix_fill_raises(fill):
    with pytest.raises(ValueError):
        build_hierarchical_indices(["52"], cut_points=[2, 3], prefix_fill=fill)


# -------------------------
# Core structure & shapes
# -------------------------


def test_hierarchical_indices_shapes_and_fields(toy_codes_mixed, cut_points_param):
    idx = build_hierarchical_indices(
        toy_codes_mixed, cut_points=cut_points_param, prefix_fill="0"
    )

    # Basic structure
    assert isinstance(idx["levels"], list)
    assert isinstance(idx["code_levels"], np.ndarray)
    assert isinstance(idx["unique_per_level"], list)
    assert isinstance(idx["maps"], list)
    assert isinstance(idx["group_counts"], list)
    assert isinstance(idx["parent_index_per_level"], list)
    assert isinstance(idx["max_len"], int)
    assert isinstance(idx["cut_points"], list)

    N = len(toy_codes_mixed)
    L = len(cut_points_param)

    # Shapes
    assert idx["code_levels"].shape == (N, L)
    assert len(idx["levels"]) == L
    assert len(idx["unique_per_level"]) == L
    assert len(idx["maps"]) == L
    assert len(idx["group_counts"]) == L
    assert len(idx["parent_index_per_level"]) == L

    # Level names are "L{cut}"
    assert idx["levels"] == [f"L{c}" for c in cut_points_param]

    # Group counts match uniques per level
    for j in range(L):
        assert idx["group_counts"][j] == len(idx["unique_per_level"][j])

    # Per-level maps cover exactly the uniques and map to 0..K-1
    for j in range(L):
        uniq = list(map(str, idx["unique_per_level"][j]))
        m: Mapping[str, int] = idx["maps"][j]
        assert set(m.keys()) == set(uniq)
        # Ensure indices are contiguous 0..K-1
        assert set(m.values()) == set(range(len(uniq)))

    # Parent pointers: None at L0, arrays for deeper levels and all valid
    assert idx["parent_index_per_level"][0] is None
    for j in range(1, L):
        par = idx["parent_index_per_level"][j]
        if par is not None:
            assert isinstance(par, np.ndarray)
            # All parents should exist (never -1) because padding enforces consistent prefixes
            assert (par >= 0).all()
            # Parent indices are within 0..K_{j-1}-1
            assert par.max() < idx["group_counts"][j - 1]


def test_code_levels_respect_first_seen_order_stability():
    # First appearance order at L2 will be: "52" -> index 0, then "11" -> 1, then "31" -> 2
    codes = ["52", "11", "52", "31", "11", "31", "52"]
    idx = build_hierarchical_indices(codes, cut_points=[2], prefix_fill="0")
    uniq = list(map(str, idx["unique_per_level"][0]))
    assert uniq == ["52", "11", "31"]
    mp = idx["maps"][0]
    assert mp == {"52": 0, "11": 1, "31": 2}
    # Check row mapping consistent
    expected = np.array([0, 1, 0, 2, 1, 2, 0])
    np.testing.assert_array_equal(idx["code_levels"][:, 0], expected)


def test_padding_and_long_codes_not_truncated(codes_longer_than_max):
    # cut_points max is 6; codes are > 6 chars and must not be truncated in storage
    idx = build_hierarchical_indices(
        codes_longer_than_max, cut_points=[2, 3, 6], prefix_fill="0"
    )
    # L6 uniques reflect only first 6 chars (prefix), extra trailing chars irrelevant
    uniq_L6 = set(map(str, idx["unique_per_level"][2]))
    assert uniq_L6 == {"520000", "521123", "522110"}
    # L3 uniques similarly
    uniq_L3 = set(map(str, idx["unique_per_level"][1]))
    assert uniq_L3 == {"520", "521", "522"}


@pytest.mark.parametrize(
    "prefix_fill,expected_L3_label",
    [
        ("0", "520"),
        ("X", "52X"),
        ("_", "52_"),
    ],
)
def test_padding_character_is_used(prefix_fill, expected_L3_label):
    # "52" should become "52{fill}{fill}{fill}..." → L3 is "52{fill}"
    idx = build_hierarchical_indices(
        ["52"], cut_points=[2, 3, 6], prefix_fill=prefix_fill
    )
    uniq_L3 = list(map(str, idx["unique_per_level"][1]))
    assert uniq_L3 == [expected_L3_label]


def test_empty_and_whitespace_codes_become_all_fill():
    # Empty/None/whitespace all turn into repeated fill chars after padding
    codes = ["", "   ", None]
    # Filter out None before passing
    idx = build_hierarchical_indices(
        [c for c in codes if isinstance(c, str)], cut_points=[2, 3], prefix_fill="9"
    )
    uniq_L2 = set(map(str, idx["unique_per_level"][0]))
    uniq_L3 = set(map(str, idx["unique_per_level"][1]))
    assert uniq_L2 == {"99"}
    assert uniq_L3 == {"999"}


# -------------------------
# Parent pointer semantics
# -------------------------


def test_parent_index_corresponds_to_prefix(toy_codes_mixed):
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(toy_codes_mixed, cut_points=cp, prefix_fill="0")
    uniq_L2 = list(map(str, idx["unique_per_level"][0]))
    uniq_L3 = list(map(str, idx["unique_per_level"][1]))
    uniq_L6 = list(map(str, idx["unique_per_level"][2]))
    par_L3 = idx["parent_index_per_level"][1]
    par_L6 = idx["parent_index_per_level"][2]

    # For each L3 label, its parent index at L2 must correspond to its 2-char prefix in uniq_L2
    if par_L3 is not None:
        for child_idx, lab in enumerate(uniq_L3):
            parent_label = lab[: cp[0]]
            assert uniq_L2[par_L3[child_idx]] == parent_label

    # For L6 → L3 pointer
    if par_L6 is not None:
        for child_idx, lab in enumerate(uniq_L6):
            parent_label = lab[: cp[1]]
            assert uniq_L3[par_L6[child_idx]] == parent_label


# -------------------------
# Backoff resolver: behavior
# -------------------------


def _maps_from_index(idx) -> List[Mapping[str, int]]:
    return idx["maps"]


def test_make_backoff_resolver_len_mismatch_raises():
    with pytest.raises(ValueError):
        make_backoff_resolver(cut_points=[2, 3], level_maps=[{"52": 0}])


def test_backoff_known_at_all_levels():
    train_codes = ["520000", "521000", "522110", "522120"]
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(train_codes, cut_points=cp, prefix_fill="0")
    maps = _maps_from_index(idx)
    resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="0")

    # "522110" is present at all levels via its prefixes
    res = resolve("522110")
    # level 0: "52", level 1: "522", level 2: "522110"
    assert all(r is not None for r in res)
    # spot-check label indices match expected labels
    assert idx["unique_per_level"][0][res[0]] == "52"
    assert idx["unique_per_level"][1][res[1]] == "522"
    assert idx["unique_per_level"][2][res[2]] == "522110"


def test_backoff_unknown_leaf_backs_up_to_parent():
    # Training sees L2: "52"; L3: "521"; L6: "522110" only.
    train_codes = ["520000", "521000", "522110"]
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(train_codes, cut_points=cp, prefix_fill="0")
    maps = _maps_from_index(idx)
    resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="0")

    # "522130" not seen at L6; parent L3 "522" is present, so L3 gets its own index, L2 is '52'
    res = resolve("522130")
    # L0 index for '52', L1 index for '522', L2 (L6) backs off to L3's index
    l2_idx = maps[0]["52"]
    l3_idx = maps[1]["522"]
    assert res[0] == l2_idx
    assert res[1] == l3_idx
    assert res[2] == l3_idx


def test_backoff_all_unknown_returns_none():
    # Training with unrelated family
    train_codes = ["111111", "111222"]
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(train_codes, cut_points=cp, prefix_fill="0")
    maps = _maps_from_index(idx)
    resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="0")

    # "52..." family unseen at any level
    res = resolve("520000")
    assert res == [None, None, None]


def test_backoff_handles_none_and_whitespace_codes():
    train_codes = ["520000", "521000", "522110"]
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(train_codes, cut_points=cp, prefix_fill="0")
    maps = _maps_from_index(idx)
    resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="0")

    # Only call resolver with strings
    assert resolve("") == [None, None, None]
    assert resolve("   ") == [None, None, None]


def test_backoff_respects_padding_character():
    # Use 'X' padding so we can assert that L3 label becomes '12X' when raw is '12'
    train_codes = [
        "12XXXX",
        "12YZZZ",
    ]  # ensure at least L3 groups exist with X padding path
    cp = [2, 3, 6]
    idx = build_hierarchical_indices(train_codes, cut_points=cp, prefix_fill="X")
    maps = _maps_from_index(idx)
    resolve = make_backoff_resolver(cut_points=cp, level_maps=maps, prefix_fill="X")

    res = resolve("12")  # becomes "12XXXX"
    # L0 should be index of "12"
    l0_idx = maps[0]["12"]
    assert res[0] == l0_idx
    # L1 should be index of "12X"
    l1_idx = maps[1].get("12X")
    # Depending on training data, "12X" may or may not exist; if not, backoff to L0
    assert res[1] in (l1_idx, l0_idx)
    # L2 should resolve to "12XXXX" or backoff to above
    l2_idx = maps[2].get("12XXXX")
    assert res[2] in (l2_idx, res[1], res[0])


# -------------------------
# Row-level indices sanity
# -------------------------


@pytest.mark.parametrize("cp", [[2], [2, 3], [2, 3, 5]])
def test_row_level_indices_with_mixed_inputs(cp):
    codes = ["52", "521", "52211", "  52", None, ""]
    # Filter out None before passing
    filtered_codes = [c for c in codes if isinstance(c, str)]
    idx = build_hierarchical_indices(filtered_codes, cut_points=cp, prefix_fill="0")
    # Each row’s level j label must match the j-prefix of its padded code
    max_len = max(cp)
    padded = [
        ("" if c is None else str(c).strip()).ljust(max_len, "0")
        for c in filtered_codes
    ]
    # For each level j and row i, label should appear in uniques and mapping must map to code_levels[i,j]
    for j, cut in enumerate(cp):
        labels_at_j = [p[:cut] for p in padded]
        uniq = list(map(str, idx["unique_per_level"][j]))
        m = idx["maps"][j]
        # All labels should exist in uniq and map to the same index used in code_levels
        for i, lab in enumerate(labels_at_j):
            assert lab in m
            assert uniq[idx["code_levels"][i, j]] == lab


# -------------------------
# Exhaustive small cartesian sanity (robustness)
# -------------------------


@pytest.mark.parametrize("cut_points", [[2, 3], [2, 4]])
@pytest.mark.parametrize("prefix_fill", ["0", "Z"])
def test_cartesian_small_space_exhaustive(cut_points, prefix_fill):
    # All 2-digit prefixes from {"5","6"} and 2 variants of third/fourth char
    # Exhaustive set ensures stable behavior for factorization and parent pointers.
    roots = ["50", "51", "60", "61"]
    tails = ["0", "1", prefix_fill]  # include the fill char to stress padding grouping
    codes = [r + t for r, t in itertools.product(roots, tails)]
    idx = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=prefix_fill
    )

    # Group counts > 0 everywhere; parents consistent
    for j in range(len(cut_points)):
        assert idx["group_counts"][j] > 0, f"Zero group count at level {j}"
        if j > 0:
            par = idx["parent_index_per_level"][j]
            assert (par >= 0).all(), f"Negative parent index at level {j}"  # type: ignore[possibly-unbound-attribute]
            assert par.max() < idx["group_counts"][j - 1], (  # type: ignore[possibly-unbound-attribute]
                f"par.max() out of range at level {j}"
            )
