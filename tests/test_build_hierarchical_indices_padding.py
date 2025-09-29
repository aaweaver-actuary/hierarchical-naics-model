from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
import numpy as np


def test_prefix_padding_enforced_by_default():
    # Short codes should be right-padded with '0' by default
    codes = ["52", "51"]
    idx = build_hierarchical_indices(codes, cut_points=[2, 3, 6])
    # The padded codes should be '520000', '510000'
    expected = np.array([["52", "520", "520000"], ["51", "510", "510000"]])
    # Check that unique_per_level matches expected padded codes
    for j, arr in enumerate(idx["unique_per_level"]):
        assert set(arr) == set(expected[:, j])


def test_no_padding_if_disabled():
    codes = ["52", "51"]
    idx = build_hierarchical_indices(codes, cut_points=[2, 3, 6], prefix_fill="")
    # No padding: codes remain as-is, so last level is just '52', '51'
    expected = np.array([["52", "52", "52"], ["51", "51", "51"]])
    for j, arr in enumerate(idx["unique_per_level"]):
        assert set(arr) == set(expected[:, j])
