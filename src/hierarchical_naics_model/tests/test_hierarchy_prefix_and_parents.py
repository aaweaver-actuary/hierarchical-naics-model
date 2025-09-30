# tests/test_hierarchy_prefix_and_parents.py
import numpy as np
import pytest
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)

from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)


@pytest.fixture
def naics_codes():
    return ["52", "521", "522110", "522120"]


@pytest.fixture
def naics_indices(naics_codes):
    return build_hierarchical_indices(
        naics_codes, cut_points=[2, 3, 6], prefix_fill="0"
    )


@pytest.mark.parametrize("expected_levels", [["L2", "L3", "L6"]])
@log_test_performance
def test_hierarchical_indices_levels_are_correct(
    naics_indices, expected_levels, test_run_id
):
    assert naics_indices["levels"] == expected_levels


@log_test_performance
def test_hierarchical_indices_code_levels_shape_is_correct(
    naics_indices, naics_codes, test_run_id
):
    assert naics_indices["code_levels"].shape == (len(naics_codes), 3)


@log_test_performance
def test_hierarchical_indices_single_l2_group(naics_indices, test_run_id):
    assert naics_indices["group_counts"][0] == 1


@log_test_performance
def test_hierarchical_indices_l3_group_count_is_at_least_one(
    naics_indices, test_run_id
):
    assert naics_indices["group_counts"][1] >= 1


@log_test_performance
def test_hierarchical_indices_parent_index_none_for_l2(naics_indices, test_run_id):
    assert naics_indices["parent_index_per_level"][0] is None


@log_test_performance
def test_hierarchical_indices_parent_index_is_ndarray_for_l3_and_l6(
    naics_indices, test_run_id
):
    assert isinstance(naics_indices["parent_index_per_level"][1], np.ndarray)
    assert isinstance(naics_indices["parent_index_per_level"][2], np.ndarray)


@log_test_performance
def test_hierarchical_indices_l3_groups_have_single_l2_parent(
    naics_indices, test_run_id
):
    assert np.all(naics_indices["parent_index_per_level"][1] == 0)


@log_test_performance
def test_hierarchical_indices_l3_labels_do_not_include_padded_l2_label(naics_indices):
    l3_labels = naics_indices["unique_per_level"][1]
    assert "52" not in set(map(str, l3_labels))


@pytest.mark.parametrize(
    "codes,cut_points,prefix_fill,expected_l2_groups",
    [
        (["00", "01", "02"], [2, 3], "0", 3),
        (["52", "521", "522110", "522120"], [2, 3, 6], "0", 1),
        ([""], [2], "0", 1),
        (["000", "001", "002"], [2, 3], "0", 1),
    ],
)
def test_hierarchical_indices_l2_group_count_various_cases(
    codes, cut_points, prefix_fill, expected_l2_groups
):
    idx = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=prefix_fill
    )
    assert idx["group_counts"][0] == expected_l2_groups


@pytest.fixture
def zip_codes():
    return ["02139", "0213", "02"]  # Cambridge ZIP with truncations


@pytest.fixture
def zip_indices(zip_codes):
    return build_hierarchical_indices(zip_codes, cut_points=[2, 3, 5], prefix_fill="0")


def test_prefix_padding_zip_l2_group_count_is_one(zip_indices):
    assert zip_indices["group_counts"][0] == 1


def test_prefix_padding_zip_l3_group_has_prefix_021(zip_indices):
    assert any(str(lbl).startswith("021") for lbl in zip_indices["unique_per_level"][1])


@pytest.mark.parametrize(
    "codes,cut_points,prefix_fill,expected_l2_groups",
    [
        (["00", "01", "02"], [2, 3], "0", 3),
        (["52", "521", "522110", "522120"], [2, 3, 6], "0", 1),
        ([""], [2], "0", 1),
        (["000", "001", "002"], [2, 3], "0", 1),
        (["02139", "0213", "02"], [2, 3, 5], "0", 1),
        (["1", "12", "123"], [1, 2, 3], "0", 1),
        ([""], [1], "0", 1),
    ],
)
def test_hierarchical_indices_l2_group_count_parametrized(
    codes, cut_points, prefix_fill, expected_l2_groups
):
    idx = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=prefix_fill
    )
    assert idx["group_counts"][0] == expected_l2_groups


@pytest.mark.parametrize(
    "codes,cut_points,prefix_fill,expected_l3_prefix",
    [
        (["02139", "0213", "02"], [2, 3, 5], "0", "021"),
        (["12345", "1234", "12"], [2, 4, 5], "0", "12"),
        (["00001", "0000", "00"], [2, 4, 5], "0", "00"),
    ],
)
def test_hierarchical_indices_l3_group_has_expected_prefix(
    codes, cut_points, prefix_fill, expected_l3_prefix
):
    idx = build_hierarchical_indices(
        codes, cut_points=cut_points, prefix_fill=prefix_fill
    )
    assert any(
        str(lbl).startswith(expected_l3_prefix) for lbl in idx["unique_per_level"][1]
    )
