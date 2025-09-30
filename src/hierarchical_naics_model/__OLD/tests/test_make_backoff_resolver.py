from __future__ import annotations


from hierarchical_naics_model.make_backoff_resolver import make_backoff_resolver
import pytest


@pytest.fixture
def resolver():
    cut_points = [2, 3, 4]
    level_maps = [
        {"12": 7, "51": 1},  # L2 indices
        {"123": 3, "511": 2},  # L3 indices
        {"1234": 5},  # L4 indices
    ]
    return make_backoff_resolver(
        cut_points=cut_points, level_maps=level_maps, prefix_fill="0"
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        ("1234", [7, 3, 5]),  # Exact match at all levels
        ("1239", [7, 3, 3]),  # Missing at deepest level, backoff to L3
        ("51", [1, 1, 1]),  # Short code, padded, backoff to L2
        ("99", [None, None, None]),  # Completely unseen code
        ("", [None, None, None]),  # Empty code
        ("12345", [7, 3, 5]),  # Longer code, slices to max cut_points
        (
            "511",
            [1, 2, 2],
        ),  # L2: '51'->1, L3: '511'->2, L4: '5110' not found, backoff to L2 idx
    ],
)
def test_make_backoff_resolver_returns_expected_indices_for_various_codes(
    resolver, code, expected
):
    assert resolver(code) == expected
