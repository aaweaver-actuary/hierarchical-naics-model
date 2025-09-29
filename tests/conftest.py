import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pytest


# Ensure the src/ layout is importable during tests without installation
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def naics_pool() -> Sequence[str]:
    # Mix of lengths to exercise prefix handling
    return [
        "511110",
        "511120",
        "51213",  # 5 chars
        "52",  # 2 chars
        "52412",
        "52413",
    ]


@pytest.fixture(scope="session")
def zip_pool() -> Sequence[str]:
    return [
        "30309",
        "94103",
        "10001",
        "02139",
        "73301",
    ]


@pytest.fixture(scope="session")
def naics_cut_points() -> List[int]:
    # Typical NAICS like 2,3,6
    return [2, 3, 6]


@pytest.fixture(scope="session")
def zip_cut_points() -> List[int]:
    return [2, 3, 5]


@pytest.fixture(scope="session")
def synthetic_df(naics_pool, zip_pool):
    from hierarchical_naics_model.generate_synthetic_data import (
        generate_synthetic_data,
    )

    df = generate_synthetic_data(
        1000, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-1.2, seed=123
    )
    return df


@pytest.fixture(scope="session")
def naics_indices(synthetic_df, naics_cut_points):
    from hierarchical_naics_model.build_hierarchical_indices import (
        build_hierarchical_indices,
    )

    return build_hierarchical_indices(
        list(synthetic_df["naics"].astype(str)), cut_points=naics_cut_points
    )


@pytest.fixture(scope="session")
def zip_indices(synthetic_df, zip_cut_points):
    from hierarchical_naics_model.build_hierarchical_indices import (
        build_hierarchical_indices,
    )

    return build_hierarchical_indices(
        list(synthetic_df["zip"].astype(str)), cut_points=zip_cut_points
    )


@pytest.fixture(scope="session")
def model_inputs(synthetic_df, naics_indices, zip_indices) -> Dict[str, object]:
    y = synthetic_df["is_written"].to_numpy()
    naics_levels = naics_indices["code_levels"]
    zip_levels = zip_indices["code_levels"]
    naics_group_counts = naics_indices["group_counts"]
    zip_group_counts = zip_indices["group_counts"]
    return dict(
        y=y,
        naics_levels=naics_levels,
        zip_levels=zip_levels,
        naics_group_counts=naics_group_counts,
        zip_group_counts=zip_group_counts,
    )
