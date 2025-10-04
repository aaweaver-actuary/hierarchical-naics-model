import numpy as np
import pytest
from hierarchical_naics_model.core.effects import eta_nested


def test_eta_nested_empty_levels():
    beta0 = 1.0
    naics_base = np.array([0.1, -0.1], dtype=float)
    naics_deltas = [np.array([0.0, 0.05, -0.05], dtype=float)]
    zip_base = np.array([0.2], dtype=float)
    zip_deltas = [np.array([-0.1, 0.0], dtype=float)]
    naics_levels = np.zeros((2, 0), dtype=int)  # J=0
    zip_levels = np.zeros((2, 2), dtype=int)
    with pytest.raises(ValueError, match="must each have at least one level"):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )
    naics_levels = np.zeros((2, 2), dtype=int)
    zip_levels = np.zeros((2, 0), dtype=int)  # M=0
    with pytest.raises(ValueError, match="must each have at least one level"):
        eta_nested(
            beta0,
            naics_base,
            naics_deltas,
            zip_base,
            zip_deltas,
            naics_levels,
            zip_levels,
        )
