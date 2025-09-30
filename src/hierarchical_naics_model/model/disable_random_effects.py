import pymc as pm
import numpy as np
from ..types import Integers


def _disable_random_effects(
    group_name: str, group_counts: Integers
) -> pm.Deterministic:
    return pm.Deterministic(
        "naics_base",
        pm.math.constant(np.zeros(int(group_counts[0]), dtype=float)),
        dims=("NAICS_L0",),
    )
