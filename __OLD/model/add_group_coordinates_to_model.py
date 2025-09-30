import numpy as np
import pymc as pm
from ..types import Integers


def _add_group_coordinates_to_model(
    group_name: str, group_counts: Integers, length: int, model: pm.Model
):
    for j in range(length):
        model.add_coord(f"{group_name}_L{j}", np.arange(int(group_counts[j])))
