import pymc as pm


def _create_index_from_levels(levels, level_name="naics"):
    length = levels.shape[1]
    return [
        pm.Data(f"{level_name}_idx_{j}", levels[:, j], dims=("obs_id",))
        for j in range(length)
    ]
