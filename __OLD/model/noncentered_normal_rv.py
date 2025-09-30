import pymc as pm


def _noncentered_normal_rv(name: str, mu: float, sigma: float, shape: tuple, dims=None):
    """Return a non-centered parameterization of a Normal RV."""
    offset = pm.Normal(f"{name}_offset", 0.0, 1.0, shape=shape, dims=dims)
    return pm.Deterministic(name, mu + offset * sigma, dims=dims)
