from __future__ import annotations

from typing import Any, Dict

try:
    import pymc as pm
    import arviz as az
except Exception:  # pragma: no cover
    pm = None  # type: ignore
    az = None  # type: ignore


DEFAULT_SAMPLING: Dict[str, Any] = dict(
    draws=1000, tune=1000, chains=2, cores=1, target_accept=0.92, progressbar=False
)


def sample_posterior(model, **overrides):
    """
    Sample with package defaults, allowing test-time overrides.

    Parameters
    ----------
    model
        PyMC model context.
    **overrides
        Keyword args passed to `pm.sample`.

    Returns
    -------
    arviz.InferenceData
        Posterior (and optionally posterior predictive if requested).
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")
    kw = {**DEFAULT_SAMPLING, **overrides}
    with model:
        return pm.sample(**kw)
