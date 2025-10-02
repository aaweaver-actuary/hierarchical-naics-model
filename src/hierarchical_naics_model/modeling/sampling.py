# nested_quotewrite/modeling/sampling.py
from __future__ import annotations

from typing import Any, Dict

try:
    import pymc as pm
except Exception:  # pragma: no cover
    pm = None  # type: ignore

DEFAULT_SAMPLING: Dict[str, Any] = dict(
    draws=1000, tune=1000, chains=2, cores=1, target_accept=0.92, progressbar=False
)


def sample_posterior(model, **overrides):
    """
    Sample the posterior using package defaults, allowing test-time overrides.

    Returns
    -------
    arviz.InferenceData
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")
    kw = {**getattr(model, "default_sampling_kwargs", DEFAULT_SAMPLING), **overrides}
    with model:
        return pm.sample(**kw)


def sample_prior_predictive(model, samples: int = 200, random_seed: int = 42):
    """
    Draw prior predictive samples (fast smoke tests / quick sanity).

    Returns
    -------
    arviz.InferenceData
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")
    with model:
        return pm.sample_prior_predictive(draws=samples, random_seed=random_seed)
