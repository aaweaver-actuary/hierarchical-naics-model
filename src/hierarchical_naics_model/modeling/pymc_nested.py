from __future__ import annotations

import numpy as np
import pymc as pm

from ..types import Integers
from ..utils import exp, sigmoid, check_inputs


def build_conversion_model_nested_deltas(
    *,
    y: np.ndarray,
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
    naics_group_counts: Integers,
    zip_group_counts: Integers,
    target_accept: float = 0.92,
    use_student_t_level0: bool = False,
):
    """
    Build a PyMC logistic model with base + nested deltas for NAICS and ZIP.

    Parameters
    ----------
    y
        Binary outcomes (N,) as 0/1 int array.
    naics_levels
        Integer indices (N, J) across NAICS hierarchy levels. Column 0 indexes the base,
        columns 1..J-1 index the corresponding delta vectors.
    zip_levels
        Integer indices (N, M) across ZIP hierarchy levels (same convention).
    naics_group_counts
        Group counts per NAICS level (len J).
    zip_group_counts
        Group counts per ZIP level (len M).
    target_accept
        NUTS target_accept to store in `model.default_sampling_kwargs`.
    use_student_t_level0
        If True, the base vectors are scaled StudentT (nu=4) rather than Normal.

    Returns
    -------
    pm.Model
        Model with named variables:
        - beta0
        - naics_base, naics_delta_1..naics_delta_{J-1}
        - zip_base,   zip_delta_1..zip_delta_{M-1}
        - eta, p
        - is_written (Bernoulli likelihood)
        - pm.Data containers: y_obs, naics_idx_0.., zip_idx_0..
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")

    check_inputs(y, naics_levels, zip_levels, naics_group_counts, zip_group_counts)

    N = y.shape[0]
    J = naics_levels.shape[1]
    M = zip_levels.shape[1]

    # Coordinates for readability
    obs_coord = np.arange(N)
    coords = {"obs": obs_coord}
    # Add group coords per level
    for j, K in enumerate(naics_group_counts):
        coords[f"naics_g{j}"] = np.arange(K)
    for m, K in enumerate(zip_group_counts):
        coords[f"zip_g{m}"] = np.arange(K)

    with pm.Model(coords=coords) as model:
        # Data containers
        pm.Data("y_obs", y, dims=("obs",))
        for j in range(J):
            pm.Data(f"naics_idx_{j}", naics_levels[:, j], dims=("obs",))
        for m in range(M):
            pm.Data(f"zip_idx_{m}", zip_levels[:, m], dims=("obs",))

        # Global intercept
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)

        # --- NAICS hierarchy ---
        # Base level (j=0)
        _K0 = naics_group_counts[0]
        if use_student_t_level0:
            naics_base_offset = pm.StudentT(
                "naics_base_offset", nu=4.0, mu=0.0, sigma=1.0, dims=("naics_g0",)
            )
        else:
            naics_base_offset = pm.Normal(
                "naics_base_offset", mu=0.0, sigma=1.0, dims=("naics_g0",)
            )
        sigma_naics_base = pm.HalfNormal("naics_sigma_base", sigma=1.0)
        naics_base = pm.Deterministic(
            "naics_base",
            naics_base_offset * sigma_naics_base,  # type: ignore
            dims=("naics_g0",),
        )

        # Depth shrinkage via exponential decay from base scale
        kappa_naics = pm.HalfNormal("naics_kappa", sigma=1.0)
        if J > 1:
            for j in range(1, J):
                _Kj = naics_group_counts[j]

                sigma_j = pm.Deterministic(
                    f"naics_sigma_{j}",
                    sigma_naics_base * exp(-kappa_naics * j),  # type: ignore
                )
                offset = pm.Normal(
                    f"naics_delta_{j}_offset", mu=0.0, sigma=1.0, dims=(f"naics_g{j}",)
                )
                pm.Deterministic(
                    f"naics_delta_{j}",
                    offset * sigma_j,  # type: ignore
                    dims=(f"naics_g{j}",),
                )

        # --- ZIP hierarchy ---
        # Base
        _Z0 = zip_group_counts[0]
        if use_student_t_level0:
            zip_base_offset = pm.StudentT(
                "zip_base_offset", nu=4.0, mu=0.0, sigma=1.0, dims=("zip_g0",)
            )
        else:
            zip_base_offset = pm.Normal(
                "zip_base_offset", mu=0.0, sigma=1.0, dims=("zip_g0",)
            )
        sigma_zip_base = pm.HalfNormal("zip_sigma_base", sigma=1.0)
        zip_base = pm.Deterministic(
            "zip_base",
            zip_base_offset * sigma_zip_base,  # type: ignore
            dims=("zip_g0",),
        )

        kappa_zip = pm.HalfNormal("zip_kappa", sigma=1.0)
        if M > 1:
            for m in range(1, M):
                _Km = zip_group_counts[m]
                sigma_m = pm.Deterministic(
                    f"zip_sigma_{m}",
                    sigma_zip_base * exp(-kappa_zip * m),  # type: ignore
                )
                offset = pm.Normal(
                    f"zip_delta_{m}_offset", mu=0.0, sigma=1.0, dims=(f"zip_g{m}",)
                )
                pm.Deterministic(
                    f"zip_delta_{m}",
                    offset * sigma_m,
                    dims=(f"zip_g{m}",),  # type: ignore
                )

        # Linear predictor: base + deltas (NAICS & ZIP)
        eta = beta0
        # base contributions
        eta = eta + naics_base[model["naics_idx_0"]]
        eta = eta + zip_base[model["zip_idx_0"]]

        # deltas
        for j in range(1, J):
            eta = eta + model[f"naics_delta_{j}"][model[f"naics_idx_{j}"]]
        for m in range(1, M):
            eta = eta + model[f"zip_delta_{m}"][model[f"zip_idx_{m}"]]

        pm.Deterministic("eta", eta, dims=("obs",))

        # Convert logit to probability
        p = pm.Deterministic("p", sigmoid(eta), dims=("obs",))

        # Likelihood
        pm.Bernoulli("is_written", p=p, observed=model["y_obs"], dims=("obs",))

        # Default sampler settings for convenience (tests/CLI can override)
        model.default_sampling_kwargs = dict(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=float(target_accept),
            progressbar=False,
        )

    return model
