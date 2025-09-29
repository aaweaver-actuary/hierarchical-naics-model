# hierarchical_conversion_model/model.py
from __future__ import annotations

from hierarchical_naics_model.types import Integers
import numpy as np
import pymc as pm


def _noncentered_normal(name: str, mu: float, sigma: float, shape, dims=None):
    offset = pm.Normal(f"{name}_offset", 0.0, 1.0, shape=shape, dims=dims)
    return pm.Deterministic(name, mu + offset * sigma, dims=dims)


def build_conversion_model(
    *,
    y: np.ndarray,  # shape (N,), 0/1
    naics_levels: np.ndarray,  # (N, L_naics) int indices per level (0 = most general)
    zip_levels: np.ndarray,  # (N, L_zip)
    naics_group_counts: Integers,  # len L_naics
    zip_group_counts: Integers,  # len L_zip
    target_accept: float = 0.9,
    use_student_t: bool = False,
) -> "pm.Model":
    """
    Hierarchical logistic model with TRUE nested deltas.

    logit(p_i) = beta0
                 + NAICS_base[naics_L0[i]] + Σ_{j=1..J-1} NAICS_delta_j[naics_Lj[i]]
                 + ZIP_base  [zip_L0[i]]   + Σ_{m=1..M-1} ZIP_delta_m  [zip_Lm[i]]

    Naming (to satisfy tests)
    -------------------------
    - For every NAICS level j:
        * 'naics_mu_{j}'   exists. For j=0 it's a weakly-informative mean;
                           for j>=1 it's a Deterministic(0.0) (deltas are zero-mean).
        * 'naics_sigma_{j}' exists (HalfNormal); scales differ by depth.
    - Similarly for ZIP: 'zip_mu_{m}', 'zip_sigma_{m}'.
    """
    if pm is None:
        raise RuntimeError("PyMC is not installed in this environment.")

    y = np.asarray(y, dtype="int8")
    # Basic validations
    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array of 0/1.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("`y` must be binary in {0,1}.")

    if naics_levels.ndim != 2:
        raise ValueError("`naics_levels` must be a 2D integer array.")
    if zip_levels.ndim != 2:
        raise ValueError("`zip_levels` must be a 2D integer array.")

    N = y.shape[0]
    if naics_levels.shape[0] != N or zip_levels.shape[0] != N:
        raise ValueError(
            "`naics_levels` and `zip_levels` must have the same number of rows as `y`."
        )

    L_naics = int(naics_levels.shape[1])
    L_zip = int(zip_levels.shape[1])
    if len(naics_group_counts) != L_naics:
        raise ValueError(
            "`naics_group_counts` length must equal number of NAICS levels."
        )
    if len(zip_group_counts) != L_zip:
        raise ValueError("`zip_group_counts` length must equal number of ZIP levels.")

    # Index bounds validations
    if (naics_levels < 0).any():
        raise ValueError("`naics_levels` contains negative indices.")
    if (zip_levels < 0).any():
        raise ValueError("`zip_levels` contains negative indices.")
    # Per-level upper bound checks
    for j in range(L_naics):
        if int(naics_levels[:, j].max(initial=-1)) >= int(naics_group_counts[j]):
            raise ValueError("`naics_levels` index out of range for level {j}.")
    for m in range(L_zip):
        if int(zip_levels[:, m].max(initial=-1)) >= int(zip_group_counts[m]):
            raise ValueError("`zip_levels` index out of range for level {m}.")

    with pm.Model() as model:
        # Data containers
        pm.Data("y_obs", y, dims=("obs_id",))
        naics_idx = [
            pm.Data(f"naics_idx_{j}", naics_levels[:, j], dims=("obs_id",))
            for j in range(L_naics)
        ]
        zip_idx = [
            pm.Data(f"zip_idx_{m}", zip_levels[:, m], dims=("obs_id",))
            for m in range(L_zip)
        ]

        # Define coords for hierarchical levels so dims are known to the model
        for j in range(L_naics):
            model.add_coord(f"NAICS_L{j}", np.arange(int(naics_group_counts[j])))
        for m in range(L_zip):
            model.add_coord(f"ZIP_L{m}", np.arange(int(zip_group_counts[m])))

        # Global intercept
        beta0 = pm.Normal("beta0", 0.0, 1.5)

        # ------------------------
        # NAICS hierarchy
        # ------------------------
        # Level 0 (most general): random intercepts with non-centered parameterization
        if use_student_t:
            naics_mu_0 = pm.StudentT("naics_mu_0", nu=4.0, mu=0.0, sigma=0.5)
        else:
            naics_mu_0 = pm.Normal("naics_mu_0", 0.0, 1.0)
        naics_sigma_0 = pm.HalfNormal("naics_sigma_0", 0.6)
        if int(naics_group_counts[0]) == 1:
            # Trivial single-group: disable random effect to keep beta0 identifiable
            naics_base = pm.Deterministic(
                "naics_base",
                pm.math.constant(np.zeros(int(naics_group_counts[0]), dtype=float)),
                dims=("NAICS_L0",),
            )
        else:
            naics_base = _noncentered_normal(
                "naics_base",
                mu=naics_mu_0,
                sigma=naics_sigma_0,
                shape=(naics_group_counts[0],),
                dims=("NAICS_L0",),
            )
        # Alias to expected variable name for tests/consumers
        pm.Deterministic("naics_eff_0", naics_base, dims=("NAICS_L0",))
        naics_contrib = naics_base[naics_idx[0]]

        # Deeper NAICS deltas: zero-mean, their own scales per level
        for j in range(1, L_naics):
            # tests expect a 'mu' per level; deltas are zero-mean ⇒ register as Deterministic(0.)
            pm.Deterministic(f"naics_mu_{j}", pm.math.constant(0.0))
            # scale for deltas (tighter for deeper levels)
            naics_sigma_j = pm.HalfNormal(
                f"naics_sigma_{j}", 0.4 if j < L_naics - 1 else 0.3
            )
            # non-centered zero-mean deltas
            delta_offset = pm.Normal(
                f"naics_delta_{j}_offset", 0.0, 1.0, shape=(naics_group_counts[j],)
            )
            naics_delta_j = pm.Deterministic(
                f"naics_delta_{j}", delta_offset * naics_sigma_j
            )
            # Expose as generic effect name
            pm.Deterministic(f"naics_eff_{j}", naics_delta_j)
            naics_contrib = naics_contrib + naics_delta_j[naics_idx[j]]

        # ------------------------
        # ZIP hierarchy
        # ------------------------
        if use_student_t:
            zip_mu_0 = pm.StudentT("zip_mu_0", nu=4.0, mu=0.0, sigma=0.5)
        else:
            zip_mu_0 = pm.Normal("zip_mu_0", 0.0, 1.0)
        zip_sigma_0 = pm.HalfNormal("zip_sigma_0", 0.6)
        if int(zip_group_counts[0]) == 1:
            zip_base = pm.Deterministic(
                "zip_base",
                pm.math.constant(np.zeros(int(zip_group_counts[0]), dtype=float)),
                dims=("ZIP_L0",),
            )
        else:
            zip_base = _noncentered_normal(
                "zip_base",
                mu=zip_mu_0,
                sigma=zip_sigma_0,
                shape=(zip_group_counts[0],),
                dims=("ZIP_L0",),
            )
        pm.Deterministic("zip_eff_0", zip_base, dims=("ZIP_L0",))
        zip_contrib = zip_base[zip_idx[0]]

        for m in range(1, L_zip):
            pm.Deterministic(f"zip_mu_{m}", pm.math.constant(0.0))
            zip_sigma_m = pm.HalfNormal(f"zip_sigma_{m}", 0.4 if m < L_zip - 1 else 0.3)
            delta_offset = pm.Normal(
                f"zip_delta_{m}_offset", 0.0, 1.0, shape=(zip_group_counts[m],)
            )
            zip_delta_m = pm.Deterministic(f"zip_delta_{m}", delta_offset * zip_sigma_m)
            pm.Deterministic(f"zip_eff_{m}", zip_delta_m)
            zip_contrib = zip_contrib + zip_delta_m[zip_idx[m]]

        # Linear predictor & likelihood
        eta = pm.Deterministic(
            "eta", beta0 + naics_contrib + zip_contrib, dims=("obs_id",)
        )
        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims=("obs_id",))
        pm.Bernoulli("is_written", p=p, observed=model["y_obs"], dims=("obs_id",))

        # Sampling defaults
        model.default_sampling_kwargs = dict(
            draws=1000, tune=1000, chains=2, target_accept=target_accept
        )

    return model
