from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import pymc as pm


def _noncentered_normal(name: str, mu, sigma, shape, dims=None):
    """
    Internal helper: non-centered parameterization for hierarchical effects.

    Parameters
    ----------
    name
        Base variable name.
    mu
        Mean (could be a scalar or per-level mean).
    sigma
        Positive scale (scalar).
    shape
        Shape of the vector of effects.
    dims
        Optional coords dims for PyMC.

    Returns
    -------
    rv : pm.Deterministic
        Deterministic transformed variable = mu + offset * sigma
    """
    offset = pm.Normal(f"{name}_offset", 0.0, 1.0, shape=shape, dims=dims)
    return pm.Deterministic(name, mu + offset * sigma, dims=dims)


def build_conversion_model(
    *,
    y: np.ndarray,  # shape (N,), 0/1
    naics_levels: np.ndarray,  # shape (N, L_naics), integer indices
    zip_levels: np.ndarray,  # shape (N, L_zip), integer indices
    naics_group_counts: Sequence[int],  # length L_naics
    zip_group_counts: Sequence[int],  # length L_zip
    coords: Dict[str, List[str]] | None = None,
    level_names_naics: Sequence[str] | None = None,
    level_names_zip: Sequence[str] | None = None,
    target_accept: float = 0.9,
    use_student_t: bool = False,
) -> "pm.Model":
    """
    Build a hierarchical partial-pooling logistic model with NAICS and ZIP hierarchies.

    Structure
    ---------
    logit(p_i) = β0
                 + sum_{ℓ in NAICS levels} a_naics[ℓ][ naics_levels[i, ℓ] ]
                 + sum_{m in ZIP    levels} a_zip  [m][ zip_levels[i,   m] ]

    with non-centered parameterization at each level ℓ/m.

    Parameters
    ----------
    y
        Binary vector (0/1) of shape (N,).
    naics_levels
        Integer indices per observation per NAICS level, shape (N, L_naics).
    zip_levels
        Integer indices per observation per ZIP level, shape (N, L_zip).
    naics_group_counts
        Number of groups at each NAICS level.
    zip_group_counts
        Number of groups at each ZIP level.
    coords
        Optional PyMC coords dict. If omitted, reasonable defaults are created.
    level_names_naics
        Optional names for NAICS levels (e.g., ["N_L2","N_L3","N_L6"]).
    level_names_zip
        Optional names for ZIP levels   (e.g., ["Z_L2","Z_L3","Z_L5"]).
    target_accept
        NUTS target_accept; 0.9–0.95 recommended for logistic models with
        hierarchical structure.
    use_student_t
        If True, use StudentT priors for level means; else Normal(0, 1).

    Returns
    -------
    model : pm.Model
        A compiled PyMC model (not sampled).
    """
    y = np.asarray(y, dtype="int8")
    N = y.shape[0]
    L_naics = naics_levels.shape[1]
    L_zip = zip_levels.shape[1]

    # Coords
    if coords is None:
        coords = {"obs_id": np.arange(N)}
        # Assign a coordinate array per level for NAICS & ZIP
        for j in range(L_naics):
            coords[f"NAICS_{j}"] = np.arange(naics_group_counts[j])
        for j in range(L_zip):
            coords[f"ZIP_{j}"] = np.arange(zip_group_counts[j])

    # Level names
    if level_names_naics is None:
        level_names_naics = [f"NAICS_L{j}" for j in range(L_naics)]
    if level_names_zip is None:
        level_names_zip = [f"ZIP_L{j}" for j in range(L_zip)]

    with pm.Model(coords=coords) as model:
        # Data containers
        y_obs = pm.Data("y_obs", y, dims=("obs_id",))
        naics_idx = []
        for j in range(L_naics):
            arr = pm.Data(f"naics_idx_{j}", naics_levels[:, j], dims=("obs_id",))
            naics_idx.append(arr)
        zip_idx = []
        for j in range(L_zip):
            arr = pm.Data(f"zip_idx_{j}", zip_levels[:, j], dims=("obs_id",))
            zip_idx.append(arr)

        # Global intercept
        beta0 = pm.Normal("beta0", 0.0, 1.5)

        # Hierarchical NAICS random intercepts across levels
        contrib_naics = 0.0
        for j in range(L_naics):
            # hyperpriors per level
            if use_student_t:
                mu = pm.StudentT(f"naics_mu_{j}", nu=4.0, mu=0.0, sigma=0.5)
            else:
                mu = pm.Normal(f"naics_mu_{j}", 0.0, 1.0)
            sigma = pm.HalfNormal(f"naics_sigma_{j}", 0.5)

            a_j = _noncentered_normal(
                f"naics_eff_{j}",
                mu=mu,
                sigma=sigma,
                shape=(naics_group_counts[j],),
                dims=(f"NAICS_{j}",),
            )
            contrib_naics = contrib_naics + a_j[naics_idx[j]]

        # Hierarchical ZIP random intercepts across levels
        contrib_zip = 0.0
        for j in range(L_zip):
            if use_student_t:
                mu = pm.StudentT(f"zip_mu_{j}", nu=4.0, mu=0.0, sigma=0.5)
            else:
                mu = pm.Normal(f"zip_mu_{j}", 0.0, 1.0)
            sigma = pm.HalfNormal(f"zip_sigma_{j}", 0.5)

            b_j = _noncentered_normal(
                f"zip_eff_{j}",
                mu=mu,
                sigma=sigma,
                shape=(zip_group_counts[j],),
                dims=(f"ZIP_{j}",),
            )
            contrib_zip = contrib_zip + b_j[zip_idx[j]]

        # Linear predictor and likelihood
        eta = pm.Deterministic(
            "eta", beta0 + contrib_naics + contrib_zip, dims=("obs_id",)
        )
        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims=("obs_id",))
        pm.Bernoulli("is_written", p=p, observed=y_obs, dims=("obs_id",))

        # Store a default sampling config on the model for convenience
        model.default_sampling_kwargs = dict(
            draws=1000, tune=1000, chains=2, target_accept=target_accept
        )

    return model
