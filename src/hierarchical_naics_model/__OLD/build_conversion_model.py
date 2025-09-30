# hierarchical_conversion_model/model.py
from __future__ import annotations

from hierarchical_naics_model.types import Integers
from .model import (
    _validate_model_input_data,
    _get_naics_base,
    _noncentered_normal_rv,
    _create_index_from_levels,
    _add_group_coordinates_to_model,
)
import numpy as np
import pymc as pm


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
    y, L_naics, L_zip = _validate_model_input_data(
        y, naics_levels, zip_levels, naics_group_counts, zip_group_counts
    )

    def _build_naics_hierarchy(L_naics, naics_group_counts, naics_idx, use_student_t):
        if use_student_t:
            naics_mu_0 = pm.StudentT("naics_mu_0", nu=4.0, mu=0.0, sigma=0.5)
        else:
            naics_mu_0 = pm.Normal("naics_mu_0", 0.0, 1.0)
        naics_sigma_0 = pm.HalfNormal("naics_sigma_0", 0.6)
        naics_base = _get_naics_base(naics_group_counts, naics_mu_0, naics_sigma_0)
        pm.Deterministic("naics_eff_0", naics_base, dims=("NAICS_L0",))
        naics_contrib = naics_base[naics_idx[0]]
        for j in range(1, L_naics):
            pm.Deterministic(f"naics_mu_{j}", pm.math.constant(0.0))
            naics_sigma_j = pm.HalfNormal(
                f"naics_sigma_{j}", 0.4 if j < L_naics - 1 else 0.3
            )
            delta_offset = pm.Normal(
                f"naics_delta_{j}_offset", 0.0, 1.0, shape=(naics_group_counts[j],)
            )
            naics_delta_j = pm.Deterministic(
                f"naics_delta_{j}", delta_offset * naics_sigma_j
            )
            pm.Deterministic(f"naics_eff_{j}", naics_delta_j)
            naics_contrib = naics_contrib + naics_delta_j[naics_idx[j]]
        return naics_contrib

    def _build_zip_hierarchy(L_zip, zip_group_counts, zip_idx, use_student_t):
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
            zip_base = _noncentered_normal_rv(
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
        return zip_contrib

    with pm.Model() as model:
        _data = _create_observed_data(y)
        naics_idx = _create_index_from_levels(naics_levels, "naics")
        zip_idx = _create_index_from_levels(zip_levels, "zip")

        _add_group_coordinates_to_model("NAICS", naics_group_counts, L_naics, model)
        _add_group_coordinates_to_model("ZIP", zip_group_counts, L_zip, model)

        beta0 = pm.Normal("beta0", 0.0, 1.5)
        naics_contrib = _build_naics_hierarchy(
            L_naics, naics_group_counts, naics_idx, use_student_t
        )
        zip_contrib = _build_zip_hierarchy(
            L_zip, zip_group_counts, zip_idx, use_student_t
        )

        eta = pm.Deterministic(
            "eta", beta0 + naics_contrib + zip_contrib, dims=("obs_id",)
        )
        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims=("obs_id",))
        pm.Bernoulli("is_written", p=p, observed=model["y_obs"], dims=("obs_id",))

        model.default_sampling_kwargs = dict(
            draws=1000, tune=1000, chains=2, target_accept=target_accept
        )

    return model


def _create_observed_data(y):
    return pm.Data("y_obs", y, dims=("obs_id",))
