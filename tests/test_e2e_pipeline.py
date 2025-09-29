from __future__ import annotations

import numpy as np
import pymc as pm

from hierarchical_naics_model.generate_synthetic_data import generate_synthetic_data
from hierarchical_naics_model.build_hierarchical_indices import (
    build_hierarchical_indices,
)
from hierarchical_naics_model.build_conversion_model import build_conversion_model


def test_end_to_end_pipeline():
    # 1) Generate synthetic input data
    naics_pool = ["511110", "511120", "51213", "52", "52412", "52413"]
    zip_pool = ["30309", "94103", "10001", "02139", "73301"]
    df = generate_synthetic_data(
        n=500, naics_codes=naics_pool, zip_codes=zip_pool, base_logit=-1.2, seed=42
    )

    # 2) Build hierarchical indices for NAICS and ZIP
    naics_cuts = [2, 3, 6]
    zip_cuts = [2, 3, 5]
    naics_idx = build_hierarchical_indices(
        df["naics"].astype(str).tolist(), cut_points=naics_cuts
    )
    zip_idx = build_hierarchical_indices(
        df["zip"].astype(str).tolist(), cut_points=zip_cuts
    )

    # Basic checks
    assert naics_idx["code_levels"].shape[0] == len(df)
    assert zip_idx["code_levels"].shape[0] == len(df)
    assert len(naics_idx["group_counts"]) == len(naics_cuts)
    assert len(zip_idx["group_counts"]) == len(zip_cuts)

    # 3) Build and sample the model on a subset to keep runtime modest
    y = df["is_written"].to_numpy()
    take = np.arange(min(200, len(df)))
    model = build_conversion_model(
        y=y[take],
        naics_levels=naics_idx["code_levels"][take, :],
        zip_levels=zip_idx["code_levels"][take, :],
        naics_group_counts=naics_idx["group_counts"],
        zip_group_counts=zip_idx["group_counts"],
        target_accept=0.9,
    )

    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=1,
            random_seed=123,
            progressbar=False,
        )

    # 4) Validate posterior contains expected random variables and shapes
    assert "posterior" in idata.groups()
    for key in ["beta0", "eta", "p"]:
        assert key in idata.posterior or key in idata.posterior.coords
