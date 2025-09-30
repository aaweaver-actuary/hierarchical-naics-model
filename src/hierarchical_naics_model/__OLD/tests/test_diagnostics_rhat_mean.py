from __future__ import annotations

import pandas as pd
import pymc as pm

from hierarchical_naics_model.diagnostics import compute_rhat
from hierarchical_naics_model.build_conversion_model import build_conversion_model


def test_compute_rhat_uses_rhat_mean(monkeypatch, model_inputs):
    # Monkeypatch az.summary to return a DataFrame with r_hat_mean column
    import arviz as az

    def fake_summary(*_a, **_k):
        df = pd.DataFrame({"r_hat_mean": [1.02]}, index=["beta0"])
        return df

    monkeypatch.setattr(az, "summary", fake_summary)

    model = build_conversion_model(**model_inputs)  # type: ignore[missing-argument]
    with model:
        idata = pm.sample(
            draws=20, tune=20, chains=2, cores=1, progressbar=False, random_seed=21
        )

    rh = compute_rhat(idata, var_names=["beta0"])  # should pick r_hat_mean branch
    assert "beta0" in rh and abs(rh["beta0"] - 1.02) < 1e-9
