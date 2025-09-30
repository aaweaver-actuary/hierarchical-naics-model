# tests/slow/test_pymc_nested_sample.py
from __future__ import annotations

import numpy as np
import pytest
from hierarchical_naics_model.modeling.pymc_nested import (
    build_conversion_model_nested_deltas,
)


pm = pytest.importorskip("pymc")
az = pytest.importorskip("arviz")


pytestmark = pytest.mark.slow


def _tiny_data(N=80, J=2, M=2, seed=0):
    rng = np.random.default_rng(seed)
    ngc = [2, 3][:J]
    zgc = [2, 2][:M]
    nlev = np.column_stack([rng.integers(0, K, size=N, dtype=np.int64) for K in ngc])
    zlev = np.column_stack([rng.integers(0, K, size=N, dtype=np.int64) for K in zgc])
    # Mild signal for numerical stability in tiny runs
    y = rng.integers(0, 2, size=N, dtype=np.int8)
    return y, nlev, zlev, ngc, zgc


def test_short_sampling_runs_and_no_explosions():
    y, nlev, zlev, ngc, zgc = _tiny_data(N=60, J=2, M=2, seed=11)
    model = build_conversion_model_nested_deltas(
        y=y,
        naics_levels=nlev,
        zip_levels=zlev,
        naics_group_counts=ngc,
        zip_group_counts=zgc,
        target_accept=0.85,
    )
    with model:
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=1,
            cores=1,
            target_accept=0.85,
            progressbar=False,
            compute_convergence_checks=False,
            random_seed=123,
        )

    # Basic posterior presence
    assert "posterior" in idata.groups()
    for v in ("beta0", "naics_base", "zip_base"):
        assert v in idata.posterior

    # Divergences (allow a few at most on tiny runs)
    if "diverging" in idata.sample_stats:
        div = int(idata.sample_stats["diverging"].sum().values)
        assert div <= 2  # be lenient for small stress runs

    # p from posterior predictive (optional quick check)
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=["p"], random_seed=123)
    assert "p" in ppc.posterior_predictive
    p = ppc.posterior_predictive["p"].values
    assert np.isfinite(p).all()
    assert (p > 0).all() & (p < 1).all()
