# tests/smoke/test_pymc_nested_shapes.py
from __future__ import annotations

import numpy as np
import pytest
from hierarchical_naics_model.modeling import pymc_nested as pymc_nested_module
from hierarchical_naics_model.modeling.pymc_nested import (
    build_conversion_model_nested_deltas,
)
from hierarchical_naics_model.modeling.sampling import sample_prior_predictive

pm = pytest.importorskip("pymc")


def _toy_indices(N=60, J=3, M=2, seed=0):
    rng = np.random.default_rng(seed)
    # group counts per level
    naics_group_counts = [3, 5, 7][:J]
    zip_group_counts = [2, 4][:M]

    naics_levels = np.column_stack(
        [rng.integers(0, K, size=N, dtype=np.int64) for K in naics_group_counts]
    )
    zip_levels = np.column_stack(
        [rng.integers(0, K, size=N, dtype=np.int64) for K in zip_group_counts]
    )
    y = rng.integers(0, 2, size=N, dtype=np.int8)
    return y, naics_levels, zip_levels, naics_group_counts, zip_group_counts


@pytest.mark.parametrize("J,M", [(2, 2), (3, 2)])
def test_model_builds_and_prior_predictive_shapes(J, M):
    y, nlev, zlev, ngc, zgc = _toy_indices(N=50, J=J, M=M, seed=1)
    model = build_conversion_model_nested_deltas(
        y=y,
        naics_levels=nlev,
        zip_levels=zlev,
        naics_group_counts=ngc,
        zip_group_counts=zgc,
        target_accept=0.9,
        use_student_t_level0=False,
    )
    assert hasattr(model, "named_vars")

    # Key vars exist
    nv = model.named_vars
    assert "beta0" in nv
    assert "naics_base" in nv
    assert "zip_base" in nv
    for j in range(1, J):
        assert f"naics_delta_{j}" in nv
    for m in range(1, M):
        assert f"zip_delta_{m}" in nv
    assert "eta" in nv and "p" in nv and "is_written" in nv

    # Prior predictive: p in (0,1), correct shape
    idata = sample_prior_predictive(model, samples=64, random_seed=2)
    assert "prior" in idata.groups()
    p = idata.prior["p"].values  # shape (draw, obs)
    assert p.shape[-1] == y.shape[0]
    assert np.isfinite(p).all()
    assert (p > 0).all() and (p < 1).all()


def test_input_validation_errors():
    y, nlev, zlev, ngc, zgc = _toy_indices(N=20, J=2, M=2, seed=2)
    # y wrong dtype
    with pytest.raises(ValueError):
        build_conversion_model_nested_deltas(
            y=y.astype(float),
            naics_levels=nlev,
            zip_levels=zlev,
            naics_group_counts=ngc,
            zip_group_counts=zgc,
        )
    # non-binary y
    y_bad = y.copy()
    y_bad[0] = 3
    with pytest.raises(ValueError):
        build_conversion_model_nested_deltas(
            y=y_bad,
            naics_levels=nlev,
            zip_levels=zlev,
            naics_group_counts=ngc,
            zip_group_counts=zgc,
        )
    # shape mismatch
    with pytest.raises(ValueError):
        build_conversion_model_nested_deltas(
            y=y,
            naics_levels=nlev[:-1, :],
            zip_levels=zlev,
            naics_group_counts=ngc,
            zip_group_counts=zgc,
        )
    # out-of-bounds index
    nlev2 = nlev.copy()
    nlev2[0, 1] = ngc[1]  # invalid
    with pytest.raises(ValueError):
        build_conversion_model_nested_deltas(
            y=y,
            naics_levels=nlev2,
            zip_levels=zlev,
            naics_group_counts=ngc,
            zip_group_counts=zgc,
        )


def test_model_requires_pymc(monkeypatch):
    y = np.zeros(1, dtype=int)
    nlev = np.zeros((1, 1), dtype=int)
    zlev = np.zeros((1, 1), dtype=int)

    monkeypatch.setattr(pymc_nested_module, "pm", None)

    with pytest.raises(RuntimeError):
        pymc_nested_module.build_conversion_model_nested_deltas(
            y=y,
            naics_levels=nlev,
            zip_levels=zlev,
            naics_group_counts=[1],
            zip_group_counts=[1],
        )


def test_student_t_branch(monkeypatch):
    y = np.zeros(3, dtype=int)
    nlev = np.zeros((3, 1), dtype=int)
    zlev = np.zeros((3, 1), dtype=int)

    called: list[str] = []

    def recorder(name, **kwargs):
        called.append(name)
        return pm.Normal(name, mu=0.0, sigma=1.0, dims=kwargs.get("dims"))

    monkeypatch.setattr(pymc_nested_module.pm, "StudentT", recorder)

    model = pymc_nested_module.build_conversion_model_nested_deltas(
        y=y,
        naics_levels=nlev,
        zip_levels=zlev,
        naics_group_counts=[1],
        zip_group_counts=[1],
        use_student_t_level0=True,
    )

    assert "naics_base_offset" in called
    assert "zip_base_offset" in called
    assert "naics_base_offset" in model.named_vars
