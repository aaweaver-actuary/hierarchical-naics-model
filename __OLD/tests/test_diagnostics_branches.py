import numpy as np
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)
from hierarchical_naics_model.diagnostics import compute_rhat, extract_observed
import arviz as az
import xarray as xr


@log_test_performance
def test_compute_rhat_fallback(test_run_id):
    # Simulate idata with missing summary
    arr = xr.DataArray(np.ones((2, 100)), dims=["chain", "draw"])
    idata = az.from_dict(posterior={"foo": arr})
    # Should fallback to az.rhat
    result = compute_rhat(idata, var_names=["foo"])
    assert "foo" in result
    assert result["foo"] == 1.0


@log_test_performance
def test_extract_observed_keyerror(test_run_id):
    # Simulate idata with missing observed_data
    arr = xr.DataArray(np.ones((2, 100)), dims=["chain", "draw"])
    idata = az.from_dict(posterior={"foo": arr})
    # Should return None if observed_data missing
    assert extract_observed(idata, observed_name="bar") is None
