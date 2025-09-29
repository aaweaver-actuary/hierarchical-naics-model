import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from hierarchical_naics_model.extract_effect_tables import extract_effect_tables


def test_extract_effect_tables_custom_names():
    # Simulate posterior with expected keys
    arr = xr.DataArray(np.ones((2, 100, 3)), dims=["chain", "draw", "NAICS_L0"])
    idata = az.from_dict(
        posterior={
            "naics_base": arr,
            "zip_base": arr,
            "beta0": xr.DataArray(np.ones((2, 100)), dims=["chain", "draw"]),
            "naics_delta_1": arr,
            "zip_delta_1": arr,
        }
    )
    out = extract_effect_tables(
        idata, naics_level_names=["A", "B"], zip_level_names=["X", "Y"]
    )
    assert "naics_tables" in out
    assert "zip_tables" in out
    assert isinstance(out["naics_tables"], list)
    assert isinstance(out["zip_tables"], list)
    assert out["naics_tables"][0].name == "naics_base"
    assert out["zip_tables"][0].name == "zip_base"
