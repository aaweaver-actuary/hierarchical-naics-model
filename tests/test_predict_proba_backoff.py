import sys

sys.path.insert(0, "../src")
import pandas as pd
import numpy as np
from hierarchical_naics_model.logger import get_logger
from hierarchical_naics_model.predict_proba import predict_proba, serialize_level_maps

log = get_logger(__name__)


def test_predict_proba_backoff_and_flags():
    # Three codes, one is unseen at all levels
    df = pd.DataFrame(
        {
            "naics": ["100", "200", "999"],
            "zip": ["111", "222", "888"],
        }
    )
    # Maps: only first two codes are known at each level
    naics_level_maps = [
        {"1": 0, "2": 1},
        {"10": 0, "20": 1},
        {"100": 0, "200": 1},
    ]
    zip_level_maps = [
        {"1": 0, "2": 1},
        {"11": 0, "22": 1},
        {"111": 0, "222": 1},
    ]
    effects = {
        "beta0": 0.0,
        "naics_base": pd.Series([0.1, -0.1], index=[0, 1]),
        "naics_deltas": [
            pd.Series([0.05, -0.05], index=[0, 1]),
            pd.Series([0.02, -0.02], index=[0, 1]),
        ],
        "zip_base": pd.Series([0.2, -0.2], index=[0, 1]),
        "zip_deltas": [
            pd.Series([0.01, -0.01], index=[0, 1]),
            pd.Series([0.03, -0.03], index=[0, 1]),
        ],
    }
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects,
        prefix_fill="0",
    )
    # First two codes: all levels known, last code: all levels unknown
    for i in range(2):
        for j in range(3):
            assert not out[f"backoff_naics_{j}"][i]
            assert not out[f"backoff_zip_{j}"][i]
    for j in range(3):
        assert out[f"backoff_naics_{j}"][2]
        assert out[f"backoff_zip_{j}"][2]
    # Eta and p should be finite
    assert np.isfinite(out["eta"]).all()
    assert np.isfinite(out["p"]).all()
    # Unseen code: eta should be just beta0
    assert np.isclose(out["eta"][2], effects["beta0"])


def test_predict_proba_partial_backoff():
    # Some codes are only known at parent level
    # Use codes that only exist at parent level (not in level 1 or 2 maps)
    df = pd.DataFrame({"naics": ["103", "203"], "zip": ["117", "227"]})
    naics_level_maps = [
        {"1": 0, "2": 1},  # Only root codes present
        {},  # No codes at level 1
        {},  # No codes at level 2
    ]
    zip_level_maps = [
        {"1": 0, "2": 1},
        {},
        {},
    ]
    effects = {
        "beta0": 0.0,
        "naics_base": pd.Series([0.1, -0.1], index=[0, 1]),
        "naics_deltas": [
            pd.Series([0.05, -0.05], index=[0, 1]),
            pd.Series([0.02, -0.02], index=[0, 1]),
        ],
        "zip_base": pd.Series([0.2, -0.2], index=[0, 1]),
        "zip_deltas": [
            pd.Series([0.01, -0.01], index=[0, 1]),
            pd.Series([0.03, -0.03], index=[0, 1]),
        ],
    }
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects,
        prefix_fill="0",
    )
    # Extra debug output
    log.debug(f"predict_proba output:\n{out}")
    log.debug(
        f"naics_idx: {out[['backoff_naics_0', 'backoff_naics_1', 'backoff_naics_2']].values.tolist()}"
    )
    log.debug(
        f"zip_idx: {out[['backoff_zip_0', 'backoff_zip_1', 'backoff_zip_2']].values.tolist()}"
    )
    for i in range(2):
        log.debug(
            f"Row {i} backoff_naics: {[out[f'backoff_naics_{j}'][i] for j in range(3)]}"
        )
        log.debug(
            f"Row {i} backoff_zip: {[out[f'backoff_zip_{j}'][i] for j in range(3)]}"
        )
    # All levels should be backoff except parent level
    for i in range(2):
        print(
            f"Row {i} backoff_naics: {[out[f'backoff_naics_{j}'][i] for j in range(3)]}"
        )
        print(f"Row {i} backoff_zip: {[out[f'backoff_zip_{j}'][i] for j in range(3)]}")
        # Only parent level (level 0) is known
        assert not out[f"backoff_naics_{0}"][i]
        assert not out[f"backoff_zip_{0}"][i]
        for j in [1, 2]:
            assert out[f"backoff_naics_{j}"][i]
            assert out[f"backoff_zip_{j}"][i]
    # Eta should include only base contributions
    for i in range(2):
        expected_eta = effects["beta0"]
        expected_eta += float(
            effects["naics_base"].loc[naics_level_maps[0][df["naics"][i][0]]]
        )
        expected_eta += float(
            effects["zip_base"].loc[zip_level_maps[0][df["zip"][i][0]]]
        )
        assert np.isclose(out["eta"][i], expected_eta)


def test_predict_proba_empty_df():
    # Edge case: empty input
    df = pd.DataFrame({"naics": [], "zip": []})
    naics_level_maps = [{}, {}, {}]
    zip_level_maps = [{}, {}, {}]
    effects = {
        "beta0": 0.0,
        "naics_base": pd.Series([], dtype=float),
        "naics_deltas": [pd.Series([], dtype=float), pd.Series([], dtype=float)],
        "zip_base": pd.Series([], dtype=float),
        "zip_deltas": [pd.Series([], dtype=float), pd.Series([], dtype=float)],
    }
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects,
        prefix_fill="0",
    )
    assert out.shape[0] == 0


def test_serialize_level_maps():
    # Should convert all per-level dicts to plain dicts
    level_maps = [
        {"a": 1, "b": 2},
        {"c": 3},
    ]
    out = serialize_level_maps(level_maps)
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, dict)
        assert all(isinstance(k, str) for k in d.keys())
