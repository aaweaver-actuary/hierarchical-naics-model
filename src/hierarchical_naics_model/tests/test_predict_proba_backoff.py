import pandas as pd
import numpy as np
from hierarchical_naics_model.logger import get_logger
from hierarchical_naics_model.predict_proba import predict_proba, serialize_level_maps
import pytest
from hierarchical_naics_model.tests.test_performance_decorator import (
    log_test_performance,
)

log = get_logger(__name__)


@pytest.fixture
def effects_fixture():
    return {
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


@pytest.fixture
def naics_zip_level_maps_fixture():
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
    return naics_level_maps, zip_level_maps


@pytest.fixture
def test_df_fixture():
    return pd.DataFrame({"naics": ["100", "200", "999"], "zip": ["111", "222", "888"]})


@pytest.mark.parametrize(
    "naics,zip,expected_backoff_naics,expected_backoff_zip",
    [
        (
            ["100", "200", "300"],
            ["111", "222", "333"],
            [False, False, True],
            [False, False, True],
        ),
        (["999"], ["888"], [True], [True]),
        (["100"], ["888"], [False], [True]),
        (["999"], ["111"], [True], [False]),
        ([], [], [], []),
    ],
)
@log_test_performance
def test_predict_proba_backoff_flags_are_correct_for_known_and_unknown_codes(
    naics,
    zip,
    expected_backoff_naics,
    expected_backoff_zip,
    naics_zip_level_maps_fixture,
    effects_fixture,
    test_run_id,
):
    df = pd.DataFrame({"naics": naics, "zip": zip})
    naics_level_maps, zip_level_maps = naics_zip_level_maps_fixture
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects_fixture,
        prefix_fill="0",
    )
    actual = out["backoff_naics_0"].tolist()
    assert len(actual) == len(expected_backoff_naics), (
        f"Length mismatch: got {len(actual)}, expected {len(expected_backoff_naics)}"
    )
    assert actual == expected_backoff_naics


@pytest.mark.parametrize(
    "naics,zip,expected_backoff_naics,expected_backoff_zip",
    [
        (
            ["100", "200", "300"],
            ["111", "222", "333"],
            [False, False, True],
            [False, False, True],
        ),
        (["999"], ["888"], [True], [True]),
        (["100"], ["888"], [False], [True]),
        (["999"], ["111"], [True], [False]),
        ([], [], [], []),
    ],
)
@log_test_performance
def test_predict_proba_backoff_zip_flags_are_correct_for_known_and_unknown_codes(
    naics,
    zip,
    expected_backoff_naics,
    expected_backoff_zip,
    naics_zip_level_maps_fixture,
    effects_fixture,
    test_run_id,
):
    df = pd.DataFrame({"naics": naics, "zip": zip})
    naics_level_maps, zip_level_maps = naics_zip_level_maps_fixture
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects_fixture,
        prefix_fill="0",
    )
    actual = out["backoff_zip_0"].tolist()
    assert len(actual) == len(expected_backoff_zip), (
        f"Length mismatch: got {len(actual)}, expected {len(expected_backoff_zip)}"
    )
    assert actual == expected_backoff_zip


@pytest.mark.parametrize(
    "naics,zip,expected_eta",
    [
        (["999"], ["888"], [0.0]),  # Both codes unseen, eta should be beta0
        (["100"], ["111"], [0.1 + 0.2 + 0.05 + 0.02 + 0.01 + 0.03]),  # Both codes seen
    ],
)
def test_predict_proba_eta_is_correct_for_known_and_unseen_codes(
    naics, zip, expected_eta, naics_zip_level_maps_fixture, effects_fixture
):
    df = pd.DataFrame({"naics": naics, "zip": zip})
    naics_level_maps, zip_level_maps = naics_zip_level_maps_fixture
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects_fixture,
        prefix_fill="0",
    )
    # Assert eta matches expected for each row
    assert np.allclose(out["eta"].tolist(), expected_eta)


def test_predict_proba_eta_and_p_are_finite_for_all_codes(
    naics_zip_level_maps_fixture, effects_fixture
):
    df = pd.DataFrame(
        {
            "naics": ["100", "200", "999"],
            "zip": ["111", "222", "888"],
        }
    )
    naics_level_maps, zip_level_maps = naics_zip_level_maps_fixture
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects_fixture,
        prefix_fill="0",
    )
    assert np.isfinite(out["eta"]).all() and np.isfinite(out["p"]).all()


def test_predict_proba_eta_is_beta0_for_unseen_codes(
    naics_zip_level_maps_fixture, effects_fixture
):
    df = pd.DataFrame(
        {
            "naics": ["100", "200", "999"],
            "zip": ["111", "222", "888"],
        }
    )
    naics_level_maps, zip_level_maps = naics_zip_level_maps_fixture
    out = predict_proba(
        df_new=df,
        naics_col="naics",
        zip_col="zip",
        naics_cut_points=[1, 2, 3],
        zip_cut_points=[1, 2, 3],
        naics_level_maps=naics_level_maps,
        zip_level_maps=zip_level_maps,
        effects=effects_fixture,
        prefix_fill="0",
    )
    # Only one assertion: eta for unseen code is beta0
    assert np.isclose(out["eta"][2], effects_fixture["beta0"])


def test_predict_proba_partial_backoff():
    def make_maps():
        return [
            {"1": 0, "2": 1},
            {},
            {},
        ], [
            {"1": 0, "2": 1},
            {},
            {},
        ]

    def expected_eta_for_row(i, df, naics_level_maps, zip_level_maps, effects):
        naics_idx = naics_level_maps[0][df["naics"][i][0]]
        zip_idx = zip_level_maps[0][df["zip"][i][0]]
        eta = effects["beta0"]
        eta += float(effects["naics_base"].loc[naics_idx])
        eta += float(effects["zip_base"].loc[zip_idx])
        for delta_tbl in effects["naics_deltas"]:
            eta += float(delta_tbl.loc[naics_idx])
        for delta_tbl in effects["zip_deltas"]:
            eta += float(delta_tbl.loc[zip_idx])
        return eta

    # Initial test: codes only known at parent level
    df = pd.DataFrame({"naics": ["301", "401"], "zip": ["317", "427"]})
    naics_level_maps, zip_level_maps = make_maps()
    from hierarchical_naics_model.make_backoff_resolver import make_backoff_resolver

    naics_resolver = make_backoff_resolver(
        cut_points=[1, 2, 3], level_maps=naics_level_maps, prefix_fill="0"
    )
    zip_resolver = make_backoff_resolver(
        cut_points=[1, 2, 3], level_maps=zip_level_maps, prefix_fill="0"
    )
    for i in range(2):
        print(f"naics: {df['naics'][i]}, resolved: {naics_resolver(df['naics'][i])}")
        print(f"zip: {df['zip'][i]}, resolved: {zip_resolver(df['zip'][i])}")

    # Main test: codes only known at parent level
    df = pd.DataFrame({"naics": ["103", "203"], "zip": ["117", "227"]})
    naics_level_maps, zip_level_maps = make_maps()
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
        print(
            f"Row {i} backoff_naics: {[out[f'backoff_naics_{j}'][i] for j in range(3)]}"
        )
        print(f"Row {i} backoff_zip: {[out[f'backoff_zip_{j}'][i] for j in range(3)]}")
        for j in range(3):
            assert not out[f"backoff_naics_{j}"][i]
            assert not out[f"backoff_zip_{j}"][i]
        # Eta should include all fallback delta contributions
        expected_eta = expected_eta_for_row(
            i, df, naics_level_maps, zip_level_maps, effects
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
