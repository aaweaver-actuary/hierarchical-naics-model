from __future__ import annotations

import pickle

import pandas as pd
import pytest

from hierarchical_naics_model.io.artifacts import (
    Artifacts,
    LevelMaps,
    save_artifacts,
    load_artifacts,
)


def _sample_artifacts() -> Artifacts:
    effects = {
        "beta0": 0.0,
        "naics_base": pd.Series([0.0]),
        "naics_deltas": [],
        "zip_base": pd.Series([0.0]),
        "zip_deltas": [],
    }
    naics_maps: LevelMaps = {
        "levels": ["L2"],
        "maps": [{"52": 0}],
        "cut_points": [2],
    }
    zip_maps: LevelMaps = {
        "levels": ["L2"],
        "maps": [{"12": 0}],
        "cut_points": [2],
    }
    return {
        "naics_maps": naics_maps,
        "zip_maps": zip_maps,
        "effects": effects,
        "meta": {"target_col": "is_written"},
    }


def test_save_and_load_artifacts_file(tmp_path):
    art = _sample_artifacts()
    path = tmp_path / "artifacts.pkl"
    save_artifacts(art, path)

    loaded = load_artifacts(path)
    assert loaded["effects"]["beta0"] == art["effects"]["beta0"]
    assert loaded["meta"]["target_col"] == "is_written"


def test_save_and_load_artifacts_directory(tmp_path):
    art = _sample_artifacts()
    directory = tmp_path / "bundle"
    save_artifacts(art, directory)

    loaded = load_artifacts(directory)
    assert loaded["naics_maps"]["cut_points"] == [2]


def test_load_artifacts_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path / "missing")


def test_load_artifacts_invalid_content(tmp_path):
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_bytes(pickle.dumps([]))

    with pytest.raises(ValueError):
        load_artifacts(bad_path)
