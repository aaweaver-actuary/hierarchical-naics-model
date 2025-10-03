import pytest
from functools import reduce
from hierarchical_naics_model.io.artifacts_portable import (
    save_artifacts_portable,
    load_artifacts_portable,
)


@pytest.fixture
def sample_artifacts():
    return {
        "naics_maps": {"levels": ["L2"], "maps": [{"11": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["Z3"], "maps": [{"451": 0}], "cut_points": [3]},
        "effects": {"beta0": [0.1], "naics_base": [0.5], "zip_base": [0.3]},
        "meta": {"schema_version": 1},
    }


@pytest.fixture
def saved_artifacts(tmp_path, sample_artifacts):
    out_dir = save_artifacts_portable(tmp_path / "art", sample_artifacts)
    return out_dir


@pytest.fixture
def loaded_artifacts(saved_artifacts):
    return load_artifacts_portable(saved_artifacts)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("meta.schema_version", 1),
        ("naics_maps.cut_points", [2]),
        ("effects.naics_base", [0.5]),
        ("effects.beta0", [0.1]),
        ("zip_maps.levels", ["Z3"]),
        ("zip_maps.cut_points", [3]),
    ],
)
def test_json_roundtrip_single_field_value_is_preserved(
    loaded_artifacts, key, expected
):
    keys = key.split(".")
    try:
        value = reduce(lambda d, k: d[k], keys, loaded_artifacts)
    except Exception as e:
        print(f"Failed to access key path '{key}' in loaded artifacts: {e}")
        raise
    assert value == expected, f"Key '{key}' expected value {expected}, got {value}"


@pytest.mark.parametrize(
    "artifacts,key,expected",
    [
        (
            {
                "naics_maps": {"levels": [], "maps": [], "cut_points": []},
                "zip_maps": {"levels": [], "maps": [], "cut_points": []},
                "effects": {"beta0": [], "naics_base": [], "zip_base": []},
                "meta": {"schema_version": 0},
            },
            "meta.schema_version",
            0,
        ),
        (
            {
                "naics_maps": {
                    "levels": ["L2"],
                    "maps": [{"11": 0}],
                    "cut_points": [2],
                },
                "zip_maps": {"levels": ["Z3"], "maps": [{"451": 0}], "cut_points": [3]},
                "effects": {"beta0": [0.1], "naics_base": [0.5], "zip_base": [0.3]},
                "meta": {"schema_version": 1},
            },
            "effects.beta0",
            [0.1],
        ),
        (
            {
                "naics_maps": {"levels": [], "maps": [], "cut_points": []},
                "zip_maps": {"levels": [], "maps": [], "cut_points": []},
                "effects": {"beta0": [], "naics_base": [], "zip_base": []},
                "meta": {"schema_version": 0},
            },
            "effects.naics_base",
            [],
        ),
    ],
)
def test_json_roundtrip_edge_case_field_value_is_preserved(
    tmp_path, artifacts, key, expected
):
    out_dir = save_artifacts_portable(tmp_path / "art_edge", artifacts)
    loaded = load_artifacts_portable(out_dir)
    keys = key.split(".")
    try:
        value = reduce(lambda d, k: d[k], keys, loaded)
    except Exception as e:
        print(f"Failed to access key path '{key}' in loaded edge-case artifacts: {e}")
        raise
    assert value == expected, (
        f"Edge case key '{key}' expected value {expected}, got {value}"
    )


@pytest.fixture
def artifacts_for_parquet():
    return {
        "naics_maps": {"levels": ["L2"], "maps": [{"11": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["Z3"], "maps": [{"451": 0}], "cut_points": [3]},
        # All arrays must be the same length for Parquet
        "effects": {
            "naics_base": [0.5, -0.3],
            "zip_base": [0.2, 0.0],
            "beta0": [0.1, 0.0],
        },
        "meta": {"schema_version": 2},
    }


@pytest.fixture
def saved_parquet_artifacts(tmp_path, artifacts_for_parquet):
    out_dir = save_artifacts_portable(tmp_path / "art2", artifacts_for_parquet)
    return out_dir


@pytest.fixture
def loaded_parquet_artifacts(saved_parquet_artifacts):
    return load_artifacts_portable(saved_parquet_artifacts)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("effects.zip_base", [0.2, 0.0]),
        ("effects.naics_base", [0.5, -0.3]),
        ("meta.schema_version", 2),
        ("naics_maps.levels", ["L2"]),
        ("zip_maps.cut_points", [3]),
    ],
)
def test_parquet_roundtrip_single_field_value_is_preserved(
    loaded_parquet_artifacts, key, expected
):
    keys = key.split(".")
    try:
        value = reduce(lambda d, k: d[k], keys, loaded_parquet_artifacts)
    except Exception as e:
        print(f"Failed to access key path '{key}' in loaded parquet artifacts: {e}")
        raise
    assert value == expected, (
        f"Parquet key '{key}' expected value {expected}, got {value}"
    )


def test_parquet_roundtrip_effects_key_is_present(loaded_parquet_artifacts):
    assert "effects" in loaded_parquet_artifacts, (
        f"'effects' key missing in loaded parquet artifacts: {list(loaded_parquet_artifacts.keys())}"
    )


def test_parquet_roundtrip_zip_base_value_is_correct(loaded_parquet_artifacts):
    zip_base = loaded_parquet_artifacts["effects"]["zip_base"][0]
    assert abs(zip_base - 0.2) < 1e-9, (
        f"zip_base value mismatch: expected 0.2, got {zip_base}"
    )


@pytest.fixture
def loaded_parquet_artifacts2(saved_parquet_artifacts):
    return load_artifacts_portable(saved_parquet_artifacts)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("effects.zip_base", [0.2, 0.0]),
        ("effects.naics_base", [0.5, -0.3]),
        ("meta.schema_version", 2),
        ("naics_maps.levels", ["L2"]),
        ("zip_maps.cut_points", [3]),
    ],
)
def test_parquet_roundtrip_field_values_are_preserved(
    loaded_parquet_artifacts2, key, expected
):
    keys = key.split(".")
    value = loaded_parquet_artifacts2
    # Use functools.reduce for performance and clarity
    try:
        value = reduce(lambda d, k: d[k], keys, loaded_parquet_artifacts2)
    except Exception as e:
        print(f"Failed to access key path '{key}' in loaded artifacts: {e}")
        raise
    assert value == expected, f"Expected {expected} for key '{key}', got {value}"


def test_parquet_roundtrip_effects_key_present(loaded_parquet_artifacts):
    assert "effects" in loaded_parquet_artifacts, (
        f"'effects' key missing in loaded artifacts: {loaded_parquet_artifacts.keys()}"
    )


def test_parquet_roundtrip_zip_base_value_is_correct2(loaded_parquet_artifacts):
    zip_base = loaded_parquet_artifacts["effects"]["zip_base"][0]
    assert abs(zip_base - 0.2) < 1e-9, (
        f"zip_base value mismatch: expected 0.2, got {zip_base}"
    )


def test_save_artifacts_portable_ragged_effects_error(tmp_path):
    artifacts = {
        "naics_maps": {"levels": ["L2"], "maps": [{"11": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["Z3"], "maps": [{"451": 0}], "cut_points": [3]},
        "effects": {"naics_base": [0.5, -0.3], "zip_base": [0.2], "beta0": [0.1]},
        "meta": {"schema_version": 2},
    }
    import pytest

    with pytest.raises(Exception) as excinfo:
        save_artifacts_portable(tmp_path / "art3", artifacts)
    assert "expected length" in str(excinfo.value) or "lengths" in str(excinfo.value)
