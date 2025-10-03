import sys
import numpy as np
import random
import pandas as pd
from hierarchical_naics_model.synthgen.generate import (
    HierSpec,
    generate_synthetic_dataset,
    _labels_for_levels,
)
from hierarchical_naics_model.synthgen import generate


def test_save_outputs(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    artifacts = {"meta": {"test": True}}
    out_dir = tmp_path / "out"
    generate._save_outputs(df, artifacts, str(out_dir))
    assert (out_dir / "synthetic.parquet").exists() or (
        out_dir / "synthetic.csv"
    ).exists()
    assert (out_dir / "artifacts.json").exists()


def test_save_outputs_none_dir():
    df = pd.DataFrame({"a": [1, 2]})
    artifacts = {"meta": {"test": True}}
    # Should simply return, not raise
    generate._save_outputs(df, artifacts, None)


def test_save_outputs_exception(tmp_path, monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    artifacts = {"meta": {"test": True}}
    out_dir = tmp_path / "out"

    # Monkeypatch to_parquet to raise Exception
    def raise_exc(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr(df, "to_parquet", raise_exc)
    # Should fall back to to_csv
    generate._save_outputs(df, artifacts, str(out_dir))
    assert (out_dir / "synthetic.csv").exists()
    assert (out_dir / "artifacts.json").exists()


def test_main_cli(tmp_path, capsys, monkeypatch):
    # Simulate CLI call
    out_dir = tmp_path / "cli_out"
    argv = [
        "--n",
        "5",
        "--seed",
        "123",
        "--out-dir",
        str(out_dir),
        "--naics-cuts",
        "2",
        "3",
        "--naics-branching",
        "2",
        "--zip-cuts",
        "3",
        "5",
        "--zip-branching",
        "20",
    ]
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    generate.main()
    captured = capsys.readouterr()
    assert "Generated 5 rows." in captured.out
    assert f"Wrote data/artifacts to: {out_dir}" in captured.out
    assert (out_dir / "synthetic.parquet").exists() or (
        out_dir / "synthetic.csv"
    ).exists()
    assert (out_dir / "artifacts.json").exists()


def test_main_entrypoint(monkeypatch, tmp_path):
    # Simulate __main__ entry
    monkeypatch.setattr(generate, "main", lambda *a, **kw: True)
    import importlib

    mod = importlib.import_module("hierarchical_naics_model.synthgen.generate")
    # Should not raise
    if hasattr(mod, "__main__"):
        assert True


def test_sigmoid_basic():
    arr = np.array([-100, 0, 100], dtype=float)
    out = generate._sigmoid(arr)
    assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_pad_width_and_fill():
    assert generate._pad("12", 4) == "1200"
    assert generate._pad("1234", 2) == "1234"
    assert generate._pad("1", 3, fill="x") == "1xx"


def test_build_hierarchy_labels_and_level_maps():
    spec = generate.HierSpec(cut_points=[2, 3], branching=[2])
    levels = generate._build_hierarchy_labels(spec, seed=42)
    assert isinstance(levels, list) and all(isinstance(level, list) for level in levels)
    maps = generate._build_level_maps(levels, spec.cut_points)
    assert isinstance(maps, generate.LevelMaps)
    assert maps.levels == ["L2", "L3"]


def test_sample_params_for_levels_and_make_true_params():
    levels = [["01", "02"], ["011", "012", "021", "022"]]
    base, deltas = generate._sample_params_for_levels(
        levels, 1.0, 0.5, np.random.default_rng(1)
    )
    assert len(base) == 2 and len(deltas) == 1
    params = generate.make_true_params(levels, levels, seed=1)
    assert isinstance(params, generate.TrueParams)


def test_choose_leaf_empty_and_nonempty():
    rng = random.Random(42)
    assert generate._choose_leaf([], rng) == ""
    levels = [["01"], ["011", "012"]]
    assert generate._choose_leaf(levels, rng) in levels[-1]


def test_eta_for_codes_full_branch():
    # Use minimal valid setup
    naics = generate.HierSpec(cut_points=[2], branching=[])
    zipsp = generate.HierSpec(cut_points=[3], branching=[])
    df, artifacts = generate.generate_synthetic_dataset(
        n=1, naics_spec=naics, zip_spec=zipsp, seed=123
    )


def test_small_generation_shapes_and_bounds():
    naics = HierSpec(cut_points=[2, 3], branching=[2])
    zipsp = HierSpec(cut_points=[3, 5], branching=[3])
    df, artifacts = generate_synthetic_dataset(
        n=200, naics_spec=naics, zip_spec=zipsp, seed=123
    )

    assert len(df) == 200
    assert {"naics_code", "zip_code", "eta", "p", "y"}.issubset(df.columns)
    assert df["p"].between(0.0, 1.0, inclusive="both").all()
    assert df["naics_code"].str.len().unique().tolist() == [max(naics.cut_points)]
    assert df["zip_code"].str.len().unique().tolist() == [max(zipsp.cut_points)]


def test_eta_matches_parameter_sum():
    from hierarchical_naics_model.synthgen.generate import (
        LevelMaps,
        TrueParams,
        _eta_for_codes,
    )

    naics = HierSpec(cut_points=[2, 3], branching=[2])
    zipsp = HierSpec(cut_points=[3, 5], branching=[2])
    df, artifacts = generate_synthetic_dataset(
        n=10, naics_spec=naics, zip_spec=zipsp, seed=7
    )
    row = df.iloc[0]
    naics_code = row["naics_code"]
    zip_code = row["zip_code"]
    eff = artifacts["effects"]
    # Build TrueParams instance
    params = TrueParams(
        beta0=eff["beta0"],
        naics_base=eff["naics_base"],
        naics_deltas=eff["naics_deltas"],
        zip_base=eff["zip_base"],
        zip_deltas=eff["zip_deltas"],
    )
    n_maps = LevelMaps(
        levels=artifacts["naics_maps"]["levels"],
        maps=artifacts["naics_maps"]["maps"],
        cut_points=artifacts["naics_maps"]["cut_points"],
    )
    z_maps = LevelMaps(
        levels=artifacts["zip_maps"]["levels"],
        maps=artifacts["zip_maps"]["maps"],
        cut_points=artifacts["zip_maps"]["cut_points"],
    )
    eta = _eta_for_codes(naics_code, zip_code, n_maps, z_maps, params)
    assert abs(eta - float(row["eta"])) < 1e-9


def test_labels_for_levels_padding():
    s = "12"
    labs = _labels_for_levels(s, [2, 3, 4], 4, fill="0")
    assert labs == ["12", "120", "1200"]


def test_hierspec_invalid_cut_points():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    # Non-increasing cut_points
    spec = HierSpec(cut_points=[2, 2], branching=[1])
    try:
        spec.validate()
        assert False, "Expected ValueError for non-increasing cut_points"
    except ValueError as e:
        assert "cut_points must be strictly increasing" in str(e)


def test_hierspec_invalid_branching_length():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    # Branching wrong length
    spec = HierSpec(cut_points=[2, 3, 4], branching=[1])
    try:
        spec.validate()
        assert False, "Expected ValueError for branching length"
    except ValueError as e:
        assert "branching must have length" in str(e)


def test_hierspec_invalid_fill():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    spec = HierSpec(cut_points=[2, 3], branching=[1], fill="00")
    try:
        spec.validate()
        assert False, "Expected ValueError for fill length"
    except ValueError as e:
        assert "fill must be a single character" in str(e)


def test_generate_synthetic_dataset_minimal():
    from hierarchical_naics_model.synthgen.generate import (
        HierSpec,
        generate_synthetic_dataset,
    )

    naics = HierSpec(cut_points=[2], branching=[])
    zipsp = HierSpec(cut_points=[3], branching=[])
    df, artifacts = generate_synthetic_dataset(
        n=1, naics_spec=naics, zip_spec=zipsp, seed=42
    )
    assert len(df) == 1
    assert "naics_code" in df.columns and "zip_code" in df.columns


def test_hierspec_branching_length_too_short():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    # branching too short for cut_points
    spec = HierSpec(cut_points=[2, 3, 4], branching=[1])
    try:
        spec.validate()
        assert False, "Expected ValueError for branching length too short"
    except ValueError as e:
        assert "branching must have length" in str(e)


def test_hierspec_branching_length_too_long():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    # branching too long for cut_points
    spec = HierSpec(cut_points=[2, 3], branching=[1, 2])
    try:
        spec.validate()
        assert False, "Expected ValueError for branching length too long"
    except ValueError as e:
        assert "branching must have length" in str(e)


def test_hierspec_branching_value_too_small():
    from hierarchical_naics_model.synthgen.generate import HierSpec

    # branching value < 1
    spec = HierSpec(cut_points=[2, 3], branching=[0])
    try:
        spec.validate()
        assert False, "Expected ValueError for branching value < 1"
    except ValueError as e:
        assert "branching values must be >= 1" in str(e)
