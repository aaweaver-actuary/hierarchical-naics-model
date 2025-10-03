from hierarchical_naics_model.synthgen.generate import (
    HierSpec,
    generate_synthetic_dataset,
    _labels_for_levels,
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
