from hierarchical_naics_model.scoring.explain import explain_row


def test_explain_row_basic_known():
    effects = {
        "beta0": 0.1,
        "naics_base": [0.5, -0.2],
        "naics_deltas": [[0.05, 0.07]],
        "zip_base": [0.3],
        "zip_deltas": [[-0.1, 0.2]],
    }
    naics_maps = [
        {"11": 0, "22": 1},  # L2
        {"111": 0, "221": 1},  # L3
    ]
    zip_maps = [
        {"451": 0},  # L3
        {"45101": 0, "45102": 1},  # L5
    ]

    out = explain_row(
        naics_code="111",
        zip_code="45102",
        naics_cut_points=[2, 3],
        zip_cut_points=[3, 5],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
    )
    # eta = 0.1 + 0.5 + 0.05 + 0.3 + 0.2 = 1.15
    assert abs(out["eta"] - 1.15) < 1e-9
    assert out["naics_path"] == [("11", 0.5), ("111", 0.05)]
    assert out["zip_path"] == [("451", 0.3), ("45102", 0.2)]
    assert 0.75 < out["p"] < 0.77  # logistic(1.15)


def test_explain_row_unknown_levels():
    effects = {
        "beta0": 0.0,
        "naics_base": [1.0],
        "naics_deltas": [[0.5]],
        "zip_base": [0.0],
        "zip_deltas": [[]],
    }
    naics_maps = [
        {"12": 0},  # known base
        {"123": 0},  # known delta
    ]
    zip_maps = [
        {"999": 0},
        {},
    ]

    out = explain_row(
        naics_code="129",  # base known, deeper unknown
        zip_code="00000",  # unknown at all levels
        naics_cut_points=[2, 3],
        zip_cut_points=[3, 5],
        naics_level_maps=naics_maps,
        zip_level_maps=zip_maps,
        effects=effects,
    )
    assert out["naics_path"][0] == ("12", 1.0)
    assert out["naics_path"][1][1] == 0.0
    assert out["zip_path"][0][1] == 0.0
    assert out["eta"] == 1.0
