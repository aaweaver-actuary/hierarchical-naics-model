import os
import tempfile
import pandas as pd
import pytest
from hierarchical_naics_model.calibration_cli import main as calibration_main


def make_dummy_df(path):
    df = pd.DataFrame(
        {
            "is_written": [0, 1, 1, 0, 1],
            "p_hat": [0.1, 0.8, 0.7, 0.2, 0.9],
            "date": [
                "2025-09-01",
                "2025-09-02",
                "2025-09-03",
                "2025-09-04",
                "2025-09-05",
            ],
        }
    )
    df.to_parquet(path)
    return path


def test_calibration_cli_smoke(tmp_path):
    pq = tmp_path / "dummy.parquet"
    make_dummy_df(pq)
    outdir = tmp_path / "out"
    args = [
        "--input",
        str(pq),
        "--outdir",
        str(outdir),
        "--y-col",
        "is_written",
        "--p-col",
        "p_hat",
        "--bins",
        "3",
        "--ks",
        "1,2,3",
    ]
    calibration_main(args)
    # Check output files
    assert (outdir / "metrics.json").exists()
    assert (outdir / "reliability.csv").exists()
    assert (outdir / "ranking.csv").exists()
    assert (outdir / "sorted_scores.csv").exists()
    assert (outdir / "reliability_curve.png").exists()
    assert (outdir / "lift_and_gain.png").exists()
    assert (outdir / "precision_at_k.png").exists()
    assert (outdir / "brier_logloss.png").exists()


def test_calibration_cli_plotly_html(tmp_path):
    pq = tmp_path / "dummy.parquet"
    make_dummy_df(pq)
    outdir = tmp_path / "out_html"
    args = [
        "--input",
        str(pq),
        "--outdir",
        str(outdir),
        "--plotly-html",
    ]
    calibration_main(args)
    assert (outdir / "reliability_curve.html").exists()
    assert (outdir / "lift_and_gain.html").exists()
    assert (outdir / "precision_at_k.html").exists()
    assert (outdir / "brier_logloss.html").exists()


def test_calibration_cli_temporal_split(tmp_path):
    pq = tmp_path / "dummy.parquet"
    make_dummy_df(pq)
    outdir = tmp_path / "out_split"
    args = [
        "--input",
        str(pq),
        "--outdir",
        str(outdir),
        "--date-col",
        "date",
        "--cutoff-date",
        "2025-09-03",
    ]
    calibration_main(args)
    assert (outdir / "metrics.json").exists()
    # Should run without error and produce metrics for score > cutoff


def test_calibration_cli_missing_file(tmp_path):
    outdir = tmp_path / "out_missing"
    args = [
        "--input",
        str(tmp_path / "notfound.parquet"),
        "--outdir",
        str(outdir),
    ]
    with pytest.raises(Exception):
        calibration_main(args)


def test_calibration_cli_invalid_bins(tmp_path):
    pq = tmp_path / "dummy.parquet"
    make_dummy_df(pq)
    outdir = tmp_path / "out_invalid_bins"
    args = [
        "--input",
        str(pq),
        "--outdir",
        str(outdir),
        "--bins",
        "0",
    ]
    with pytest.raises(Exception):
        calibration_main(args)
