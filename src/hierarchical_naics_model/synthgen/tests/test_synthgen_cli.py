import sys
import polars as pl
import tempfile
from pathlib import Path
import pytest
from hierarchical_naics_model.synthgen import cli


def test_cli_main_runs_and_writes(monkeypatch):
    # Patch sys.argv
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_synth.parquet"
        monkeypatch.setattr(
            sys, "argv", ["synthgen", "--n", "10", "--out", str(out_path)]
        )
        cli.main()
        assert out_path.exists()
        df = pl.read_parquet(out_path)
        assert df.shape[0] == 10
        assert set(["naics_code", "zip_code", "eta", "p", "y"]).issubset(df.columns)


@pytest.mark.parametrize(
    "bad_args",
    [
        ["synthgen", "--n", "-1"],  # negative n
        ["synthgen", "--naics-cut-points", "2", "2"],  # non-increasing cut_points
        [
            "synthgen",
            "--naics-cut-points",
            "2",
            "3",
            "4",
            "--naics-branching",
            "1",
        ],  # wrong branching length (cut_points has 3, branching must have 2)
        [
            "synthgen",
            "--zip-cut-points",
            "3",
            "--zip-branching",
            "1",
            "--out",
            "bad.parquet",
        ],  # minimal branching
    ],
)
def test_cli_main_invalid_args(monkeypatch, bad_args):
    monkeypatch.setattr(sys, "argv", bad_args)
    with pytest.raises(SystemExit):
        cli.main()
