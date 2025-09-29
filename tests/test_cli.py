from __future__ import annotations

from pathlib import Path

import polars as pl

import pytest

from hierarchical_naics_model.cli import main as cli_main


def test_cli_smoke_and_artifacts(tmp_path: Path):
    # Create a tiny dataset as Parquet
    df = pl.DataFrame(
        {
            "is_written": [0, 1, 0, 1, 0, 1],
            "naics": ["511110", "511120", "52", "52413", "51213", "511110"],
            "zip": ["30309", "94103", "02139", "10001", "73301", "30309"],
        }
    )
    pq = tmp_path / "toy.parquet"
    df.write_parquet(str(pq))

    # Run the CLI with subset and low draws for speed
    outdir = tmp_path / "artifacts"
    rc = cli_main(
        [
            "--input",
            str(pq),
            "--subset",
            "4",
            "--draws",
            "50",
            "--tune",
            "50",
            "--chains",
            "2",
            "--cores",
            "1",
            "--random-seed",
            "123",
            "--outdir",
            str(outdir),
        ]
    )
    assert rc == 0
    # Ensure some artifact was written
    assert (outdir / "summary_beta0.csv").exists() or (outdir / "idata.nc").exists()


def test_cli_invalid_cut_points(tmp_path: Path):
    df = pl.DataFrame(
        {
            "is_written": [0, 1, 0, 1],
            "naics": ["511110", "511120", "52", "52413"],
            "zip": ["30309", "94103", "02139", "10001"],
        }
    )
    pq = tmp_path / "toy.parquet"
    df.write_parquet(str(pq))

    with pytest.raises(Exception):  # argparse.ArgumentTypeError from _parse_cuts
        cli_main(["--input", str(pq), "--subset", "2", "--naics-cuts", "a,b"])


def test_cli_missing_columns(tmp_path: Path):
    # Omit ZIP column
    df = pl.DataFrame(
        {
            "is_written": [0, 1],
            "naics": ["52", "511110"],
        }
    )
    pq = tmp_path / "toy_missing.parquet"
    df.write_parquet(str(pq))

    with pytest.raises(SystemExit):
        cli_main(["--input", str(pq)])


def test_cli_input_not_found(tmp_path: Path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(SystemExit):
        cli_main(["--input", str(missing)])


def test_cli_validations(tmp_path: Path):
    # Non-binary target
    df = pl.DataFrame(
        {
            "is_written": [0, 2, 1],
            "naics": ["52", "511110", "52413"],
            "zip": ["02139", "30309", "94103"],
        }
    )
    pq = tmp_path / "bad_target.parquet"
    df.write_parquet(str(pq))
    with pytest.raises(SystemExit):
        cli_main(
            ["--input", str(pq), "--subset", "3"]
        )  # validation should fire before sampling

    # Invalid draws parameter
    df2 = pl.DataFrame(
        {
            "is_written": [0, 1, 0],
            "naics": ["52", "511110", "52413"],
            "zip": ["02139", "30309", "94103"],
        }
    )
    pq2 = tmp_path / "ok.parquet"
    df2.write_parquet(str(pq2))
    with pytest.raises(SystemExit):
        cli_main(
            ["--input", str(pq2), "--subset", "3", "--draws", "0"]
        )  # invalid draws

        # Invalid subset
        with pytest.raises(SystemExit):
            cli_main(["--input", str(pq2), "--subset", "0"])  # invalid subset


def test_cli_summary_and_save_exceptions(tmp_path: Path, monkeypatch):
    # Create minimal valid parquet
    df = pl.DataFrame(
        {
            "is_written": [0, 1, 0],
            "naics": ["52", "511110", "52413"],
            "zip": ["02139", "30309", "94103"],
        }
    )
    pq = tmp_path / "tiny.parquet"
    df.write_parquet(str(pq))

    # Monkeypatch pm.sample to avoid doing actual sampling and return fake idata
    from hierarchical_naics_model import cli as cli_mod

    class FakeIdata:
        def to_netcdf(self, *_args, **_kwargs):
            raise RuntimeError("fail save")

    def fake_sample(*_a, **_kw):
        return FakeIdata()

    def fake_summary(*_a, **_kw):
        raise RuntimeError("fail summary")

    monkeypatch.setattr(cli_mod.pm, "sample", fake_sample)
    # CLI uses ArviZ summary now; keep this for backwards compat paths
    monkeypatch.setattr(cli_mod.pm, "summary", fake_summary)

    outdir = tmp_path / "artifacts_exc"
    rc = cli_main(["--input", str(pq), "--subset", "2", "--outdir", str(outdir)])
    assert rc == 0
