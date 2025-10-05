from __future__ import annotations

import json
import numpy as np
import polars as pl
import pytest
import xarray as xr

import hierarchical_naics_model.cli as cli_root
from hierarchical_naics_model.cli import fit, report, score
from hierarchical_naics_model.io.artifacts import load_artifacts, save_artifacts
from hierarchical_naics_model.io.datasets import save_parquet
from hierarchical_naics_model.modeling import pymc_nested as modeling_pymc_nested


class DummyIdata:
    def __init__(self, posterior: xr.Dataset) -> None:
        self.posterior = posterior

    def groups(self):
        return ["posterior"]


def _fake_indices(n_rows: int, cut_points, prefix_fill: str):
    levels = [f"L{c}" for c in cut_points]
    code_levels = np.zeros((n_rows, len(cut_points)), dtype=int)
    unique_per_level = [np.array(["placeholder"], dtype=object) for _ in cut_points]
    maps = [{"placeholder"[:c].ljust(c, prefix_fill): 0} for c in cut_points]
    parent_index_per_level = [None] + [np.zeros(1, dtype=int) for _ in cut_points[1:]]
    return {
        "levels": levels,
        "code_levels": code_levels,
        "unique_per_level": unique_per_level,
        "maps": maps,
        "group_counts": [1] * len(cut_points),
        "parent_index_per_level": parent_index_per_level,
        "max_len": max(cut_points) if cut_points else 0,
        "cut_points": list(cut_points),
    }


@pytest.fixture
def posterior_stub() -> DummyIdata:
    posterior = xr.Dataset(
        data_vars={
            "beta0": (("chain", "draw"), np.array([[0.0]])),
            "naics_base": (("chain", "draw", "naics_g0"), np.array([[[0.0]]])),
            "zip_base": (("chain", "draw", "zip_g0"), np.array([[[0.0]]])),
        },
        coords={
            "chain": [0],
            "draw": [0],
            "naics_g0": [0],
            "zip_g0": [0],
        },
    )
    return DummyIdata(posterior)


def test_fit_cli_minimal(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    class DummyStrategy:
        kwargs: dict[str, object] = {}

        def __init__(self, **kwargs):
            pass

        def build_model(self, **kwargs):
            return object()

        def sample_posterior(self, *args, **kwargs):
            DummyStrategy.kwargs = kwargs
            return posterior_stub

    monkeypatch.setattr(fit, "PymcNestedDeltaStrategy", DummyStrategy)

    artifacts_path = tmp_path / "artifacts.pkl"
    summary_path = tmp_path / "summary.json"

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(artifacts_path),
            "--summary",
            str(summary_path),
            "--naics-cuts",
            "2",
            "--zip-cuts",
            "2",
        ]
    )

    assert exit_code == 0
    assert artifacts_path.exists()
    loaded = load_artifacts(artifacts_path)
    assert loaded["meta"]["n_rows"] == 2
    assert summary_path.exists()
    assert DummyStrategy.kwargs["progressbar"] is True


def test_fit_cli_variational_strategy(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    captured: dict[str, object] = {}

    class DummyADVIStrategy:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def build_model(self, **kwargs):
            captured["build_model"] = kwargs
            return object()

        def sample_posterior(self, *args, **kwargs):
            captured["sample"] = kwargs
            return posterior_stub

    monkeypatch.setattr(modeling_pymc_nested, "PymcADVIStrategy", DummyADVIStrategy)

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(tmp_path / "artifacts.pkl"),
            "--inference-strategy",
            "pymc_advi",
            "--variational-steps",
            "123",
        ]
    )

    assert exit_code == 0
    assert captured["init"]["fit_steps"] == 123
    assert "build_model" in captured
    assert captured["sample"]["progressbar"] is True


def test_fit_cli_no_progressbar(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    class DummyStrategy:
        kwargs: dict[str, object] = {}

        def __init__(self, **kwargs):
            pass

        def build_model(self, **kwargs):
            return object()

        def sample_posterior(self, *args, **kwargs):
            DummyStrategy.kwargs = kwargs
            return posterior_stub

    monkeypatch.setattr(fit, "PymcNestedDeltaStrategy", DummyStrategy)

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(tmp_path / "artifacts.pkl"),
            "--no-progressbar",
        ]
    )

    assert exit_code == 0
    assert DummyStrategy.kwargs["progressbar"] is False


def test_fit_cli_dashboard_generation(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    class DummyStrategy:
        def __init__(self, **kwargs):
            pass

        def build_model(self, **kwargs):
            return object()

        def sample_posterior(self, *args, **kwargs):
            return posterior_stub

    monkeypatch.setattr(fit, "PymcNestedDeltaStrategy", DummyStrategy)

    captured: dict[str, object] = {}
    html_path = tmp_path / "dash" / "model_dashboard.html"

    def fake_dashboard(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "figure": object(),
            "fit_stats": {"n_train": kwargs["train_summary"]["n_train"]},
            "validation": {"loo": 0.0, "ece": 0.1, "brier": 0.2, "log_loss": 0.3},
            "test_metrics": {"brier": 0.2, "ece": 0.1, "log_loss": 0.3},
            "artifacts": {"html_path": html_path},
        }

    monkeypatch.setattr(fit, "build_dashboard", fake_dashboard)

    opened: dict[str, object] = {}

    def fake_open(path):
        opened["path"] = path
        return True

    monkeypatch.setattr(fit.webbrowser, "open", fake_open)

    dashboard_dir = tmp_path / "dash"

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(tmp_path / "artifacts.pkl"),
            "--dashboard",
            str(dashboard_dir),
        ]
    )

    assert exit_code == 0
    assert captured["kwargs"]["output_dir"] == dashboard_dir
    assert captured["kwargs"]["train_summary"]["n_train"] == 2
    assert opened["path"] == str(html_path)


def test_fit_cli_dashboard_suppressed_open(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    class DummyStrategy:
        def __init__(self, **kwargs):
            pass

        def build_model(self, **kwargs):
            return object()

        def sample_posterior(self, *args, **kwargs):
            return posterior_stub

    monkeypatch.setattr(fit, "PymcNestedDeltaStrategy", DummyStrategy)

    html_path = tmp_path / "dash" / "model_dashboard.html"

    def fake_dashboard(**kwargs):
        return {
            "figure": object(),
            "fit_stats": {"n_train": kwargs["train_summary"]["n_train"]},
            "validation": {"loo": 0.0, "ece": 0.1, "brier": 0.2, "log_loss": 0.3},
            "test_metrics": {"brier": 0.2, "ece": 0.1, "log_loss": 0.3},
            "artifacts": {"html_path": html_path},
        }

    monkeypatch.setattr(fit, "build_dashboard", fake_dashboard)

    monkeypatch.setattr(fit.webbrowser, "open", pytest.fail)

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(tmp_path / "artifacts.pkl"),
            "--dashboard",
            str(tmp_path / "dash"),
            "--no-open-dashboard",
        ]
    )

    assert exit_code == 0


def test_fit_cli_map_strategy(monkeypatch, tmp_path, posterior_stub):
    df = pl.LazyFrame(
        {"is_written": [1, 0], "NAICS": ["52", "52"], "ZIP": ["12", "12"]}
    )

    monkeypatch.setattr(fit, "load_parquet", lambda _: df)
    monkeypatch.setattr(
        fit,
        "build_hierarchical_indices",
        lambda codes, cut_points, prefix_fill: _fake_indices(
            len(codes), cut_points, prefix_fill
        ),
    )

    captured: dict[str, object] = {}

    class DummyMAPStrategy:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def build_model(self, **kwargs):
            captured["build_model"] = kwargs
            return object()

        def sample_posterior(self, *args, **kwargs):
            captured["sample"] = kwargs
            return posterior_stub

    monkeypatch.setattr(modeling_pymc_nested, "PymcMAPStrategy", DummyMAPStrategy)

    exit_code = fit.main(
        [
            "--train",
            "train.parquet",
            "--artifacts",
            str(tmp_path / "artifacts.pkl"),
            "--inference-strategy",
            "pymc_map",
            "--map-method",
            "Powell",
        ]
    )

    assert exit_code == 0
    assert captured["init"]["map_kwargs"]["method"] == "Powell"
    assert "build_model" in captured
    assert captured["sample"]["progressbar"] is True


def test_fit_cli_missing_required_column(monkeypatch, tmp_path):
    df = pl.LazyFrame({"NAICS": ["52"], "ZIP": ["12"]})
    monkeypatch.setattr(fit, "load_parquet", lambda _: df)

    with pytest.raises(ValueError, match="missing required columns"):
        fit.main(
            [
                "--train",
                "train.parquet",
                "--artifacts",
                str(tmp_path / "artifacts.pkl"),
            ]
        )


def test_cli_entrypoint_invokes_command(monkeypatch):
    calls = {}

    def stub(args):
        calls["args"] = args
        return 9

    with monkeypatch.context() as m:
        m.setitem(cli_root._COMMANDS, "fit", stub)
        exit_code = cli_root.main(["fit", "--dry-run"])

    assert exit_code == 9
    assert calls["args"] == ["--dry-run"]


def test_score_cli_roundtrip(tmp_path):
    artifacts_path = tmp_path / "bundle"
    artifacts = {
        "naics_maps": {"levels": ["L2"], "maps": [{"52": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["L2"], "maps": [{"12": 0}], "cut_points": [2]},
        "effects": {
            "beta0": 0.0,
            "naics_base": np.array([0.0], dtype=float),
            "naics_deltas": [],
            "zip_base": np.array([0.0], dtype=float),
            "zip_deltas": [],
        },
        "meta": {
            "target_col": "is_written",
            "naics_col": "NAICS",
            "zip_col": "ZIP",
            "prefix_fill": "0",
        },
    }
    save_artifacts(artifacts, artifacts_path)

    df = pl.LazyFrame({"NAICS": ["52"], "ZIP": ["12"], "is_written": [1]})
    input_path = tmp_path / "incoming.parquet"
    save_parquet(df, input_path)
    output_path = tmp_path / "scored.parquet"
    summary_path = tmp_path / "score_summary.json"

    exit_code = score.main(
        [
            "--data",
            str(input_path),
            "--artifacts",
            str(artifacts_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    scored_df = pl.read_parquet(output_path)
    assert {"eta", "p"}.issubset(scored_df.columns)
    assert summary_path.exists()


def test_score_cli_requires_metadata(tmp_path):
    artifacts_path = tmp_path / "bundle"
    artifacts = {
        "naics_maps": {"levels": ["L2"], "maps": [{"52": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["L2"], "maps": [{"12": 0}], "cut_points": [2]},
        "effects": {
            "beta0": 0.0,
            "naics_base": np.array([0.0], dtype=float),
            "naics_deltas": [],
            "zip_base": np.array([0.0], dtype=float),
            "zip_deltas": [],
        },
        "meta": {},
    }
    save_artifacts(artifacts, artifacts_path)

    df = pl.LazyFrame({"NAICS": ["52"], "ZIP": ["12"], "is_written": [1]})
    input_path = tmp_path / "incoming.parquet"
    save_parquet(df, input_path)

    with pytest.raises(ValueError, match="naics_col"):
        score.main(
            [
                "--data",
                str(input_path),
                "--artifacts",
                str(artifacts_path),
                "--output",
                str(tmp_path / "out.parquet"),
            ]
        )


def test_score_cli_with_overrides(tmp_path):
    artifacts_path = tmp_path / "bundle"
    artifacts = {
        "naics_maps": {"levels": ["L2"], "maps": [{"52": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["L2"], "maps": [{"12": 0}], "cut_points": [2]},
        "effects": {
            "beta0": 0.0,
            "naics_base": np.array([0.0], dtype=float),
            "naics_deltas": [],
            "zip_base": np.array([0.0], dtype=float),
            "zip_deltas": [],
        },
        "meta": {},
    }
    save_artifacts(artifacts, artifacts_path)

    df = pl.LazyFrame(
        {"NAICS_OVERRIDE": ["52"], "ZIP_OVERRIDE": ["12"], "is_written": [1]}
    )
    input_path = tmp_path / "incoming.parquet"
    save_parquet(df, input_path)

    output_path = tmp_path / "scored.parquet"

    exit_code = score.main(
        [
            "--data",
            str(input_path),
            "--artifacts",
            str(artifacts_path),
            "--output",
            str(output_path),
            "--naics-col",
            "NAICS_OVERRIDE",
            "--zip-col",
            "ZIP_OVERRIDE",
            "--prefix-fill",
            "0",
        ]
    )

    assert exit_code == 0
    assert pl.read_parquet(output_path).height == 1


def test_report_cli_outputs(tmp_path):
    artifacts_path = tmp_path / "bundle"
    artifacts = {
        "naics_maps": {"levels": ["L2"], "maps": [{"52": 0}], "cut_points": [2]},
        "zip_maps": {"levels": ["L2"], "maps": [{"12": 0}], "cut_points": [2]},
        "effects": {
            "beta0": 0.0,
            "naics_base": np.array([0.0], dtype=float),
            "naics_deltas": [],
            "zip_base": np.array([0.0], dtype=float),
            "zip_deltas": [],
        },
        "meta": {
            "target_col": "is_written",
            "naics_col": "NAICS",
            "zip_col": "ZIP",
            "prefix_fill": "0",
        },
    }
    save_artifacts(artifacts, artifacts_path)

    scored = pl.LazyFrame({"is_written": [1, 0, 1], "p": [0.9, 0.2, 0.6]})
    scored_path = tmp_path / "scored.parquet"
    save_parquet(scored, scored_path)

    out_dir = tmp_path / "reports"
    summary_path = tmp_path / "report_summary.json"

    exit_code = report.main(
        [
            "--scored",
            str(scored_path),
            "--out-dir",
            str(out_dir),
            "--artifacts",
            str(artifacts_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "calibration_reliability.csv").exists()
    assert (out_dir / "ranking_summary.csv").exists()
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    meta = json.loads(metrics_path.read_text())
    assert meta["rows"] == 3
    assert summary_path.exists()


def test_report_cli_requires_target_column(tmp_path):
    scored = pl.LazyFrame({"p": [0.1]})
    scored_path = tmp_path / "scored.parquet"
    save_parquet(scored, scored_path)

    out_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="Target column"):
        report.main(
            [
                "--scored",
                str(scored_path),
                "--out-dir",
                str(out_dir),
            ]
        )


def test_report_cli_target_override(tmp_path):
    scored = pl.LazyFrame({"is_written": [1], "prob": [0.8]})
    scored_path = tmp_path / "scored.parquet"
    save_parquet(scored, scored_path)

    out_dir = tmp_path / "reports"

    exit_code = report.main(
        [
            "--scored",
            str(scored_path),
            "--out-dir",
            str(out_dir),
            "--target-col",
            "is_written",
            "--prob-col",
            "prob",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "calibration_reliability.csv").exists()


def test_report_cli_probability_column_required(tmp_path):
    scored = pl.LazyFrame({"is_written": [1]})
    scored_path = tmp_path / "scored.parquet"
    save_parquet(scored, scored_path)

    out_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="Probability column"):
        report.main(
            [
                "--scored",
                str(scored_path),
                "--out-dir",
                str(out_dir),
                "--target-col",
                "is_written",
            ]
        )


def test_cli_entrypoint_help(capsys):
    assert cli_root.main([]) == 0
    captured = capsys.readouterr()
    assert "Available commands" in captured.out


def test_cli_entrypoint_unknown_command(capsys):
    assert cli_root.main(["unknown"]) == 2
    captured = capsys.readouterr()
    assert "Unknown command" in captured.err
