from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, cast

import polars as pl

from ..eval.calibration import calibration_report
from ..eval.ranking import ranking_report
from ..io.artifacts import load_artifacts
from ..io.datasets import load_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierarchical-naics-model report",
        description="Generate calibration and ranking diagnostics for scored data.",
    )
    parser.add_argument("--scored", required=True, help="Scored Parquet file.")
    parser.add_argument(
        "--out-dir", required=True, help="Directory for report outputs."
    )
    parser.add_argument(
        "--target-col",
        help="Outcome column (defaults to training metadata if --artifacts is provided).",
    )
    parser.add_argument(
        "--prob-col",
        default="p",
        help="Probability column (default: %(default)s).",
    )
    parser.add_argument(
        "--artifacts",
        help="Optional artifacts bundle to infer default column names.",
    )
    parser.add_argument(
        "--ks",
        nargs="*",
        type=int,
        help="Percent thresholds for ranking metrics (default: 5 10 20 30).",
    )
    parser.add_argument(
        "--summary",
        help="Optional path to write a compact JSON summary.",
    )
    return parser


def _resolve_target_col(
    args_target: Optional[str], artifacts_path: Optional[str]
) -> Optional[str]:
    if args_target:
        return args_target
    if not artifacts_path:
        return None
    art = load_artifacts(artifacts_path)
    meta = art.get("meta", {})
    return cast(Optional[str], meta.get("target_col"))


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target_col = _resolve_target_col(args.target_col, args.artifacts)
    if not target_col:
        raise ValueError(
            "Target column not specified. Provide --target-col or supply --artifacts with metadata."
        )

    lf = load_parquet(args.scored)
    available_columns = set(lf.collect_schema().names())
    if target_col not in available_columns:
        raise ValueError(f"Target column '{target_col}' not found in scored data.")
    if args.prob_col not in available_columns:
        raise ValueError(
            f"Probability column '{args.prob_col}' not found in scored data."
        )

    df = lf.select([pl.col(target_col), pl.col(args.prob_col)]).collect()
    y_series = df[target_col]
    p_series = df[args.prob_col]

    cal = calibration_report(y_series, p_series, bins=10)
    ks: Iterable[int] = tuple(args.ks) if args.ks else (5, 10, 20, 30)
    rank = ranking_report(y_series, p_series, ks=ks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reliability_obj = cal["reliability"]
    ranking_obj = rank["summary"]

    if not isinstance(reliability_obj, pl.DataFrame):  # pragma: no cover - defensive
        raise TypeError("Calibration report did not produce a DataFrame.")
    if not isinstance(ranking_obj, pl.DataFrame):  # pragma: no cover - defensive
        raise TypeError("Ranking report did not produce a DataFrame.")

    reliability_obj.write_csv(out_dir / "calibration_reliability.csv")
    ranking_obj.write_csv(out_dir / "ranking_summary.csv")

    metrics = {
        "scored_path": str(Path(args.scored).resolve()),
        "rows": int(df.height),
        "positives": int(df[target_col].sum()),
        "target_col": target_col,
        "prob_col": args.prob_col,
        "ece": cal["ece"],
        "brier": cal["brier"],
        "log_loss": cal["log_loss"],
        "base_rate": rank["base_rate"],
        "ks": list(ks),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "[report] completed",
        json.dumps({"out_dir": str(out_dir.resolve())}, indent=None),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
