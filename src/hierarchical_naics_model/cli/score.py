from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..io.artifacts import Artifacts, load_artifacts
from ..io.datasets import load_parquet, save_parquet
from ..scoring.predict import predict_proba_nested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierarchical-naics-model score",
        description="Score records using previously fitted nested-delta effects.",
    )
    parser.add_argument("--data", required=True, help="Input Parquet file to score.")
    parser.add_argument(
        "--artifacts", required=True, help="Artifacts bundle from fitting."
    )
    parser.add_argument("--output", required=True, help="Destination Parquet file.")
    parser.add_argument(
        "--naics-col",
        help="Override NAICS column name (defaults to training metadata).",
    )
    parser.add_argument(
        "--zip-col",
        help="Override ZIP column name (defaults to training metadata).",
    )
    parser.add_argument(
        "--prefix-fill",
        help="Override right-padding character (defaults to training metadata).",
    )
    parser.add_argument(
        "--no-components",
        action="store_true",
        help="Skip per-level known/backoff flags for a lighter output.",
    )
    parser.add_argument(
        "--summary",
        help="Optional path to write a JSON scoring summary.",
    )
    return parser


def _get_setting(value, meta: dict, key: str):
    if value is not None:
        return value
    if key in meta:
        return meta[key]
    raise ValueError(
        f"Missing required option '{key}'. Provide a CLI override or ensure the artifacts metadata contains it."
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    artifacts: Artifacts = load_artifacts(args.artifacts)
    meta = artifacts.get("meta", {})

    naics_col = _get_setting(args.naics_col, meta, "naics_col")
    zip_col = _get_setting(args.zip_col, meta, "zip_col")
    prefix_fill = _get_setting(args.prefix_fill, meta, "prefix_fill")

    df = load_parquet(args.data)

    scored = predict_proba_nested(
        df,
        naics_col=naics_col,
        zip_col=zip_col,
        naics_cut_points=artifacts["naics_maps"]["cut_points"],
        zip_cut_points=artifacts["zip_maps"]["cut_points"],
        naics_level_maps=artifacts["naics_maps"]["maps"],
        zip_level_maps=artifacts["zip_maps"]["maps"],
        effects=artifacts["effects"],
        prefix_fill=prefix_fill,
        return_components=not args.no_components,
    )

    save_parquet(scored, args.output)

    if args.summary:
        summary = {
            "data": str(Path(args.data).resolve()),
            "output": str(Path(args.output).resolve()),
            "artifacts": str(Path(args.artifacts).resolve()),
            "rows_scored": int(len(scored)),
        }
        dst = Path(args.summary)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[score] completed",
        json.dumps({"output": str(Path(args.output).resolve())}, indent=None),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
