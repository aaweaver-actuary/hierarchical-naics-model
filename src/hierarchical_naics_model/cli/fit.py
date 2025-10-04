from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from ..core.hierarchy import build_hierarchical_indices
from ..io.artifacts import Artifacts, LevelMaps, save_artifacts
from ..io.datasets import load_parquet
from ..modeling.pymc_nested import PymcNestedDeltaStrategy
from ..scoring.extract import extract_effect_tables_nested


def _parse_cut_points(values: List[int] | None, default: List[int]) -> List[int]:
    if values:
        return [int(v) for v in values]
    return default.copy()


def _maps_to_plain_dicts(maps: Sequence[Mapping[str, int]]) -> List[Dict[str, int]]:
    plain: List[Dict[str, int]] = []
    for mapping in maps:
        plain.append({str(k): int(v) for k, v in mapping.items()})
    return plain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierarchical-naics-model fit",
        description="Fit the Bayesian nested-delta conversion model and persist artifacts.",
    )
    parser.add_argument("--train", required=True, help="Training data Parquet file.")
    parser.add_argument(
        "--target-col",
        default="is_written",
        help="Binary target column (default: %(default)s).",
    )
    parser.add_argument(
        "--naics-col",
        default="NAICS",
        help="NAICS code column (default: %(default)s).",
    )
    parser.add_argument(
        "--zip-col",
        default="ZIP",
        help="ZIP column (default: %(default)s).",
    )
    parser.add_argument(
        "--naics-cuts",
        nargs="*",
        type=int,
        help="NAICS cut points, e.g. --naics-cuts 2 3 6 (default: 2 3 6).",
    )
    parser.add_argument(
        "--zip-cuts",
        nargs="*",
        type=int,
        help="ZIP cut points, e.g. --zip-cuts 2 3 5 (default: 2 3 5).",
    )
    parser.add_argument(
        "--prefix-fill",
        default="0",
        help="Right-padding character for codes (default: %(default)s).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=500,
        help="Posterior draws per chain (default: %(default)s).",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=500,
        help="Number of tuning steps (default: %(default)s).",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of chains (default: %(default)s).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Parallel cores for sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.92,
        help="NUTS target_accept setting (default: %(default)s).",
    )
    parser.add_argument(
        "--student-t-level0",
        action="store_true",
        help="Use StudentT offsets for level-0 base vectors.",
    )
    parser.add_argument(
        "--inference-strategy",
        choices=("pymc_nuts", "pymc_advi", "pymc_map"),
        default="pymc_nuts",
        help="Inference backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--variational-steps",
        type=int,
        default=20000,
        help="Gradient steps for ADVI when using --inference-strategy=pymc_advi.",
    )
    parser.add_argument(
        "--map-method",
        default="BFGS",
        help="Optimizer method for MAP when using --inference-strategy=pymc_map.",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        help="Output file or directory for fitted artifacts.",
    )
    parser.add_argument(
        "--summary",
        help="Optional path to write a JSON training summary.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    naics_cuts = _parse_cut_points(args.naics_cuts, [2, 3, 6])
    zip_cuts = _parse_cut_points(args.zip_cuts, [2, 3, 5])

    df_lf = load_parquet(args.train)
    available_columns = set(df_lf.collect_schema().names())
    required_cols = {args.target_col, args.naics_col, args.zip_col}
    missing_cols = required_cols - available_columns
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Training data missing required columns: {missing}")

    df = df_lf.select(list(required_cols)).collect()

    y_series = df[args.target_col]
    y = y_series.to_numpy()
    if set(np.unique(y)) - {0, 1}:  # pragma: no cover - guard unexpected values
        raise ValueError(
            "Target column must contain only 0/1 values for the POC model."
        )
    y = y.astype(int)

    naics_idx = build_hierarchical_indices(
        df[args.naics_col].to_list(),
        cut_points=naics_cuts,
        prefix_fill=args.prefix_fill,
    )
    zip_idx = build_hierarchical_indices(
        df[args.zip_col].to_list(),
        cut_points=zip_cuts,
        prefix_fill=args.prefix_fill,
    )

    naics_levels = naics_idx["code_levels"].astype(int, copy=False)
    zip_levels = zip_idx["code_levels"].astype(int, copy=False)

    strategy_name = args.inference_strategy
    if strategy_name == "pymc_nuts":
        strategy = PymcNestedDeltaStrategy(
            default_target_accept=float(args.target_accept),
            use_student_t_level0=bool(args.student_t_level0),
        )
    elif strategy_name == "pymc_advi":
        from ..modeling.pymc_nested import PymcADVIStrategy

        strategy = PymcADVIStrategy(
            default_target_accept=float(args.target_accept),
            use_student_t_level0=bool(args.student_t_level0),
            fit_steps=int(args.variational_steps),
        )
    elif strategy_name == "pymc_map":
        from ..modeling.pymc_nested import PymcMAPStrategy

        strategy = PymcMAPStrategy(
            default_target_accept=float(args.target_accept),
            use_student_t_level0=bool(args.student_t_level0),
            map_kwargs={"method": args.map_method},
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown inference strategy '{strategy_name}'")

    model = strategy.build_model(
        y=y,
        naics_levels=naics_levels,
        zip_levels=zip_levels,
        naics_group_counts=naics_idx["group_counts"],
        zip_group_counts=zip_idx["group_counts"],
    )

    idata = strategy.sample_posterior(
        model,
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        cores=int(args.cores),
        target_accept=float(args.target_accept),
    )

    effects = extract_effect_tables_nested(idata)

    naics_maps: LevelMaps = {
        "levels": list(naics_idx["levels"]),
        "maps": _maps_to_plain_dicts(list(naics_idx["maps"])),
        "cut_points": list(naics_idx["cut_points"]),
    }
    zip_maps: LevelMaps = {
        "levels": list(zip_idx["levels"]),
        "maps": _maps_to_plain_dicts(list(zip_idx["maps"])),
        "cut_points": list(zip_idx["cut_points"]),
    }

    artifacts: Artifacts = {
        "naics_maps": naics_maps,
        "zip_maps": zip_maps,
        "effects": effects,
        "meta": {
            "train_path": str(Path(args.train).resolve()),
            "n_rows": int(df.height),
            "n_positive": int(y.sum()),
            "target_col": args.target_col,
            "naics_col": args.naics_col,
            "zip_col": args.zip_col,
            "naics_cut_points": naics_cuts,
            "zip_cut_points": zip_cuts,
            "prefix_fill": args.prefix_fill,
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "cores": int(args.cores),
            "target_accept": float(args.target_accept),
            "student_t_level0": bool(args.student_t_level0),
        },
    }

    save_artifacts(artifacts, args.artifacts)

    summary_path = args.summary
    if summary_path:
        summary_info = {
            "artifacts": str(Path(args.artifacts).resolve()),
            "meta": artifacts["meta"],
        }
        dst = Path(summary_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary_info, indent=2), encoding="utf-8")

    print(
        "[fit] completed",
        json.dumps({"artifacts": str(Path(args.artifacts).resolve())}, indent=None),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
