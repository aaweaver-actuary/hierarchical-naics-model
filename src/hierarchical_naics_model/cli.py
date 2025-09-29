from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import polars as pl
import pymc as pm

from .build_hierarchical_indices import build_hierarchical_indices
from .build_conversion_model import build_conversion_model
from .diagnostics import compute_rhat, posterior_predictive_checks


def _parse_cuts(arg: Optional[str]) -> Optional[List[int]]:
    if arg is None or arg.strip() == "":
        return None
    try:
        return [int(x) for x in arg.split(",") if x.strip()]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid cut points: {arg}") from e


def _ensure_columns(df: pl.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}; available: {df.columns}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hierarchical-naics-model",
        description="Fit a hierarchical NAICS/ZIP logistic model from a Parquet dataset.",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input Parquet file"
    )
    parser.add_argument(
        "--is-written-col",
        default="is_written",
        help="Binary outcome column name (default: is_written)",
    )
    parser.add_argument(
        "--naics-col", default="naics", help="NAICS code column name (default: naics)"
    )
    parser.add_argument(
        "--zip-col", default="zip", help="ZIP code column name (default: zip)"
    )

    parser.add_argument(
        "--naics-cuts",
        type=str,
        default=None,
        help="Comma-separated NAICS cut points (e.g., 2,3,4,5,6). If omitted, defaults are inferred.",
    )
    parser.add_argument(
        "--zip-cuts",
        type=str,
        default=None,
        help="Comma-separated ZIP cut points (e.g., 1,2,3,4,5). If omitted, defaults are inferred.",
    )
    parser.add_argument(
        "--prefix-fill-naics",
        type=str,
        default=None,
        help="Optional fill character to right-pad NAICS codes (e.g., '0').",
    )
    parser.add_argument(
        "--prefix-fill-zip",
        type=str,
        default=None,
        help="Optional fill character to right-pad ZIP codes (e.g., '0').",
    )

    # Sampling controls
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Optional number of rows to sample for fitting (for quick runs)",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=2025)

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional directory to save artifacts (idata .nc and a brief summary).",
    )

    args = parser.parse_args(argv)

    # Load data with Polars to avoid requiring pyarrow in pandas
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df_pl = pl.read_parquet(str(input_path))
    _ensure_columns(df_pl, [args.is_written_col, args.naics_col, args.zip_col])

    # Optional subset for speed
    if args.subset is not None and args.subset < len(df_pl):
        df_pl = df_pl.head(args.subset)

    # Extract columns
    y = df_pl[args.is_written_col].to_numpy()
    naics = df_pl[args.naics_col].cast(pl.Utf8).to_list()
    zips = df_pl[args.zip_col].cast(pl.Utf8).to_list()

    # Build indices (defaults inferred when cuts not provided)
    naics_cuts = _parse_cuts(args.naics_cuts)
    zip_cuts = _parse_cuts(args.zip_cuts)

    naics_idx = build_hierarchical_indices(
        naics, cut_points=naics_cuts, prefix_fill=args.prefix_fill_naics
    )
    zip_idx = build_hierarchical_indices(
        zips, cut_points=zip_cuts, prefix_fill=args.prefix_fill_zip
    )

    # Model
    model = build_conversion_model(
        y=np.asarray(y, dtype="int8"),
        naics_levels=np.asarray(naics_idx["code_levels"]),
        zip_levels=np.asarray(zip_idx["code_levels"]),
        naics_group_counts=list(naics_idx["group_counts"]),
        zip_group_counts=list(zip_idx["group_counts"]),
        target_accept=float(args.target_accept),
    )

    with model:
        idata = pm.sample(
            draws=int(args.draws),
            tune=int(args.tune),
            chains=int(args.chains),
            cores=int(args.cores),
            target_accept=float(args.target_accept),
            random_seed=int(args.random_seed),
            progressbar=True,
        )

    # Print a concise summary
    try:
        import arviz as az

        smry = az.summary(idata, var_names=["beta0"], kind="stats")
        print(smry.to_string())
    except Exception:  # pragma: no cover - summary failure just prints fallback
        print("Model fit complete; summary unavailable.")

    # Diagnostics: R-hat and PPC
    try:
        rhats = compute_rhat(idata, var_names=["beta0"])  # include more names as needed
        print({"rhat": rhats})
    except Exception:  # pragma: no cover - diagnostics are optional
        pass
    try:
        ppc_metrics = posterior_predictive_checks(
            model, idata, observed_name="is_written", samples=min(200, args.draws)
        )
        print({"ppc": ppc_metrics})
    except Exception:  # pragma: no cover - diagnostics are optional
        pass

    # Save artifacts if requested
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            idata.to_netcdf(outdir / "idata.nc")
        except Exception:  # pragma: no cover - optional artifact
            pass
        try:
            import arviz as az

            smry = az.summary(idata, var_names=["beta0"], kind="stats")
            smry.to_csv(outdir / "summary_beta0.csv")
        except Exception:  # pragma: no cover - optional artifact
            pass
        try:
            # Save diagnostics
            rhats = compute_rhat(idata, var_names=["beta0"])  # minimal for now
            (outdir / "diagnostics.txt").write_text(str(rhats))
        except Exception:  # pragma: no cover - optional artifact
            pass
        try:
            ppc_metrics = posterior_predictive_checks(
                model, idata, observed_name="is_written", samples=min(200, args.draws)
            )
            (outdir / "ppc.txt").write_text(str(ppc_metrics))
        except Exception:  # pragma: no cover - optional artifact
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
