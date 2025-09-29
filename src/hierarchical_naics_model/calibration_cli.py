from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
import json
from .calibration_and_lift_report import calibration_and_lift_report
from .plot_calibration_and_lift import (
    plot_reliability_curve,
    plot_lift_and_gain,
    plot_precision_at_k,
    plot_brier_logloss,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="calibration-cli",
        description="Compute calibration and lift metrics, output metrics.json and CSVs, and plot interactive visualizations.",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input Parquet or CSV file"
    )
    parser.add_argument(
        "--y-col", default="is_written", help="Binary outcome column name"
    )
    parser.add_argument(
        "--p-col", default="p_hat", help="Predicted probability column name"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Temporal split: train ≤ cutoff, score > cutoff (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-col", type=str, default=None, help="Date column for temporal split"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write metrics.json, CSVs, and plots",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of bins for reliability curve"
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10,20,30",
        help="Comma-separated percent thresholds for ranking metrics",
    )
    parser.add_argument(
        "--plotly-html", action="store_true", help="Save interactive plotly HTML files"
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.input.endswith(".parquet"):
        df = pl.read_parquet(args.input).to_pandas()
    else:
        df = pl.read_csv(args.input).to_pandas()

    # Temporal split
    if args.cutoff_date and args.date_col:
        train = df[df[args.date_col] <= args.cutoff_date]
        score = df[df[args.date_col] > args.cutoff_date]
    else:
        train = df
        score = df

    # Metrics
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    metrics = calibration_and_lift_report(
        y_true=score[args.y_col],
        p_hat=score[args.p_col],
        bins=args.bins,
        ks=ks,
    )
    # Write metrics.json
    summary = metrics["summary"]
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Write CSVs
    metrics["reliability"].to_csv(outdir / "reliability.csv", index=False)
    metrics["ranking"].to_csv(outdir / "ranking.csv", index=False)
    metrics["sorted_scores"].to_csv(outdir / "sorted_scores.csv", index=False)

    # Plotly visualizations
    fig_rel = plot_reliability_curve(metrics["reliability"])
    fig_lift = plot_lift_and_gain(metrics["ranking"])
    fig_prec = plot_precision_at_k(metrics["ranking"])
    fig_brier = plot_brier_logloss(metrics["summary"])
    if args.plotly_html:
        fig_rel.write_html(outdir / "reliability_curve.html")
        fig_lift.write_html(outdir / "lift_and_gain.html")
        fig_prec.write_html(outdir / "precision_at_k.html")
        fig_brier.write_html(outdir / "brier_logloss.html")
    else:
        fig_rel.write_image(outdir / "reliability_curve.png")
        fig_lift.write_image(outdir / "lift_and_gain.png")
        fig_prec.write_image(outdir / "precision_at_k.png")
        fig_brier.write_image(outdir / "brier_logloss.png")


if __name__ == "__main__":
    main()
