from __future__ import annotations

# import argparse


def main(argv: list[str] | None = None) -> int:
    """
    Generate calibration & ranking reports for a scored dataset.

    CLI (suggested)
    ---------------
    nested-quotewrite report \
        --scored scored.parquet \
        --outdir reports/

    Returns
    -------
    int
        Exit code.
    """
    # TODO: argparse: load scored, compute calibration_report and ranking_report, write CSV/JSON.
    # parser = argparse.ArgumentParser(prog="nested-quotewrite report")
    # TODO: add args and implement
    return 0
