from __future__ import annotations

# import argparse


def main(argv: list[str] | None = None) -> int:
    """
    Fit a nested-deltas model and write artifacts.

    CLI (suggested)
    ---------------
    nested-quotewrite fit \
        --train parquet_path \
        --cut-naics 2 3 4 5 6 \
        --cut-zip 2 3 5 \
        --draws 1000 --tune 1000 --chains 2 --cores 1 \
        --out artifacts_dir

    Returns
    -------
    int
        Exit code (0 = success).
    """
    # TODO: argparse: read data, build indices, build model, sample, extract tables, save artifacts.
    # NOTE: include fast flags for tests: --no-idata, --draws 100 --tune 100 etc.
    # parser = argparse.ArgumentParser(prog="nested-quotewrite fit")
    # TODO: add args and implement
    return 0
