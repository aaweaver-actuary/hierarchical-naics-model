from __future__ import annotations

# import argparse


def main(argv: list[str] | None = None) -> int:
    """
    Score new rows using saved artifacts.

    CLI (suggested)
    ---------------
    nested-quotewrite score \
        --artifacts artifacts_dir \
        --input new_quotes.parquet \
        --output scored.parquet

    Returns
    -------
    int
        Exit code.
    """
    # TODO: argparse: load artifacts, read input, predict_proba_nested, write scored parquet.
    # parser = argparse.ArgumentParser(prog="nested-quotewrite score")
    # TODO: add args and implement
    return 0
