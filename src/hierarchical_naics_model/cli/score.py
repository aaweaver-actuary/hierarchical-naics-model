# from __future__ import annotations
# import argparse
# import sys


# def main(argv: list[str] | None = None) -> int:
#     """
#     Score new rows using saved artifacts.
#     """
#     parser = argparse.ArgumentParser(
#         prog="nested-quotewrite score",
#         description="Score new rows using saved artifacts.",
#     )
#     parser.add_argument("--artifacts", type=str, help="Path to artifacts directory.")
#     parser.add_argument("--input", type=str, help="Path to input parquet file.")
#     parser.add_argument(
#         "--output", type=str, help="Path to output scored parquet file."
#     )
#     parser.parse_args(argv)
#     # TODO: implement actual scoring logic
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
