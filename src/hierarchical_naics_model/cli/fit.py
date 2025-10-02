from __future__ import annotations
# import sys


# def main(argv: list[str] | None = None) -> int:
#     """
#     Fit a nested-deltas model and write artifacts.
#     """
#     parser = argparse.ArgumentParser(
#         prog="nested-quotewrite fit",
#         description="Fit a nested-deltas model and write artifacts.",
#     )
#     parser.add_argument("--train", type=str, help="Path to training parquet file.")
#     parser.add_argument("--cut-naics", nargs="*", type=int, help="NAICS cut points.")
#     parser.add_argument("--cut-zip", nargs="*", type=int, help="ZIP cut points.")
#     parser.add_argument("--draws", type=int, default=1000, help="Number of draws.")
#     parser.add_argument(
#         "--tune", type=int, default=1000, help="Number of tuning steps."
#     )
#     parser.add_argument("--chains", type=int, default=2, help="Number of chains.")
#     parser.add_argument("--cores", type=int, default=1, help="Number of cores.")
#     parser.add_argument("--out", type=str, help="Output directory for artifacts.")
#     parser.parse_args(argv)
#     # TODO: implement actual fitting logic
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
