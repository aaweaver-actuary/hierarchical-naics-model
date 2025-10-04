import argparse
import sys
from hierarchical_naics_model.synthgen.generate import (
    HierSpec,
    generate_synthetic_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hierarchical NAICS/ZIP data."
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of rows")
    parser.add_argument(
        "--naics-cut-points",
        nargs="*",
        type=int,
        default=[2, 3],
        help="NAICS cut points",
    )
    parser.add_argument(
        "--naics-branching", nargs="*", type=int, default=[2], help="NAICS branching"
    )
    parser.add_argument(
        "--zip-cut-points", nargs="*", type=int, default=[3, 5], help="ZIP cut points"
    )
    parser.add_argument(
        "--zip-branching", nargs="*", type=int, default=[2], help="ZIP branching"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out", type=str, default="synthetic.parquet", help="Output file (parquet)"
    )
    args = parser.parse_args()

    try:
        if args.n < 1:
            print("Error: --n must be >= 1", file=sys.stderr)
            sys.exit(1)
        naics_spec = HierSpec(
            cut_points=args.naics_cut_points, branching=args.naics_branching
        )
        zip_spec = HierSpec(
            cut_points=args.zip_cut_points, branching=args.zip_branching
        )
        df, artifacts = generate_synthetic_dataset(
            n=args.n, naics_spec=naics_spec, zip_spec=zip_spec, seed=args.seed
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    df.write_parquet(args.out)
    print(f"Synthetic data written to {args.out} with shape {df.shape}")


if __name__ == "__main__":
    main()
