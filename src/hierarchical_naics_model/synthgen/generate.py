from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[neg])
    out[neg] = z / (1.0 + z)
    return out


def _pad(code: str, width: int, fill: str = "0") -> str:
    return code if len(code) >= width else code.ljust(width, fill)


def _labels_for_levels(
    code: str, cut_points: Sequence[int], max_width: int, fill: str = "0"
) -> List[str]:
    padded = _pad(code, max_width, fill)
    return [padded[:c] for c in cut_points]


@dataclass
class HierSpec:
    cut_points: List[int]
    branching: List[int]
    fill: str = "0"
    digits: str = "0123456789"

    def validate(self) -> None:
        if sorted(self.cut_points) != self.cut_points or len(
            set(self.cut_points)
        ) != len(self.cut_points):
            raise ValueError("cut_points must be strictly increasing")
        if len(self.branching) != max(0, len(self.cut_points) - 1):
            raise ValueError("branching must have length len(cut_points) - 1")
        if any(b < 1 for b in self.branching):
            raise ValueError("branching values must be >= 1")
        if len(self.fill) != 1:
            raise ValueError("fill must be a single character")


@dataclass
class TrueParams:
    beta0: float
    naics_base: List[float]
    naics_deltas: List[List[float]]
    zip_base: List[float]
    zip_deltas: List[List[float]]


@dataclass
class LevelMaps:
    levels: List[str]
    maps: List[Dict[str, int]]
    cut_points: List[int]


@dataclass
class SynthArtifacts:
    naics_maps: LevelMaps
    zip_maps: LevelMaps
    effects: Dict[str, Any]
    meta: Dict[str, Any]


def _build_hierarchy_labels(spec: HierSpec, seed: int | None = None) -> List[List[str]]:
    rng = random.Random(seed)
    cps = spec.cut_points
    levels: List[List[str]] = []

    w0 = cps[0]
    base_candidates = [str(d).zfill(w0) for d in range(10**w0)]
    rng.shuffle(base_candidates)
    n0 = min(len(base_candidates), 9 * (10 ** (w0 - 1)) if w0 > 0 else 1)
    level0 = sorted(base_candidates[: max(1, min(20, n0))])
    levels.append(level0)

    for li in range(1, len(cps)):
        prev = levels[-1]
        need = spec.branching[li - 1]
        width = cps[li]
        children: List[str] = []
        for p in prev:
            for _ in range(need):
                needed = width - len(p)
                tail = "".join(rng.choice(spec.digits) for _ in range(max(0, needed)))
                child = (p + tail)[:width]
                children.append(child)
        seen = set()
        uniq = []
        for x in children:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        levels.append(uniq)

    return levels


def _build_level_maps(levels: List[List[str]], cps: Sequence[int]) -> LevelMaps:
    maps = [{lbl: i for i, lbl in enumerate(level)} for level in levels]
    return LevelMaps(levels=[f"L{c}" for c in cps], maps=maps, cut_points=list(cps))


def _sample_params_for_levels(
    levels: List[List[str]], sigma0: float, decay: float, rng: np.random.Generator
) -> Tuple[List[float], List[List[float]]]:
    base = rng.normal(loc=0.0, scale=sigma0, size=len(levels[0])).astype(float).tolist()
    deltas: List[List[float]] = []
    for j, lev in enumerate(levels[1:], start=1):
        sigma = sigma0 * (decay**j)
        deltas.append(
            rng.normal(loc=0.0, scale=sigma, size=len(lev)).astype(float).tolist()
        )
    return base, deltas


def make_true_params(
    naics_levels: List[List[str]],
    zip_levels: List[List[str]],
    beta0: float = -0.5,
    naics_sigma0: float = 0.8,
    naics_decay: float = 0.6,
    zip_sigma0: float = 0.5,
    zip_decay: float = 0.6,
    seed: int | None = None,
) -> TrueParams:
    rng = np.random.default_rng(seed)
    naics_base, naics_deltas = _sample_params_for_levels(
        naics_levels, naics_sigma0, naics_decay, rng
    )
    zip_base, zip_deltas = _sample_params_for_levels(
        zip_levels, zip_sigma0, zip_decay, rng
    )
    return TrueParams(
        beta0=beta0,
        naics_base=naics_base,
        naics_deltas=naics_deltas,
        zip_base=zip_base,
        zip_deltas=zip_deltas,
    )


def _choose_leaf(levels: List[List[str]], rng: random.Random) -> str:
    return rng.choice(levels[-1]) if levels else ""


def _eta_for_codes(
    naics_code: str,
    zip_code: str,
    naics_maps: LevelMaps,
    zip_maps: LevelMaps,
    params: TrueParams,
) -> float:
    eta = params.beta0
    if naics_maps.cut_points:
        n_labels = _labels_for_levels(
            naics_code, naics_maps.cut_points, max(naics_maps.cut_points)
        )
        lbl0 = n_labels[0]
        idx0 = naics_maps.maps[0].get(lbl0)
        if idx0 is not None:
            eta += float(params.naics_base[idx0])
        for j in range(1, len(n_labels)):
            lbl = n_labels[j]
            idx = naics_maps.maps[j].get(lbl)
            if idx is not None:
                eta += float(params.naics_deltas[j - 1][idx])
    if zip_maps.cut_points:
        z_labels = _labels_for_levels(
            zip_code, zip_maps.cut_points, max(zip_maps.cut_points)
        )
        lbl0 = z_labels[0]
        idx0 = zip_maps.maps[0].get(lbl0)
        if idx0 is not None:
            eta += float(params.zip_base[idx0])
        for m in range(1, len(z_labels)):
            lbl = z_labels[m]
            idx = zip_maps.maps[m].get(lbl)
            if idx is not None:
                eta += float(params.zip_deltas[m - 1][idx])
    return eta


def generate_synthetic_dataset(
    n: int,
    naics_spec: HierSpec,
    zip_spec: HierSpec,
    *,
    params: TrueParams | None = None,
    seed: int | None = None,
    return_components: bool = True,
):
    naics_spec.validate()
    zip_spec.validate()

    rng = random.Random(seed)
    naics_levels = _build_hierarchy_labels(naics_spec, seed=rng.randint(0, 2**31 - 1))
    zip_levels = _build_hierarchy_labels(zip_spec, seed=rng.randint(0, 2**31 - 1))

    naics_maps = _build_level_maps(naics_levels, naics_spec.cut_points)
    zip_maps = _build_level_maps(zip_levels, zip_spec.cut_points)

    if params is None:
        params = make_true_params(
            naics_levels, zip_levels, seed=rng.randint(0, 2**31 - 1)
        )

    effects: Dict[str, Any] = {
        "beta0": params.beta0,
        "naics_base": params.naics_base,
        "naics_deltas": params.naics_deltas,
        "zip_base": params.zip_base,
        "zip_deltas": params.zip_deltas,
    }

    rows = []
    for _ in range(n):
        n_leaf = _choose_leaf(naics_levels, rng)
        z_leaf = _choose_leaf(zip_levels, rng)

        eta = _eta_for_codes(n_leaf, z_leaf, naics_maps, zip_maps, params)
        p = float(_sigmoid(np.array([eta]))[0])
        y = 1 if rng.random() < p else 0

        row = {"naics_code": n_leaf, "zip_code": z_leaf, "eta": eta, "p": p, "y": y}
        if return_components:
            row.update(
                {
                    "naics_L0": n_leaf[: naics_spec.cut_points[0]]
                    if naics_spec.cut_points
                    else "",
                    "zip_L0": z_leaf[: zip_spec.cut_points[0]]
                    if zip_spec.cut_points
                    else "",
                }
            )
        rows.append(row)

    df = pl.DataFrame(rows)

    artifacts = {
        "naics_maps": {
            "levels": naics_maps.levels,
            "maps": naics_maps.maps,
            "cut_points": naics_maps.cut_points,
        },
        "zip_maps": {
            "levels": zip_maps.levels,
            "maps": zip_maps.maps,
            "cut_points": zip_maps.cut_points,
        },
        "effects": effects,
        "meta": {
            "schema_version": 1,
            "seed": seed,
            "n": n,
            "naics_spec": asdict(naics_spec),
            "zip_spec": asdict(zip_spec),
        },
    }

    return df, artifacts


def _save_outputs(
    df: pl.DataFrame | pl.LazyFrame, artifacts, out_dir: str | None
) -> None:
    if out_dir is None:
        return
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = df.collect() if isinstance(df, pl.LazyFrame) else df
    try:
        data.write_parquet(out / "synthetic.parquet")
    except Exception:
        data.write_csv(out / "synthetic.csv")
    with (out / "artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate synthetic NAICS/ZIP hierarchical dataset."
    )
    p.add_argument(
        "--n", type=int, default=10_000, help="Number of samples to generate"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory to write parquet/csv + artifacts.json",
    )

    p.add_argument(
        "--naics-cuts",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="NAICS cut points",
    )
    p.add_argument(
        "--naics-branching",
        type=int,
        nargs="+",
        default=[3, 3, 3, 2],
        help="Children per parent per deeper level",
    )

    p.add_argument(
        "--zip-cuts", type=int, nargs="+", default=[3, 5], help="ZIP cut points"
    )
    p.add_argument(
        "--zip-branching",
        type=int,
        nargs="+",
        default=[20],
        help="Children per parent for ZIP deeper levels",
    )

    args = p.parse_args(argv)

    naics_spec = HierSpec(
        cut_points=args.naics_cuts, branching=args.naics_branching, fill="0"
    )
    zip_spec = HierSpec(
        cut_points=args.zip_cuts, branching=args.zip_branching, fill="0"
    )

    df, artifacts = generate_synthetic_dataset(
        n=args.n,
        naics_spec=naics_spec,
        zip_spec=zip_spec,
        seed=args.seed,
    )
    _save_outputs(df, artifacts, args.out_dir)

    print(f"Generated {df.height} rows.")
    if args.out_dir:
        print(f"Wrote data/artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
