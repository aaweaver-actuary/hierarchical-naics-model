from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import polars as pl


__all__ = ["predict_proba_nested"]


def _pad_expr(expr: pl.Expr, *, width: int, fill: str) -> pl.Expr:
    if width <= 0:
        raise ValueError("width must be positive")
    if not isinstance(fill, str) or len(fill) != 1:
        raise ValueError("fill must be a single character string")

    fill_block = pl.lit(fill * width)
    cleaned = expr.cast(pl.Utf8).fill_null("").str.strip_chars()
    return (cleaned + fill_block).str.slice(0, width)


def _value_map(level_map: Mapping[str, int], values: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, idx in level_map.items():
        if idx < 0 or idx >= len(values):
            raise IndexError(
                f"Effect vector length {len(values)} is incompatible with index {idx}."
            )
        out[str(label)] = float(values[idx])
    return out


def _logistic_expr(eta: pl.Expr) -> pl.Expr:
    return (
        pl.when(eta >= 0)
        .then(1.0 / (1.0 + (-eta).exp()))
        .otherwise(eta.exp() / (1.0 + eta.exp()))
    )


def predict_proba_nested(
    df_new: pl.DataFrame | pl.LazyFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Mapping[str, object],
    prefix_fill: str = "0",
    return_components: bool = True,
) -> pl.DataFrame:
    """Score rows using nested delta effects with polars expressions."""

    if not isinstance(df_new, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("df_new must be a polars DataFrame or LazyFrame")

    # Normalise inputs
    beta0 = float(effects["beta0"])
    naics_base = np.asarray(effects["naics_base"], dtype=float)
    zip_base = np.asarray(effects["zip_base"], dtype=float)
    naics_deltas_raw = cast(Sequence[object], effects.get("naics_deltas", []) or [])
    zip_deltas_raw = cast(Sequence[object], effects.get("zip_deltas", []) or [])
    naics_deltas = [np.asarray(arr, dtype=float) for arr in naics_deltas_raw]
    zip_deltas = [np.asarray(arr, dtype=float) for arr in zip_deltas_raw]

    if len(naics_cut_points) != len(naics_level_maps):
        raise ValueError(
            "naics_cut_points and naics_level_maps must have the same length"
        )
    if len(zip_cut_points) != len(zip_level_maps):
        raise ValueError("zip_cut_points and zip_level_maps must have the same length")

    naics_vectors: list[np.ndarray] = [naics_base] + naics_deltas
    zip_vectors: list[np.ndarray] = [zip_base] + zip_deltas

    if len(naics_vectors) != len(naics_cut_points):
        raise ValueError(
            "Effects for NAICS do not match the number of levels; ensure deltas cover each level"
        )
    if len(zip_vectors) != len(zip_cut_points):
        raise ValueError(
            "Effects for ZIP do not match the number of levels; ensure deltas cover each level"
        )

    lf = df_new.lazy() if isinstance(df_new, pl.DataFrame) else df_new
    if isinstance(df_new, pl.DataFrame):
        original_cols = df_new.columns
    else:
        original_cols = list(df_new.collect_schema().names())

    effect_value_cols: list[str] = []
    known_cols: list[str] = []

    # Prepare NAICS level labels
    if naics_cut_points:
        max_len = max(naics_cut_points)
        padded = _pad_expr(pl.col(naics_col), width=max_len, fill=prefix_fill)
        for cut in naics_cut_points:
            col_name = f"__naics_L{cut}_label"
            lf = lf.with_columns(padded.str.slice(0, cut).alias(col_name))

        for cut, level_map, values in zip(
            naics_cut_points, naics_level_maps, naics_vectors
        ):
            label_col = f"__naics_L{cut}_label"
            value_col = f"__naics_L{cut}_value"
            value_map = _value_map(level_map, values)
            if value_map:
                mapping_df = pl.DataFrame(
                    {
                        label_col: list(value_map.keys()),
                        value_col: list(value_map.values()),
                    }
                ).lazy()
                lf = lf.join(mapping_df, on=label_col, how="left")
                if return_components:
                    known_col = f"naics_L{cut}_known"
                    lf = lf.with_columns(
                        pl.col(value_col).is_not_null().alias(known_col)
                    )
                    known_cols.append(known_col)
                lf = lf.with_columns(
                    pl.col(value_col).cast(pl.Float64).fill_null(0.0).alias(value_col)
                )
            else:
                lf = lf.with_columns(pl.lit(0.0).alias(value_col))
                if return_components:
                    known_col = f"naics_L{cut}_known"
                    lf = lf.with_columns(pl.lit(False).alias(known_col))
                    known_cols.append(known_col)
            effect_value_cols.append(value_col)

    # Prepare ZIP level labels
    if zip_cut_points:
        max_len = max(zip_cut_points)
        padded = _pad_expr(pl.col(zip_col), width=max_len, fill=prefix_fill)
        for cut in zip_cut_points:
            col_name = f"__zip_L{cut}_label"
            lf = lf.with_columns(padded.str.slice(0, cut).alias(col_name))

        for cut, level_map, values in zip(zip_cut_points, zip_level_maps, zip_vectors):
            label_col = f"__zip_L{cut}_label"
            value_col = f"__zip_L{cut}_value"
            value_map = _value_map(level_map, values)
            if value_map:
                mapping_df = pl.DataFrame(
                    {
                        label_col: list(value_map.keys()),
                        value_col: list(value_map.values()),
                    }
                ).lazy()
                lf = lf.join(mapping_df, on=label_col, how="left")
                if return_components:
                    known_col = f"zip_L{cut}_known"
                    lf = lf.with_columns(
                        pl.col(value_col).is_not_null().alias(known_col)
                    )
                    known_cols.append(known_col)
                lf = lf.with_columns(
                    pl.col(value_col).cast(pl.Float64).fill_null(0.0).alias(value_col)
                )
            else:
                lf = lf.with_columns(pl.lit(0.0).alias(value_col))
                if return_components:
                    known_col = f"zip_L{cut}_known"
                    lf = lf.with_columns(pl.lit(False).alias(known_col))
                    known_cols.append(known_col)
            effect_value_cols.append(value_col)

    # eta and probability

    if effect_value_cols:
        eta_expr = pl.fold(
            acc=pl.lit(beta0),
            exprs=[pl.col(name) for name in effect_value_cols],
            function=lambda acc, x: acc + x,
        ).alias("eta")
    else:
        eta_expr = pl.lit(beta0).alias("eta")

    lf = lf.with_columns(eta_expr)
    lf = lf.with_columns(_logistic_expr(pl.col("eta")).alias("p"))

    if return_components:
        if known_cols:
            lf = lf.with_columns(
                (~pl.all_horizontal([pl.col(name) for name in known_cols])).alias(
                    "any_backoff"
                )
            )
        else:
            lf = lf.with_columns(pl.lit(False).alias("any_backoff"))

    # Project the final set of columns
    projected_cols = [pl.col(name) for name in original_cols] + [
        pl.col("eta"),
        pl.col("p"),
    ]
    if return_components:
        projected_cols.extend(pl.col(name) for name in known_cols)
        projected_cols.append(pl.col("any_backoff"))

    result = lf.select(projected_cols).collect()
    return result
