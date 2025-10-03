from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    _HAVE_PA = True
except Exception:  # pragma: no cover
    _HAVE_PA = False


class LevelMaps(TypedDict):
    levels: List[str]
    maps: List[Dict[str, int]]
    cut_points: List[int]


class Artifacts(TypedDict):
    naics_maps: LevelMaps
    zip_maps: LevelMaps
    effects: Dict[str, Any]
    meta: Dict[str, Any]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_artifacts_portable(directory: Path | str, artifacts: Artifacts) -> Path:
    """Save artifacts into a portable directory with JSON (and Parquet if available)."""
    directory = Path(directory)
    _ensure_dir(directory)

    (directory / "meta.json").write_text(
        json.dumps(artifacts["meta"], indent=2), encoding="utf-8"
    )
    (directory / "naics_maps.json").write_text(
        json.dumps(artifacts["naics_maps"], indent=2), encoding="utf-8"
    )
    (directory / "zip_maps.json").write_text(
        json.dumps(artifacts["zip_maps"], indent=2), encoding="utf-8"
    )

    effects = artifacts["effects"]
    use_parquet = _HAVE_PA and all(
        isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v)
        for v in effects.values()
    )
    if use_parquet:
        arrays = {k: pa.array(v, type=pa.float64()) for k, v in effects.items()}
        pq.write_table(pa.table(arrays), directory / "effects.parquet")
    else:
        (directory / "effects.json").write_text(
            json.dumps(effects, indent=2), encoding="utf-8"
        )

    return directory


def load_artifacts_portable(directory: Path | str) -> Artifacts:
    """Load artifacts saved by :func:`save_artifacts_portable`."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    naics_maps = json.loads((directory / "naics_maps.json").read_text(encoding="utf-8"))
    zip_maps = json.loads((directory / "zip_maps.json").read_text(encoding="utf-8"))

    effects_path_parquet = directory / "effects.parquet"
    effects_path_json = directory / "effects.json"
    if effects_path_parquet.exists() and _HAVE_PA:
        table = pq.read_table(effects_path_parquet)
        effects = {name: table[name].to_pylist() for name in table.column_names}
    elif effects_path_json.exists():
        effects = json.loads(effects_path_json.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Missing effects file in {directory}")

    return {
        "meta": meta,
        "naics_maps": naics_maps,
        "zip_maps": zip_maps,
        "effects": effects,
    }
