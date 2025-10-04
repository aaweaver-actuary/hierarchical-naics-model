from __future__ import annotations

from pathlib import Path
import pickle
from typing import Dict, List, TypedDict

import numpy as np


class LevelMaps(TypedDict):
    """Per-level label→index dictionaries for one hierarchy."""

    levels: List[str]
    maps: List[Dict[str, int]]
    cut_points: List[int]


class EffectsTables(TypedDict):
    """Posterior mean effects for nested-deltas scoring."""

    beta0: float
    naics_base: np.ndarray
    naics_deltas: List[np.ndarray]
    zip_base: np.ndarray
    zip_deltas: List[np.ndarray]


class Artifacts(TypedDict):
    """Bundle of everything needed to score in production."""

    naics_maps: LevelMaps
    zip_maps: LevelMaps
    effects: EffectsTables
    meta: Dict[str, object]


def save_artifacts(art: Artifacts, path: str) -> None:
    """Serialize the artifacts bundle with pickle.

    The representation is intentionally simple: a single pickle file containing
    primitive Python containers and NumPy arrays. For a production system we
    would likely switch to a more structured format (Parquet/JSON), but this is
    sufficient for a proof of concept and keeps the code path compact.
    """

    target = Path(path)
    if target.suffix == "":  # allow passing a directory
        target.mkdir(parents=True, exist_ok=True)
        target = target / "artifacts.pkl"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("wb") as f:
        pickle.dump(art, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifacts(path: str) -> Artifacts:
    """Load the artifacts bundle previously written by :func:`save_artifacts`."""

    source = Path(path)
    if source.is_dir():
        source = source / "artifacts.pkl"
    if not source.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Artifacts file not found: {source}")

    with source.open("rb") as f:
        art = pickle.load(f)

    if not isinstance(art, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Artifacts file {source} did not contain a dictionary.")

    required_keys = {"naics_maps", "zip_maps", "effects", "meta"}
    missing = required_keys - set(art.keys())
    if missing:  # pragma: no cover - defensive guard
        raise ValueError(f"Artifacts file {source} is missing keys: {sorted(missing)}")

    return art  # type: ignore[return-value]
