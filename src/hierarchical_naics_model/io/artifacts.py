from __future__ import annotations

from typing import Dict, List, Mapping, TypedDict
import pandas as pd


class LevelMaps(TypedDict):
    """Per-level label→index dictionaries for one hierarchy."""

    levels: List[str]
    maps: List[Mapping[str, int]]
    cut_points: List[int]


class EffectsTables(TypedDict):
    """Posterior mean effects for nested-deltas scoring."""

    beta0: float
    naics_base: pd.Series
    naics_deltas: List[pd.Series]
    zip_base: pd.Series
    zip_deltas: List[pd.Series]


class Artifacts(TypedDict):
    """Bundle of everything needed to score in production."""

    naics_maps: LevelMaps
    zip_maps: LevelMaps
    effects: EffectsTables
    meta: Dict[str, object]


def save_artifacts(art: Artifacts, path: str) -> None:
    """
    Serialize artifacts bundle to disk (e.g., pickle/feather/JSON hybrids).

    Parameters
    ----------
    art
        Artifacts bundle.
    path
        Destination path (directory or file per your format).
    """
    # TODO: implement (choose a format: joblib + feather e.g.).
    raise NotImplementedError


def load_artifacts(path: str) -> Artifacts:
    """
    Load artifacts bundle from disk.

    Parameters
    ----------
    path
        Source path.

    Returns
    -------
    Artifacts
        Deserialized bundle.
    """
    # TODO: implement
    raise NotImplementedError
