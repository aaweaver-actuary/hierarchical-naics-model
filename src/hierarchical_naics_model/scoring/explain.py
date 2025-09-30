from __future__ import annotations

from typing import Mapping, Sequence, Dict


def explain_row(
    naics_code: str,
    zip_code: str,
    *,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Dict[str, object],
    prefix_fill: str = "0",
) -> Dict[str, object]:
    """
    Produce a path-wise decomposition for a single (NAICS, ZIP) pair.

    Returns
    -------
    dict
        {
          "beta0": float,
          "naics_path": list[tuple[level_label, contribution]],
          "zip_path": list[tuple[level_label, contribution]],
          "eta": float,
          "p": float,
        }

    Notes
    -----
    - Uses parent backoff when a leaf/child is missing.
    - Intended for stakeholder-facing “why this score?” views.
    """
    # TODO: implement (re-use same resolution logic as predict, but capture labels and contributions).
    raise NotImplementedError
