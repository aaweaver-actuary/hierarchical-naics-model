from __future__ import annotations

from typing import Mapping, Sequence, Dict, Any, List, Tuple, Optional
import math


def _pad(code: Optional[str], width: int, fill: str) -> str:
    raw = "" if code is None else str(code).strip()
    return raw if len(raw) >= width else raw.ljust(width, fill)


def _labels_for_levels(
    code: str, cut_points: Sequence[int], max_width: int, fill: str
) -> List[str]:
    padded = _pad(code, max_width, fill)
    return [padded[:c] for c in cut_points]


def _logistic(x: float) -> float:
    # numerically stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def explain_row(
    naics_code: str,
    zip_code: str,
    *,
    naics_cut_points: Sequence[int],
    zip_cut_points: Sequence[int],
    naics_level_maps: Sequence[Mapping[str, int]],
    zip_level_maps: Sequence[Mapping[str, int]],
    effects: Dict[str, Any],
    prefix_fill: str = "0",
) -> Dict[str, Any]:
    """Explain a single row's prediction by decomposing contributions.

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
    - If a level label is unknown at its level map, the contribution is ``0.0`` and
      the label is returned with a " (unknown)" suffix for clarity.
    - We do not substitute a parent at a deeper level; parent contributions are
      already captured at their own level in nested-delta modeling.
    """
    beta0 = float(effects.get("beta0", 0.0))
    naics_base = effects.get("naics_base", [])
    naics_deltas = effects.get("naics_deltas", [])
    zip_base = effects.get("zip_base", [])
    zip_deltas = effects.get("zip_deltas", [])

    max_n = max(naics_cut_points) if naics_cut_points else 0
    max_z = max(zip_cut_points) if zip_cut_points else 0
    n_labels = (
        _labels_for_levels(naics_code, naics_cut_points, max_n, prefix_fill)
        if max_n
        else []
    )
    z_labels = (
        _labels_for_levels(zip_code, zip_cut_points, max_z, prefix_fill)
        if max_z
        else []
    )

    eta = beta0
    naics_path: List[Tuple[str, float]] = []
    zip_path: List[Tuple[str, float]] = []

    # NAICS
    if n_labels:
        lbl0 = n_labels[0]
        if lbl0 in naics_level_maps[0]:
            idx0 = naics_level_maps[0][lbl0]
            v = float(naics_base[idx0])
            naics_path.append((lbl0, v))
            eta += v
        else:
            naics_path.append((f"{lbl0} (unknown)", 0.0))
        for j in range(1, len(n_labels)):
            lbl = n_labels[j]
            if j < len(naics_level_maps) and lbl in naics_level_maps[j]:
                idx = naics_level_maps[j][lbl]
                v = float(naics_deltas[j - 1][idx])
                naics_path.append((lbl, v))
                eta += v
            else:
                naics_path.append((f"{lbl} (unknown)", 0.0))

    # ZIP
    if z_labels:
        lbl0 = z_labels[0]
        if lbl0 in zip_level_maps[0]:
            idx0 = zip_level_maps[0][lbl0]
            v = float(zip_base[idx0])
            zip_path.append((lbl0, v))
            eta += v
        else:
            zip_path.append((f"{lbl0} (unknown)", 0.0))
        for m in range(1, len(z_labels)):
            lbl = z_labels[m]
            if m < len(zip_level_maps) and lbl in zip_level_maps[m]:
                idx = zip_level_maps[m][lbl]
                v = float(zip_deltas[m - 1][idx])
                zip_path.append((lbl, v))
                eta += v
            else:
                zip_path.append((f"{lbl} (unknown)", 0.0))

    p = _logistic(eta)
    return {
        "beta0": beta0,
        "naics_path": naics_path,
        "zip_path": zip_path,
        "eta": eta,
        "p": p,
    }
