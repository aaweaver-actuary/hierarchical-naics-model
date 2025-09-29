from __future__ import annotations
from typing import List, Mapping, Sequence
from .logger import get_logger

log = get_logger(__name__)


def make_backoff_resolver(
    *,
    cut_points: Sequence[int],
    level_maps: Sequence[Mapping[str, int]],
    prefix_fill: str = "0",
):
    """Build a callable that maps a raw code (e.g., '511110', '30309') to
    a list of integer group indices for each hierarchy level, with backoff.

    Backoff policy
    --------------
    - Try most specific level first; if unseen, try the parent (shorter prefix),
      continuing until success. If no level matches, return None for that level.
    - Returning None for a level means "no contribution" from that level.

    Parameters
    ----------
    cut_points
        Character lengths per level (e.g., NAICS: [2,3,4,5,6]; ZIP: [2,3,5]).
    level_maps
        Per-level dict[str -> int] for seen training labels at that level.
    prefix_fill
        Character used to right-pad codes for safe slicing.

    Returns
    -------
    resolve : Callable[[str], List[int | None]]
        Given a code, returns a list of group indices (or None) aligned to levels.
    """
    cut_points = list(cut_points)
    level_maps = list(level_maps)

    def resolve(raw: str) -> List[int | None]:
        code = (raw or "").strip()
        max_len = max(cut_points)
        if len(code) < max_len:
            code = code.ljust(max_len, prefix_fill)

        out: List[int | None] = []
        log.debug(f"Resolving code: {code}")
        for j, c in enumerate(cut_points):
            k = j
            idx_or_none = None
            log.debug(f"  Level {j}: cut_point={c}")
            while k >= 0:
                lbl = code[: cut_points[k]]
                m = level_maps[k]
                log.debug(f"    Try level {k}: lbl={lbl}, map_keys={list(m.keys())}")
                if lbl in m:
                    idx_or_none = m[lbl]
                    log.debug(f"      Found: idx={idx_or_none}")
                    break
                k -= 1
            # Guarantee None for missing
            if idx_or_none is None:
                out.append(None)
            else:
                out.append(idx_or_none)
            log.debug(f"  Result for level {j}: {idx_or_none}")
        log.debug(f"Resolved indices: {out}")
        return out

    return resolve
