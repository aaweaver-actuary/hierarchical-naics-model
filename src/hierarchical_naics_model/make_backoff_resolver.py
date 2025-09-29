from __future__ import annotations
from typing import List, Mapping, Sequence


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
        for j, c in enumerate(cut_points):
            # attempt exact level, else parent, etc.
            k = j
            idx_or_none = None
            while k >= 0:
                lbl = code[: cut_points[k]]
                m = level_maps[k]
                if lbl in m:
                    idx_or_none = m[lbl]
                    break
                k -= 1
            out.append(idx_or_none)
        return out

    return resolve
