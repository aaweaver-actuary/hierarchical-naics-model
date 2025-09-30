from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def generate_synthetic_data(
    n: int,
    *,
    naics_codes: Sequence[str],
    zip_codes: Sequence[str],
    base_logit: float = -1.5,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Generate toy binary conversion data driven by (NAICS, ZIP) hierarchical effects.

    Parameters
    ----------
    n
        Number of rows to generate.
    naics_codes
        A pool of NAICS strings to sample from.
    zip_codes
        A pool of ZIP strings to sample from.
    base_logit
        Global intercept on the logit scale.
    seed
        RNG seed.

    Returns
    -------
    df : pandas.DataFrame with columns
        - 'is_written' : 0/1 outcome
        - 'naics' : sampled code
        - 'zip' : sampled code

    Notes
    -----
    - The DGP is intentionally simple: outcome logit is a baseline plus
      one (non-hierarchical) lookup per code. This is just to create a
      plausible-looking classification task for method smoke-tests.
    """
    if not naics_codes or not zip_codes:
        raise ValueError("naics_codes and zip_codes must be non-empty sequences.")
    rng = np.random.default_rng(seed)
    naics = rng.choice(np.asarray(naics_codes), size=n, replace=True)
    zips = rng.choice(np.asarray(zip_codes), size=n, replace=True)

    # Make some stable code-level nudges
    uni_naics = pd.Index(naics_codes)
    uni_zip = pd.Index(zip_codes)
    nudge_naics = rng.normal(0.0, 0.4, size=len(uni_naics))
    nudge_zip = rng.normal(0.0, 0.4, size=len(uni_zip))

    logit = (
        base_logit
        + nudge_naics[uni_naics.get_indexer(naics)]
        + nudge_zip[uni_zip.get_indexer(zips)]
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p, size=n)

    return pd.DataFrame({"is_written": y, "naics": naics, "zip": zips})
