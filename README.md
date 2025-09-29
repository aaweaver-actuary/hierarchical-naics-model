# hierarchical-naics-model

Proof-of-concept hierarchical model utilities around NAICS and ZIP codes.

## What’s here

- `generate_synthetic_data.py` — creates a toy binary outcome dataset using NAICS and ZIP code effects.
- `build_hierarchical_indices.py` — converts hierarchical string codes into per-level integer indices for modeling.
- `build_conversion_model.py` — builds a PyMC logistic hierarchical model using the indices.

## Running tests

This repo includes a pytest suite. The simplest way is to use the Makefile:

```bash
make test
```

This runs pytest with coverage and fails if total coverage is below 95%.

Alternatively, run directly with `uv` or `pip`:

```bash
uv run pytest -q
# or
pip install -e .[dev]
pytest -q
```

## Expected input data schema

End-to-end, the model expects a rectangular dataset with the following columns:

- is_written: int 0/1 — Binary outcome for each observation.
- naics: string — NAICS industry code per observation. Can be 2–6 chars; shorter codes are allowed.
- zip: string — ZIP code per observation (5 characters recommended). Strings are preferred to preserve leading zeros (e.g., "02139").

Example minimal Pandas frame:

```python
import pandas as pd

df = pd.DataFrame({
	"is_written": [0, 1, 0, 1],
	"naics": ["511110", "511120", "52", "52413"],
	"zip": ["30309", "94103", "02139", "10001"],
})
```

To prepare hierarchical indices for modeling, you can either let the library infer cut points or specify them explicitly:

- Default inference:
	- If codes have max length 6 (NAICS-like), uses [2, 3, 4, 5, 6].
	- If codes have max length ≤ 5 (ZIP-like), uses [1, 2, ..., max_len].
- Explicit examples:
	- NAICS: [2, 3, 6] (2-digit sector, 3-digit subsector, 6-digit industry)
	- ZIP: [2, 3, 5]

Then build indices and the model:

```python
from hierarchical_naics_model.build_hierarchical_indices import build_hierarchical_indices
from hierarchical_naics_model.build_conversion_model import build_conversion_model

naics_cuts = [2, 3, 6]
zip_cuts = [2, 3, 5]

# Using defaults (recommended in many cases):
naics_idx = build_hierarchical_indices(df["naics"].astype(str).tolist())
zip_idx = build_hierarchical_indices(df["zip"].astype(str).tolist())

# Or explicitly specifying cut points:
# naics_idx = build_hierarchical_indices(df["naics"].astype(str).tolist(), cut_points=[2,3,6])
# zip_idx   = build_hierarchical_indices(df["zip"].astype(str).tolist(),   cut_points=[2,3,5])

model = build_conversion_model(
	y=df["is_written"].to_numpy(),
	naics_levels=naics_idx["code_levels"],
	zip_levels=zip_idx["code_levels"],
	naics_group_counts=naics_idx["group_counts"],
	zip_group_counts=zip_idx["group_counts"],
	target_accept=0.9,
)

import pymc as pm
with model:
	idata = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9)
```

Notes:

- For numeric-like codes with variable lengths (e.g., short NAICS), you can pass `prefix_fill="0"` to `build_hierarchical_indices` to right-pad shorter codes, ensuring consistent slicing at deeper levels.
- The model treats NAICS and ZIP as separate hierarchical random effects and adds them to a global intercept.

