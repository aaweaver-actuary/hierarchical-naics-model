# hierarchical-naics-model

Proof-of-concept hierarchical model utilities around NAICS and ZIP codes.

## What’s here

- `generate_synthetic_data.py` — creates a toy binary outcome dataset using NAICS and ZIP code effects.
- `build_hierarchical_indices.py` — converts hierarchical string codes into per-level integer indices for modeling.
- `build_conversion_model.py` — builds a PyMC logistic hierarchical model using the indices.

## Running tests

This repo includes a pytest suite. With `uv`:

```bash
uv run pytest -q
```

Or with a standard Python environment:

```bash
pip install -e .[dev]
pytest -q
```

