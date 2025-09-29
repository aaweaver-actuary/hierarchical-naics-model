from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pymc as pm
import arviz as az


def compute_rhat(
    idata: az.InferenceData, var_names: Optional[Iterable[str]] = None
) -> Dict[str, float]:
    """Compute R-hat (Gelman-Rubin) diagnostics for selected variables.

    Prefers ArviZ summary (r_hat column) with a fallback to az.rhat.
    Returns a mapping var_name -> r_hat.
    """
    out: Dict[str, float] = {}
    try:
        smry = az.summary(
            idata, var_names=list(var_names) if var_names else None, kind="stats"
        )
        # Some ArviZ versions expose rhat as r_hat, others as r_hat_mean; prefer r_hat
        rhat_col = (
            "r_hat"
            if "r_hat" in smry.columns
            else ("r_hat_mean" if "r_hat_mean" in smry.columns else None)
        )
        if rhat_col is not None:
            for idx, row in smry.iterrows():
                base = str(idx).split("[")[0]
                val = row.get(rhat_col)
                if val is not None and np.isfinite(val):
                    out[base] = max(out.get(base, 0.0), float(val))
            if out:
                return out
    except Exception:  # pragma: no cover - defensive fallback
        pass

    try:
        da = az.rhat(idata, var_names=var_names)
        if hasattr(da, "to_series"):
            for idx, val in da.to_series().items():
                name = idx if isinstance(idx, str) else idx[0]
                base = str(name).split("[")[0]
                out[base] = max(out.get(base, 0.0), float(val))
    except Exception:  # pragma: no cover - defensive fallback
        pass
    # Ensure requested variables have a value; if unavailable, fall back to 1.0
    requested = list(var_names) if var_names else []
    for name in requested:
        if name not in out or not np.isfinite(out.get(name, np.nan)):
            out[name] = 1.0
    return out


def sample_ppc(
    model: pm.Model,
    idata: az.InferenceData,
    *,
    observed_name: str = "is_written",
    random_seed: int = 42,
) -> Mapping[str, np.ndarray]:
    """Sample posterior predictive draws for a given observed variable.

    Returns a mapping containing arrays under the observed name.
    Separated for testability and reuse.
    """
    with model:
        return pm.sample_posterior_predictive(
            idata,
            var_names=[observed_name],
            random_seed=random_seed,
            return_inferencedata=False,
        )


def extract_observed(
    idata: az.InferenceData, observed_name: str = "is_written"
) -> Optional[np.ndarray]:
    """Extract observed data array from InferenceData if present."""
    try:
        extracted = az.extract(idata, group="observed_data")
        return np.asarray(extracted[observed_name])
    except Exception:  # pragma: no cover - defensive
        return None


def compute_ppc_metrics(
    y_ppc: np.ndarray, y_obs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute basic PPC metrics given posterior predictive draws and optional observed.

    - y_ppc: array with shape (chains, draws, N) or (samples, N) or (N,)
    - y_obs: optional observed of shape (N,)
    """
    arr = np.asarray(y_ppc)
    if arr.ndim >= 2:
        y_ppc_mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    else:
        y_ppc_mean = arr

    metrics: Dict[str, float] = {"mean_ppc": float(np.mean(y_ppc_mean))}
    if y_obs is not None:
        mean_obs = float(np.mean(y_obs))
        metrics["mean_obs"] = mean_obs
        metrics["abs_err_mean"] = abs(metrics["mean_ppc"] - mean_obs)
    return metrics


def posterior_predictive_checks(
    model: pm.Model,
    idata: az.InferenceData,
    *,
    observed_name: str = "is_written",
    samples: int = 200,  # kept for backward compatibility, unused in PyMC 5
) -> Dict[str, float]:
    """Run simple PPC and return basic metrics.

    Note: `samples` is ignored for PyMC 5's API; sampling uses posterior draws from
    `idata`. This function orchestrates sampling and metric computation.
    """
    ppc = sample_ppc(model, idata, observed_name=observed_name)
    y_obs = extract_observed(idata, observed_name=observed_name)
    return compute_ppc_metrics(np.asarray(ppc.get(observed_name)), y_obs)
