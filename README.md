# hierarchical-naics-model

Proof-of-concept hierarchical model utilities around NAICS and ZIP codes.

## 1. Project Overview

This project develops a **Bayesian logistic model** to predict whether an insurance quote will convert into a **written policy**. The output is designed to support **sales triage**: providing reliable conversion probabilities so that underwriters and agents can focus their efforts on submissions most likely to succeed.

### Goals
- **Practical triage**: Provide actionable conversion probabilities for sales and underwriting.
- **Robustness**: Handle **high-cardinality categorical predictors** (NAICS industry codes and ZIP codes) without overfitting.
- **Interpretability**: Produce decomposable effects (intercept, industry base, industry deltas, geography base, geography deltas) that are explainable to business stakeholders.
- **Production readiness**: Support deterministic scoring, auditable backoff for unseen codes, and calibration/lift reporting.

### Key Methods
- **Hierarchical structure**: Both NAICS and ZIP codes are treated as **hierarchies** (NAICS: 2→3→4→6 digits; ZIP: digit prefixes). At each level, codes inherit information from their parent and only deviate when supported by data.
- **Nested delta specification**: Each hierarchy starts with a **base effect** (coarse level), then adds **nested deltas** for deeper levels. This ensures that deeper refinements only emerge if supported by sufficient data.
- **Shrinkage priors**: Priors shrink deeper levels more strongly (exponentially decreasing variance by depth), balancing flexibility with conservatism.
- **Bayesian inference with PyMC**: The model is implemented in PyMC, using NUTS/HMC for posterior sampling. Non-centered parameterization improves sampling efficiency.
- **Scoring with explicit backoff rules**: When a code at a given level is unseen, the contribution from that level is set to zero rather than substituting a parent or sibling effect. This avoids unintended leakage and maintains interpretability.
- **Evaluation**: Model outputs are assessed with calibration curves, Expected Calibration Error (ECE), Brier score, log-loss, and ranking metrics such as Precision@k% and Lift@k%.

### Business Context
This is a **non-standard application** of Bayesian hierarchical modeling in actuarial practice:
- Actuarial models often focus on **loss ratios** or **pricing adequacy**.  
- Here, the target is the **conversion probability** (hit ratio), and the covariates are categorical **hierarchies**.  
- By leveraging NAICS and ZIP in a principled, hierarchical Bayesian framework, the model respects structure, shares information sensibly, and provides **business-aligned insights** into where opportunities lie and which submissions are most worth pursuing.


## 2. Architecture & Modules

The codebase is organized for **clarity**, **testability**, and **production readiness**. Each top-level package has a clear responsibility, with **pure functions** in `core` and `eval`, **probabilistic modeling** isolated in `modeling`, and **deterministic scoring** in `scoring`. This separation ensures reproducibility, easy unit testing, and predictable behavior in production.

### 2.1 `core`

- **`core/hierarchy.py`**
  - Builds hierarchical indices from NAICS and ZIP codes using right-padding.
  - Produces `code_levels` (N × L array of integer indices per row and level), `unique_per_level`, `maps` (label → index), `group_counts`, and `parent_index_per_level`.
  - Includes a **backoff resolver** that determines how unseen codes fall back to higher levels.
- **`core/validation.py`**
  - Centralized input validation (e.g., check that cut points are strictly increasing, indices are in-bounds, arrays align).
  - Reduces duplication across modeling and scoring.
- **`core/effects.py`**
  - Pure linear predictor functions:
    - `eta_additive`: sum across per-level vectors.
    - `eta_nested`: base + deltas with nesting.
  - Enables **unit testing of math logic** without PyMC.

**Implementation principle:** fast, deterministic, and test-heavy. All of these are validated independently of Bayesian sampling.

```python
# Minimal example: build hierarchical indices for NAICS & ZIP
import pandas as pd
from hierarchical_naics_model.core.hierarchy import build_hierarchical_indices

df = pd.DataFrame({
    "NAICS": ["311000","311100","522120","521110"],
    "ZIP":   ["45242","30301","10001","60601"],
})

naics_cuts = [2,3,6]
zip_cuts   = [2,3,5]
h_naics = build_hierarchical_indices(df["NAICS"], naics_cuts, prefix_fill="0")
h_zip   = build_hierarchical_indices(df["ZIP"],   zip_cuts,   prefix_fill="0")

# Shapes & keys
print(h_naics["code_levels"].shape)  # (N, len(naics_cuts))
print(h_naics["group_counts"])       # groups per NAICS level
print(list(h_naics["maps"][0].items())[:3])  # sample label→index at level 0

# 'code_levels' holds per-row, per-level integer indices into level-specific effect vectors
# Use these with base & delta vectors to compute eta.
```

---

### 2.2 `modeling`

- **`modeling/pymc_nested.py`**
  - The core PyMC model: intercept + NAICS base/deltas + ZIP base/deltas.
  - Implements **non-centered parameterization** for all group-level effects.
  - Includes **exponential depth shrinkage**: variance decays with level depth.
  - Exposes `eta` and `p` as deterministics for downstream evaluation.
- **`modeling/sampling.py`**
  - Thin wrappers around `pm.sample`, `pm.sample_prior_predictive`, etc.
  - Ensures consistent defaults (chains, draws, target_accept) and reproducible seeds.

**Implementation principle:** keep modeling logic clear and aligned with statistical goals, while delegating all validation and math to `core`.

---

### 2.3 `scoring`

- **`scoring/extract.py`**
  - Extracts posterior means of all group-level effects from `idata.posterior`.
  - Returns a structured dictionary:
    - `beta0: float`
    - `naics_base: pd.Series`
    - `naics_deltas: list[pd.Series]`
    - `zip_base: pd.Series`
    - `zip_deltas: list[pd.Series]`
- **`scoring/predict.py`**
  - Uses the extracted effect tables and training-time maps to compute probabilities on new data.
  - Enforces **strict per-level backoff**: a delta only applies if the exact label exists; otherwise it contributes 0.
  - Outputs `eta`, `p`, and diagnostic flags (`*_known`, `any_backoff`).

**Implementation principle:** deterministic and auditable. Scoring never depends on PyMC or sampling once effects are extracted.

---

### 2.4 `eval`

- **`eval/calibration.py`**
  - Computes calibration diagnostics: reliability tables, Expected Calibration Error (ECE), Brier score, log-loss.
- **`eval/ranking.py`**
  - Computes ranking metrics for sales prioritization: precision@k%, lift@k%, cumulative gains.
- **`eval/temporal.py`**
  - (Optional) Provides time-based train/validation splits and drift monitoring.

**Implementation principle:** lightweight, NumPy/Pandas only, enabling fast iteration and reproducibility.

---

### 2.5 `io`

- **`io/artifacts.py`**
  - Save/load model artifacts (posterior means, maps, cut points, meta info).
  - Ensures reproducibility between training and scoring environments.
- **`io/datasets.py`**
  - (Optional) Simple dataset ingestion and transformation helpers (CSV, Parquet).

**Implementation principle:** separate file I/O from modeling to keep the scientific core pure.

---

### 2.6 `cli`

- **`cli/fit.py`**
  - End-to-end workflow: build indices → fit model → save artifacts.
- **`cli/score.py`**
  - Load artifacts → score new dataset → write results.
- **`cli/report.py`**
  - Run calibration and ranking reports → generate outputs for stakeholders.

**Implementation principle:** scripts should orchestrate functions, not contain modeling logic. Keep them thin, using `core`, `modeling`, `scoring`, and `eval`.

---

### 2.7 `utils`

- **`utils/_math.py`**
  - Graph-safe wrappers for sigmoid, exp, and other math functions.
  - Dispatches to `pm.math` inside PyMC models, NumPy otherwise.
- **`utils/misc.py`**
  - (Optional) Helpers for reproducibility, seeding, logging.

**Implementation principle:** centralize small helpers to avoid duplication across modules.


## 3. Technical Specification of the Model

This section describes in detail the **statistical model**, its structure, and why this approach is appropriate for predicting conversion probabilities in an actuarial and sales triage context. The concepts—hierarchical modeling, shrinkage, calibration—are familiar to actuaries and statisticians, but the specific way NAICS and ZIP hierarchies are combined here is novel.


## Notation

**Indices:**

- $i = 1, \ldots, N$: index over rows (quotes).
- $j = 0, \ldots, J-1$: index over NAICS hierarchy levels (e.g., $J=4$ for [2,3,4,6] cuts).
- $m = 0, \ldots, M-1$: index over ZIP hierarchy levels (e.g., $M=5$ for [2,3,4,5] cuts).

**Variables:**

- $y_i$: binary response for quote $i$ (converted or not).
- $X_i$: feature vector for quote $i$ (including NAICS and ZIP codes).
- $\eta_i$: linear predictor for quote $i$ (log-odds of conversion).
- $p_i$: predicted probability of conversion for quote $i$.
- $\beta_0$: global intercept.	
- $b^{(n)}$: NAICS base effect vector (level 0).
---

### 3.1 Data & Hierarchies

- **Outcome**:  
  Binary response $ y_i \in \{0,1\} $ for each quote $ i $, where 
  
$$
	y_i = \left\{ 
		\begin{align} 
			1 & \text{ if quote } i \text{ converted to written policy } \\  
			0 & \text{ otherwise}
		\end{align} 
	\right.
$$

- **Predictors**:  
  Two hierarchical categorical variables:
  - **NAICS codes**: structured industry classification. We cut them at multiple prefix lengths (e.g., [2,3,4,6]), representing coarse-to-fine granularity.
  - **ZIP codes**: geographic location. Cut at digit prefixes (e.g., [2,3,5]).

- **Hierarchical Index Construction**:  
  - Codes are **right-padded** with a fill character (default `'0'`) to the maximum cut point length.  
  - At each cut length, we take a **prefix** of the padded code to form a level label.  
  - For example, NAICS `311000` with cuts [2,3,6] → `31` (L2), `311` (L3), `311000` (L6).

- **Training artifacts**:  
  - Unique labels per level,
  - Maps from label → index,
  - Group counts,
  - Parent pointers (L3 label has an L2 parent, etc.).

---

### 3.2 Linear Predictor

Let each hierarchy $H \in \{n,z\}$ (NAICS, ZIP) have $L_H$ levels from coarse to fine, with **0-based** indices $j=0,\dots,L_H-1$. For row $i$, let $\ell^{(H)}_{i,j} \in \{0,\dots,K^{(H)}_j-1\}$ be the integer index of the level-$j$ label using the training maps. The log-odds of conversion are

$$
\begin{align*}
\eta_i
=& \text{ } \beta_0 &\text{ (global intercept) } \\
&+ b^{(n)}_{\ell^{(n)}_{i,0}}
+ \sum_{j=1}^{L_n-1} \delta^{(n)}_{j,\ell^{(n)}_{i,j}} &\text{ (NAICS effects) } \\
&+ b^{(z)}_{\ell^{(z)}_{i,0}}
+ \sum_{m=1}^{L_z-1} \delta^{(z)}_{m,\ell^{(z)}_{i,m}} &\text{ (ZIP effects) }
\end{align*}
$$

We define $K^{(H)}_j$ as the number of unique labels at level $j$ of hierarchy $H$. The model components are:

- $\beta_0 \in \mathbb{R}$: global intercept.
- $b^{(H)} \in \mathbb{R}^{K^{(H)}_0}$: **base effects** at the coarsest level ($j=0$).
- $\delta^{(H)}_{j} \in \mathbb{R}^{K^{(H)}_j}$: **delta effects** at deeper level $j\ge 1$, i.e., deviations from the base.
- $p_i = \sigma(\eta_i)$, where $\sigma$ is the logistic sigmoid.

**Interpretation.** The base encodes broad average effects (e.g., 2-digit NAICS, coarse ZIP). Deltas add fine-grained deviations **only where supported by data**. During scoring, a level’s contribution is **0** if that exact label was unseen in training (strict per-level backoff; no parent substitution).

```python
# Pure NumPy illustration: assemble eta from indices + effect vectors (no PyMC)
import numpy as np

# Suppose you have:
# - indices: naics_levels (N, L_n), zip_levels (N, L_z)
# - effects: beta0 (float), naics_base (K_n0,), naics_deltas[j] (K_nj,)
#            zip_base (K_z0,),   zip_deltas[m]   (K_zm,)

def logistic(x):  # numerically stable
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out

def eta_from_indices(beta0, naics_levels, zip_levels, naics_base, naics_deltas, zip_base, zip_deltas):
    N = naics_levels.shape[0]
    eta = np.full(N, beta0, dtype=float)
    # NAICS base + deltas
    eta += naics_base[naics_levels[:, 0]]
    for j in range(1, naics_levels.shape[1]):
        eta += naics_deltas[j-1][naics_levels[:, j]]
    # ZIP base + deltas
    eta += zip_base[zip_levels[:, 0]]
    for m in range(1, zip_levels.shape[1]):
        eta += zip_deltas[m-1][zip_levels[:, m]]
    return eta

# Then p = logistic(eta).
```

---
### 3.3 Priors & Shrinkage (non-centered)

All priors are on the log-odds scale and **non-centered** for sampling stability. Let hierarchies be $H \in \{n,z\}$ (NAICS, ZIP), with $L_H$ levels and $K^{(H)}_j$ groups at level $j$.

**Intercept**  
The intercept is **parameterized as an offset from the historical baseline write rate**, with only a weak, zero-centered residual intercept learned during fitting. Specifically:

1. **Compute the baseline write rate $\hat r$** from a training window (e.g., the last 90 days or other appropriate period).
2. **Define the offset** as the logit of this baseline:
   $$
   \text{offset} = \operatorname{logit}(\hat r)
   $$
3. **Place a weak, zero-centered prior on the residual intercept**:
   $$
   \beta_{0,\text{resid}} \sim \mathcal{N}(0,\,0.5)
   $$
4. **The model intercept is then:**
   $$
   \beta_0 = \text{offset} + \beta_{0,\text{resid}}
   $$

This approach ensures that the model is anchored at a data-informed baseline, and only learns deviations if the data warrant it, improving stability and interpretability.

```python
# PyMC: Intercept = offset(logit(r_hat)) + weak residual
import numpy as np
import pymc as pm
from pymc import math as pmmath

r_hat = 0.18  # example from a training window (compute outside the model)
offset = np.log(r_hat) - np.log1p(-r_hat)

with pm.Model() as model:
    beta0_resid = pm.Normal("beta0_resid", mu=0.0, sigma=0.5)
    beta0 = pm.Deterministic("beta0", offset + beta0_resid)
    # ... add base/delta effects, eta, p, and likelihood ...
```

**Base effects (level 0, per hierarchy $H$)**  
For the coarsest level ($j=0$) with $K^{(H)}_0$ groups:
$$
u^{(H)}_0 \sim \mathcal{N}\!\big(0,\,I_{K^{(H)}_0}\big),\qquad
\sigma^{(H)}_0 \sim \text{HalfNormal}(1),\qquad
b^{(H)} \;=\; \sigma^{(H)}_0 \, u^{(H)}_0 \;\in\; \mathbb{R}^{K^{(H)}_0}.
$$

*Optional heavy tails:* replace $u^{(H)}_0 \sim \mathcal{N}(0, I)$ with $u^{(H)}_0 \sim \text{StudentT}(\nu, 0, 1)$ (e.g., $\nu \in [3,5]$) if a few large group effects are plausible.

**Delta effects (levels \(j \ge 1\))**  
For each deeper level $j = 1,\dots,L_H-1$ with $K^{(H)}_j$ groups:
$$
\kappa^{(H)} \sim \text{HalfNormal}(1),\qquad
\sigma^{(H)}_j \;=\; \sigma^{(H)}_0 \,\exp\!\big(-\kappa^{(H)}\, j\big),
$$
$$
u^{(H)}_j \sim \mathcal{N}\!\big(0,\,I_{K^{(H)}_j}\big),\qquad
\delta^{(H)}_j \;=\; \sigma^{(H)}_j \, u^{(H)}_j \;\in\; \mathbb{R}^{K^{(H)}_j}.
$$

This enforces **exponentially stronger shrinkage** at deeper levels ($j\uparrow \Rightarrow \sigma^{(H)}_j \downarrow$), preventing overfitting of sparse groups while allowing meaningful deviations when warranted by data.

**Likelihood and link**
$$
p_i \;=\; \sigma(\eta_i), \qquad y_i \sim \text{Bernoulli}(p_i),
$$
where $\sigma(\cdot)$ is the logistic sigmoid.

---

### 3.4 Likelihood & Inference

- **Link function**:
  $$p_i = \sigma(\eta_i) \text{ where } \sigma \text{ is the logistic sigmoid.}$$

- **Likelihood**:  
  $$ y_i \sim \text{Bernoulli}(p_i)$$

- **Inference**:  
  NUTS/HMC sampling with PyMC.  
  - Non-centered parameterization improves mixing.  
  - Diagnostics: R-hat, divergences, energy plots.

---

### 3.5 Extraction → Scoring (Backoff Rules)

- **Extraction**: Posterior means of $\beta_0$, all base vectors, and all deltas.  
- **Scoring**: New codes are padded and sliced the same way. For each row:
  - If base-level label exists: include its effect. Otherwise 0.  
  - For each delta: include only if that exact label was seen in training; otherwise 0.  

- **No parent substitution for unknown child deltas.**  
  This avoids leaking effects across siblings and ensures deviations only when supported by training data.  

- **Flags**:  
  - `*_known`: per-level booleans.  
  - `any_backoff`: overall indicator if any level was unknown.  
  These flags help production monitoring and triage decisions.

```python
# Strict per-level backoff: apply a delta only if that exact label exists
import numpy as np

def score_row(naics_code, zip_code, naics_cuts, zip_cuts, naics_maps, zip_maps, effects, prefix_fill="0"):
    def pad(code, L):
        return (code + prefix_fill * L)[:L]

    eta = effects["beta0"]
    # NAICS
    for j, cut in enumerate(naics_cuts):
        lbl = pad(naics_code, max(naics_cuts))[:cut]
        idx = naics_maps[j].get(lbl)
        if idx is None:
            continue  # strict backoff: contribute 0 at this level
        if j == 0:
            eta += effects["naics_base"].iloc[idx]
        else:
            eta += effects["naics_deltas"][j-1].iloc[idx]
    # ZIP
    for m, cut in enumerate(zip_cuts):
        lbl = pad(zip_code, max(zip_cuts))[:cut]
        idx = zip_maps[m].get(lbl)
        if idx is None:
            continue
        if m == 0:
            eta += effects["zip_base"].iloc[idx]
        else:
            eta += effects["zip_deltas"][m-1].iloc[idx]
    # logistic
    p = 1.0 / (1.0 + np.exp(-eta)) if eta >= 0 else np.exp(eta) / (1.0 + np.exp(eta))
    return eta, p
```

---

### 3.6 Calibration & Ranking KPIs

- **Calibration**:  
  - Reliability curves, Expected Calibration Error (ECE), Brier score, log-loss.  
  - Ensures predicted probabilities align with observed frequencies.

- **Ranking**:  
  - Precision@k%, Lift@k%, Cumulative gain.  
  - Demonstrates triage efficiency: are the top k% by predicted probability truly better than average?

---

### 3.7 Why This Approach (Actuarial Context)

Traditional actuarial models emphasize **loss ratios** or **pricing**. This project instead models **conversion likelihood**—the probability that a quoted policy is written—because:

- **Business value**: Sales/underwriting efficiency depends on knowing where to spend time, not just expected losses.  
- **High-cardinality covariates**: Both NAICS and ZIP have thousands of unique categories; treating them naively would overfit.  
- **Hierarchical pooling**: By structuring them into nested levels, we borrow strength from parent categories, yielding stable estimates for rare subgroups.  
- **Interpretability**: Decomposing effects into base + deltas matches human intuition (industry effect + finer subclass tweaks).  
- **Novelty**: This structured Bayesian approach is not standard in actuarial practice, but it aligns statistical rigor with real-world decision-making for new business acquisition.


## 4. Quickstart: Dataset → Fit → Evaluate → Score

This section gives a minimal, step-by-step guide for going from **raw data** to a **fit model** and then performing basic evaluation and inference. The emphasis is on practicality: enabling an analyst to stand up a working proof-of-concept quickly, while keeping enough detail that results can be explained to stakeholders.

---

### 4.1 Installation

Ensure you have Python 3.10+ and the required packages:

```bash
pip install pymc arviz numpy pandas xarray
pip install -e .   # install this project in editable mode if using pyproject/poetry
```
For testing and reproducibility:

```bash
pip install pytest pytest-cov
```

### 4.2 Minimal End-to-End Example

#### 1) Example dataset
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "is_written": [0,1,0,1,1,0,0,1,0,1],
    "NAICS": ["521110","522120","311000","521110","311100","522200","311200","521110","522120","311000"],
    "ZIP":   ["45242","45209","10001","45219","30301","60601","94107","45242","45209","10001"],
})
```

#### 2) Define hierarchy cut points and padding
```python
naics_cuts = [2,3,6]
zip_cuts   = [2,3,5]
prefix_fill = "0"
```

#### 3) Build hierarchical indices
```python
from hierarchical_naics_model.core.hierarchy import build_hierarchical_indices
h_naics = build_hierarchical_indices(df["NAICS"], naics_cuts, prefix_fill=prefix_fill)
h_zip   = build_hierarchical_indices(df["ZIP"],   zip_cuts,   prefix_fill=prefix_fill)

y = df["is_written"].to_numpy(dtype=np.int8)
naics_levels = h_naics["code_levels"]
zip_levels   = h_zip["code_levels"]
```

#### 4) Fit the PyMC model
```python
from hierarchical_naics_model.modeling.pymc_nested import build_conversion_model_nested_deltas
model = build_conversion_model_nested_deltas(
    y=y,
    naics_levels=naics_levels, zip_levels=zip_levels,
    naics_group_counts=h_naics["group_counts"], zip_group_counts=h_zip["group_counts"],
    target_accept=0.9
)
```
### 4.3 Extract Effects and Score New Data

#### 5) Sample from the posterior
```python
import pymc as pm
with model:
    idata = pm.sample(draws=500, tune=500, chains=2, cores=1, target_accept=0.9, random_seed=42)
```

#### 5) Extract posterior mean effects
```python
from hierarchical_naics_model.scoring.extract import extract_effect_tables_nested
effects = extract_effect_tables_nested(idata)
```

#### 6) Score new data
```python
from hierarchical_naics_model.scoring.predict import predict_proba_nested
scored = predict_proba_nested(
    df_new=df,
    naics_col="NAICS", zip_col="ZIP",
    naics_cut_points=naics_cuts, zip_cut_points=zip_cuts,
    naics_level_maps=h_naics["maps"], zip_level_maps=h_zip["maps"],
    effects=effects,
    prefix_fill=prefix_fill,
    return_components=True,
)

print(scored[["NAICS","ZIP","p","any_backoff"]].head())
```

### 4.4 Evaluate Calibration and Ranking

#### 7) Calibration diagnostics
```python
from hierarchical_naics_model.eval.calibration import calibration_report
cal = calibration_report(df["is_written"].to_numpy(), scored["p"].to_numpy(), bins=10)
```

#### 8) Ranking diagnostics
```python
from hierarchical_naics_model.eval.ranking import ranking_report
rk = ranking_report(df["is_written"].to_numpy(), scored["p"].to_numpy(), ks=(5,10,20,50))

print("ECE:", cal["ece"], "Brier:", cal["brier"], "LogLoss:", cal["log_loss"])
print(rk["summary"])
```
### 4.5 What You Get

- **Posterior means** for intercept, NAICS/ZIP base effects, and deltas.
- **Probabilities** $p$ for each row, with `*_known` flags and `any_backoff` indicating whether backoff occurred.
- **Calibration metrics** (ECE, Brier, log-loss) for probability quality.
- **Ranking metrics** (Precision@k%, Lift@k%) for sales efficiency.

This completes the pipeline from  
>**raw codes  
→ hierarchical features  
→ Bayesian fit  
→ probability scores  
→ business evaluation**.

## 5. Design Choices, Pitfalls, and Troubleshooting

This section explains the **design tradeoffs**, potential **failure modes**, and practical **troubleshooting tips** for running and maintaining the hierarchical NAICS/ZIP Bayesian conversion model in production.

---

### 5.1 Design Choices

1. **Strict per-level backoff**
   - **Decision:** If a label is unseen at a given level, its contribution is set to **0**, rather than substituting from the parent.
   - **Rationale:** Prevents **sibling leakage** (where one child inherits another’s delta). Keeps contributions interpretable: *only deviations supported by data are applied*.

2. **Prefix padding**
   - All codes are right-padded to maximum cut length with a consistent `prefix_fill` (default `'0'`).
   - Ensures consistent slicing across NAICS/ZIP families.
   - **Important:** the same `prefix_fill` must be used in **training** and **scoring**.

3. **Non-centered parameterization**
   - All group-level effects use non-centered priors.
   - **Rationale:** improves mixing and reduces divergences in HMC, especially with hierarchical shrinkage.
   - This is a standard best practice in Bayesian hierarchical modeling for computational stability, but does not affect the model’s substantive interpretation.

4. **Exponential shrinkage with depth**
   - Variance at deeper levels decreases exponentially: $\sigma_d = \sigma_{base}\exp(-\kappa d)$.
   - Allows deeper refinements but discourages large deviations unless strongly supported.
   - The decay rate $\kappa$ is itself learned.

5. **Evaluation metrics**
   - Calibration (ECE, Brier, log-loss) and ranking (Precision@k%, Lift@k%) chosen because they directly measure **probability quality** and **triage efficiency**.

6. **Data-informed intercept offset**
   - **Decision:** The intercept is anchored at a baseline logit corresponding to the historical write rate (e.g., last 90 days), with only a weak, zero-centered residual term learned during fitting.
   - **Rationale:** Prioritizes operational stability and interpretability by ensuring the model starts at a realistic baseline and only learns deviations if strongly supported by the data.
   - Calibration (ECE, Brier, log-loss) and ranking (Precision@k%, Lift@k%) chosen because they directly measure **probability quality** and **triage efficiency**.

---

### 5.2 Pitfalls and Risks

1. **Prefix fill mismatches**
   - If training used `'0'` but scoring uses `'X'`, label maps will break silently (all labels look “unknown”).
   - **Mitigation**: store prefix fill in artifacts and enforce consistency.

2. **Sparse groups**
   - Even with shrinkage, extremely sparse groups may be indistinguishable from noise.
   - **Mitigation**: monitor `any_backoff` rate and retrain when new codes accumulate.

3. **Graph math**
   - Using NumPy (`np.exp`, `np.log`) instead of `pm.math` inside PyMC models breaks symbolic graphs.
   - **Mitigation**: abstract math functions in `utils/_math.py` and always use those to ensure immediate failure and clear errors if using the wrong backend.

4. **Sampling speed**
   - Full NUTS on millions of rows can be expensive.
   - **Mitigation**:
     - Subsample for proof-of-concept.
     - Use variational inference or MAP for quick iterations.
     - Run HMC with reduced draws/chains for smoke tests.

5. **Evaluation leakage**
   - Using the same data for fitting and calibration leads to optimistic metrics.
   - **Mitigation**: keep out-of-sample validation sets, or use temporal splits.

---

### 5.3 Troubleshooting Checklist

- **Divergences in PyMC sampling**
  - Increase `target_accept` (0.9 → 0.95).
  - Check for poorly scaled priors.
  - Verify non-centered parameterization is applied.

- **Unstable predictions (NaNs, infs)**
  - Ensure logistic (`sigmoid`) is implemented in a **numerically stable** way.
  - Clip probabilities in log-loss calculations (e.g., eps=1e-15).

- **Unexpected backoff behavior**
  - Inspect `*_known` flags and `any_backoff`.
  - If too many unknowns, retrain maps on updated data.

- **Calibration curve flat/biased**
  - Consider widening priors or allowing heavier tails.
  - Check for data imbalance; stratified resampling may help.

- **Slow CI tests**
  - Mark MCMC tests as `slow`.
  - Use prior predictive checks only for smoke testing.
  - Restrict to 1–2 chains with small draws.

---

### 5.4 Production Monitoring

Once in production:
- Track distribution of predicted probabilities (`p`).
- Monitor proportion of rows with `any_backoff=True`.
- Track calibration metrics over time.
- Alert if backoff rates or calibration drift beyond thresholds.

```python
# Example: monitor backoff rate & calibration over time
import pandas as pd
from hierarchical_naics_model.eval.calibration import calibration_report

# 'scored' contains columns: p, any_backoff, y_true, date
daily = (
    scored
    .groupby(pd.to_datetime(scored["date"]).dt.to_period("D"))
    .apply(lambda g: pd.Series({
        "backoff_rate": g["any_backoff"].mean(),
        "n": len(g),
        "ece": calibration_report(g["y_true"].to_numpy(), g["p"].to_numpy(), bins=10)["ece"],
    }))
    .reset_index()
    .rename(columns={"date": "day"})
)

print(daily.tail())
# Alert if backoff_rate spikes or ece exceeds threshold.
```
---

**Summary:**  
The design prioritizes interpretability, stability, and auditability, with explicit rules for backoff and shrinkage. The main pitfalls involve inconsistent preprocessing, sparse categories, and sampling stability—but all have clear mitigations. With monitoring and careful validation, the model can provide reliable triage probabilities and actionable business insight.