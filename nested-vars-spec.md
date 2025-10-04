# Nested NAICS Model — Technical Specification

## 0) Purpose & Scope

Goal: Predict the probability that a quote converts to a written policy using only two high-cardinality hierarchical predictors: NAICS and ZIP Code.
Core modeling idea: encode each hierarchy with base + nested deltas (child deviations from parent) and partial pooling so we learn specific effects where data support them, while shrinking sparse nodes toward their parents.

Deliverables:
	1.	Hierarchy processing utilities (prefix padding, level indices, parent pointers).
	2.	Bayesian logistic model with nested deltas for NAICS and ZIP.
	3.	Production scoring with parent backoff for unseen codes.
	4.	Sales-facing calibration & lift report and a temporal backtest.
	5.	Tests (unit, statistical, e2e) + monitoring/ops checklist.

⸻

## 1) Data & Contracts

### 1.1 Input tables

#### Quotes table (train):
	•	submission_id: string/UUID (unique)
	•	is_written: int {0,1}
	•	naics: string (canonical 2–6 digits; may be shorter for some rows)
	•	zip: string (canonical 5 digits; may be shorter/dirty)
	•	quote_date: date/time (UTC or local, specify; used for temporal split)

#### Scoring table (inference):
	•	submission_id
	•	naics
	•	zip
	•	optional quote_date (logged for monitoring; not used to score)

### 1.2 Data normalization
	•	Treat naics and zip as strings at all times (never ints).
	•	Prefix padding default '0':
	•	NAICS: right-pad to length 6 (e.g., "52" → "520000").
	•	ZIP: right-pad to length 5 (e.g., "0213" → "02130").
	•	Validate characters against ^[0-9]+$ after stripping spaces/hyphens; rows failing validation receive null code and are handled by backoff (→ contributes 0 at all levels).
	•	Persist a data_quality_summary with counts of invalid codes, short codes, and padding applied.

⸻

## 2) Hierarchy Representation

### 2.1 Level definitions (cut points)
	•	NAICS: cut_points_naics = [2, 3, 4, 5, 6] (L0..L4)
	•	ZIP:   cut_points_zip   = [2, 3, 5]       (L0..L2)

Rationale: L0 is most general (e.g., NAICS-2), deeper levels add specificity.

### 2.2 Index extraction (per hierarchy)

Function: build_hierarchical_indices(codes: list[str], cut_points: list[int], prefix_fill="0")

#### Outputs:
	•	levels: list[str] — ["L2","L3",...] (names for logs only)
	•	code_levels: (N, L) int ndarray — for obs i and level j, the group index (0..Kj−1)
	•	unique_per_level: list[np.ndarray of str] — labels in index order
	•	maps: list[dict[str,int]] — label → index per level
	•	group_counts: list[int] — Kj per level
	•	parent_index_per_level: list[np.ndarray | None] — for j>0, an array of length Kj s.t. parent_index_per_level[j][g_child] = idx_parent_at_(j-1)
	•	max_len: int — applied padding length (6 for NAICS; 5 for ZIP)
	•	cut_points: list[int] — echo

#### Algorithm:
	1.	Right-pad all codes to max(cut_points) with prefix_fill="0".
	2.	For each level j, slice first cut_points[j] chars → labels.
	3.	Enumerate unique labels in stable order → maps[j] and unique_per_level[j].
	4.	Map obs labels to int indices → code_levels[:, j].
	5.	For j>0, compute parent_index_per_level[j] by slicing child label to the parent cut and looking up the parent index.

#### Edge cases:
	•	Codes that are empty/invalid after cleaning → treat as missing (not included in maps at any level). During scoring these rows have None indexes → contribute 0 (global intercept only).

⸻

## 3) Model Formulation (Bayesian Logistic with Nested Deltas)

### 3.1 Notation
	•	Observations i = 1..N
	•	Outcome y_i ∈ {0,1}
	•	Logit link: logit(p_i) = η_i
	•	Hierarchies:
	•	NAICS levels j = 0..J-1 (0 is most general, e.g., 2-digit)
	•	ZIP levels m = 0..M-1 (0 is most general, e.g., 2-digit)
	•	idx_naics[i,j] and idx_zip[i,m] are int group indices (0..K−1) or None if missing/unseen at scoring time.

### 3.2 Linear predictor

Nested-delta decomposition:

η_i = β0
      +   NAICS_base[ idx_naics[i,0] ]       # level-0 base
      + Σ_{j=1}^{J-1} NAICS_delta_j[ idx_naics[i,j] ]    # zero-mean deltas
      +   ZIP_base  [ idx_zip  [i,0] ]
      + Σ_{m=1}^{M-1} ZIP_delta_m[ idx_zip  [i,m] ]

	•	If idx_naics[i,0] is None → contribute 0 at that term; similarly for deltas.
	•	Same for ZIP.

### 3.3 Priors (weakly informative, depth-aware)
	•	Global intercept: β0 ~ Normal(0, 1.5)
	•	NAICS level-0 base:
	•	Mean: μ_naics_0 ~ Normal(0, 1) (or StudentT(ν=4, 0, 0.5))
	•	Scale: σ_naics_0 ~ HalfNormal(0.6)
	•	Base effects (non-centered): NAICS_base = μ_naics_0 + z * σ_naics_0 with z ~ Normal(0,1) (vector of length K0)
	•	NAICS deeper levels (deltas):
	•	Zero-mean by construction (center on 0)
	•	Scales shrink with depth, e.g.,
	•	σ_naics_1 ~ HalfNormal(0.4)
	•	σ_naics_2 ~ HalfNormal(0.35)
	•	σ_naics_3 ~ HalfNormal(0.3)
	•	σ_naics_4 ~ HalfNormal(0.25)
	•	Deltas (non-centered): NAICS_delta_j = z_j * σ_naics_j, z_j ~ Normal(0,1) (vector of length Kj)
	•	ZIP priors analogous (ZIP_base with μ_zip_0, σ_zip_0; deltas σ_zip_m shrink with depth, e.g., 0.4 → 0.3)

Rationale: encouraging most signal at coarse levels; deeper levels express residual deviations; partial pooling controls overfit in sparse leaves.

### 3.4 Likelihood
	•	y_i ~ Bernoulli(p_i) with p_i = sigmoid(η_i)

### 3.5 Implementation notes (PyMC)
	•	Use non-centered parameterization at all random-effect vectors to avoid funnels.
	•	Register these variables (for audit & test expectations):
	•	naics_mu_0 (scalar), naics_sigma_0 (scalar), naics_base (vector K0)
	•	For j≥1: deterministic naics_mu_j = 0.0 and naics_sigma_j (scalar), naics_delta_j (vector Kj)
	•	Same pattern for ZIP
	•	Deterministics eta, p
	•	Sampling config: start with draws=1000, tune=1000, chains=2, target_accept=0.9–0.95.
	•	Consider ADVI/Pathfinder warm-starts in large K scenarios (optional).

⸻

## 4) Training Pipeline
	1.	Split: Temporal split by quote_date:
	•	Train on quotes with quote_date ≤ cutoff_date
	•	Validate on quotes with quote_date > cutoff_date
	2.	Preprocess: Clean strings, right-pad codes, build indices separately for NAICS and ZIP on training data only:
	•	Save maps per level for both hierarchies (JSON).
	3.	Model Build: Construct model with index arrays for the training set.
	4.	Sampling: Run MCMC; ensure diagnostics:
	•	R-hat ≤ 1.01 for all monitored variables
	•	No or few divergences; if many, raise target_accept and/or tighten deeper σ
	5.	Posterior reduction for scoring: Compute posterior means for:
	•	β0, NAICS_base, NAICS_delta_j (forall j), ZIP_base, ZIP_delta_m (forall m)
	•	Save as effects.json or Parquet (compact arrays).

⸻

## 5) Scoring (Production)

### 5.1 Backoff resolution (per code)

Function: make_backoff_resolver(cut_points, level_maps, prefix_fill="0") -> (code: str) -> list[int | None]

#### Behavior:
For requested level j, if the exact label isn’t in maps[j], try parent levels j−1, j−2, …, 0. If no ancestor is known, return None at that level. Always right-pad input codes with prefix_fill="0" before slicing.

### 5.2 Probability computation

Function: predict_proba_nested(df, naics*, zip*, effects, prefix_fill="0", return_components=True) -> df_out
	•	Compute eta = β0 + naics_base[idx0] + Σ_j≥1 naics_delta_j[idxj] + zip_base[idx0] + Σ_m≥1 zip_delta_m[idxm] where any missing index contributes 0.
	•	Return:
	•	p = sigmoid(eta), eta
	•	Per-level boolean flags backoff_naics_j, backoff_zip_m (True iff missing/unseen at that exact level).
	•	Log backoff rates; high rates signal data coverage issues.

### 5.3 Artifacts for deployment
	•	maps per level for NAICS and ZIP (JSON with “label” → int)
	•	effects (posterior means) as arrays (Parquet/NPY):
beta0, naics_base, naics_deltas[], zip_base, zip_deltas[]
	•	Version metadata: model hash, fit date, train window, cut points, priors, sampler settings.

⸻

## 6) Evaluation (Sales-Facing)

### 6.1 Calibration
	•	Reliability curve (e.g., 10 bins by predicted probability)
	•	ECE (Expected Calibration Error):
ECE = Σ_bins (n_b/N) * | mean_p_b − mean_y_b |
	•	Brier score: mean squared error of probabilities

### 6.2 Ranking quality
	•	Precision@k% for k ∈ {5,10,20,30}, etc.
	•	Lift@k% = precision@k / global positive rate
	•	Cumulative gain = captured positives in top-k / total positives

### 6.3 Temporal backtest
	•	Report the above metrics on validation window (quotes > cutoff_date).
	•	Optionally evaluate capacity slices (e.g., “If we only review top 15% each day, what precision/lift do we get?”).

### 6.4 Reports
	•	Emit:
	•	metrics_summary.json (base rate, Brier, log-loss, ECE)
	•	reliability.csv (bin, count, mean_p, mean_y, gap)
	•	ranking.csv (k, k_count, precision@k, lift@k, cumulative_gain)

⸻

## 7) Explainability

### 7.1 Path contribution

Function: explain_row(naics_code, zip_code, effects, cut_points, maps, prefix_fill="0") -> dict
	•	Resolves:
	•	NAICS L0 base + each available delta
	•	ZIP L0 base + each available delta
	•	Returns a dict with:
	•	beta0, contributions by level, and total eta, p
	•	a path list like:
	•	NAICS: 51 (L0) +0.18, 511 (L1) +0.03, 511110 (L4) −0.02
	•	ZIP: 45 (L0) −0.05, 450 (L1) +0.01, 45040 (L2) +0.00

### 7.2 Segment summary tables
	•	For top NAICS-2 and ZIP-3 groups: show mean predicted p, conversion lift vs. global, counts.

⸻

## 8) Performance & Scale
	•	Parameter count ≈ Σ K_j (NAICS levels) + Σ K_m (ZIP) + 1 intercept.
	•	For large K (many leaf groups), consider:
	•	Tighter σ at deepest levels, and/or
	•	Minimum support thresholds to activate a level for a group; otherwise, rely on parent.
	•	Variational warm-start or Pathfinder → short NUTS.
	•	Use vectorized draws; avoid per-row Python loops in scoring (except for the short index resolution step).

⸻

## 9) Governance & Risk
	•	Usage restriction: triage/queue ordering only; not pricing/eligibility decisions.
	•	Fairness: periodically review calibration/lift by state & high-level NAICS (L0/L1); investigate systematic under-prioritization.
	•	Privacy: no PII beyond ZIP; document retention policies.
	•	Monitoring: track data drift (distribution of NAICS/ZIP labels), unseen/backoff rates, calibration drift, and precision@k over time.

⸻

## 10) Interfaces (suggested)

### 10.1 Python function signatures

#### hierarchy.py
```python
def build_hierarchical_indices(
    codes: list[str],
    *,
    cut_points: list[int],
    prefix_fill: str = "0",
) -> dict: ...
```

#### model.py
```python
def build_conversion_model_nested_deltas(
    *,
    y: np.ndarray,
    naics_levels: np.ndarray,
    zip_levels: np.ndarray,
    naics_group_counts: list[int],
    zip_group_counts: list[int],
    target_accept: float = 0.92,
    use_student_t_level0: bool = False,
) -> pm.Model: ...


class ConversionModelStrategy(Protocol):
    def build_model(
        self,
        *,
        y: np.ndarray,
        naics_levels: np.ndarray,
        zip_levels: np.ndarray,
        naics_group_counts: list[int],
        zip_group_counts: list[int],
    ) -> Any: ...

    def sample_posterior(
        self,
        model: Any,
        *,
        draws: int,
        tune: int,
        chains: int,
        cores: int,
        target_accept: float | None = None,
        progressbar: bool = False,
        random_seed: int | None = None,
    ) -> Any: ...


class PymcNestedDeltaStrategy(ConversionModelStrategy): ...
```

#### scoring.py

```python
def extract_effect_tables_nested(idata) -> dict: ...
```

```python
def make_backoff_resolver(
    *,
    cut_points: list[int],
    level_maps: list[dict[str, int]],
    prefix_fill: str = "0",
) -> callable: ...
```

```python
def predict_proba_nested(
    df_new: pl.DataFrame | pl.LazyFrame,
    *,
    naics_col: str,
    zip_col: str,
    naics_cut_points: list[int],
    zip_cut_points: list[int],
    naics_level_maps: list[dict[str, int]],
    zip_level_maps: list[dict[str, int]],
    effects: dict,
    prefix_fill: str = "0",
    return_components: bool = True,
) -> pl.DataFrame: ...
```

#### evaluation.py
```python
def calibration_and_lift_report(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    bins: int = 10,
    ks: list[int] = [5,10,20,30],
) -> dict: ...
```

#### explain.py
```python
def explain_row(
    naics_code: str,
    zip_code: str,
    *,
    naics_cut_points: list[int],
    zip_cut_points: list[int],
    naics_level_maps: list[dict[str,int]],
    zip_level_maps: list[dict[str,int]],
    effects: dict,
    prefix_fill: str = "0",
) -> dict: ...
```

### 10.2 File artifacts
	•	maps_naics.json, maps_zip.json — list of per-level {"label": "51", "index": 0} arrays or dicts.
	•	effects.parquet — columns: beta0 (single row), naics_base (array), naics_delta_1 (array), …, zip_base, zip_delta_1, …
	•	metrics_summary.json, reliability.csv, ranking.csv
	•	scored.parquet — columns: submission_id, p, eta, per-level backoff flags.

⸻

## 11) Testing Plan

### 11.1 Unit tests

#### Hierarchy:
- Padding correctness: "52" with NAICS [2,3,4,5,6] must not appear as an L3 label; L0 parent mapping correct.
- Parent pointers: each L1/L2/… group has a valid parent index at the previous level.

#### Backoff:
- Unseen leaf → backoff to parent; unseen parent chain → contribute 0.
- Backoff flags present and boolean.

#### Scoring invariants:
	•	p ∈ (0,1) finite; monotonic in eta (e.g., larger eta rows have p ≥ smaller eta rows).

### 11.2 Statistical tests (on synthetic)
	•	Parameter recovery (rough): simulate known base/deltas; fit; assert posterior means correlate with truth (e.g., Spearman ≥ 0.6 for sufficiently supported groups).
	•	Shrinkage: groups with low counts should have smaller |effect| than high-count groups on average.

### 11.3 Calibration tests
	•	Simulate perfectly calibrated probabilities; ensure ECE < 0.02, Brier ≈ expected.
	•	Reliability bins monotone w.r.t. mid-bin probabilities.

### 11.4 Temporal backtest
	•	Given a cutoff date, ensure the pipeline trains on ≤ cutoff and scores on > cutoff; assert artifacts exist and metrics are computed.

### 11.5 CLI/e2e tests
	•	Smoke test that runs build → sample (short draws) → score → metrics; asserts required outputs exist and schemas match.

⸻

## 12) Acceptance Criteria
	1.	Functionality
	•	Model trains without critical divergences; R-hat ≤ 1.01 for random-effect vectors.
	•	Scoring handles unseen codes with parent backoff; no crashes.
	2.	Calibration & Lift
	•	On validation window: ECE ≤ agreed threshold (e.g., ≤ 0.05), and lift@top10% ≥ baseline × 1.5 (tune to business reality).
	3.	Explainability
	•	explain_row shows a clean NAICS/ZIP path with per-level contributions summing to eta.
	4.	Ops
	•	Artifacts (maps/effects/metrics/scored) are versioned and reproducible.
	•	Logging includes backoff rates and data quality counts.

⸻

13) Implementation Guidance & Patterns
	•	SOLID: keep hierarchy extraction, model build, inference, scoring, and evaluation decoupled; code to interfaces above.
	•	Vectorization: do all per-row math with arrays; use Python loops only when resolving indices once per row.
	•	Numerical stability: clip probs in log-loss; use float64 for eta if overflow observed.
	•	Performance: for very large K:
	•	restrict deepest levels to groups with min_count ≥ m; otherwise rely on parent,
	•	or increase shrinkage (smaller σ at depth).

⸻

14) Examples

14.1 Example NAICS padding & slices
	•	Raw NAICS "52" → padded "520000"
	•	L0 (2-digit) "52"
	•	L1 (3-digit) "520"
	•	L2 (4-digit) "5200"
	•	…
	•	Raw ZIP "0213" → padded "02130"
	•	L0 (2-digit) "02"
	•	L1 (3-digit) "021"
	•	L2 (5-digit) "02130"

14.2 Example η breakdown for a row

beta0 = -1.10
NAICS_base["52"] = +0.18
NAICS_delta_1["521"] = +0.02
NAICS_delta_2["5211"] = +0.00  (unseen → backoff; contributes 0)
NAICS_delta_3["52111"] = +0.00 (unseen → 0)
ZIP_base["45"] = -0.05
ZIP_delta_1["450"] = +0.02
ZIP_delta_2["45040"] = +0.01
η = -1.10 + 0.18 + 0.02 + 0 + 0 + (-0.05) + 0.02 + 0.01 = -0.92
p = sigmoid(-0.92) ≈ 0.285

Backoff flags: backoff_naics_2=True, backoff_naics_3=True, others False.

⸻

15) Risks & Mitigations
	•	Sparse leaves → high variance: mitigated by depth-shrinking priors and parent backoff.
	•	Data shifts (new industries/regions): monitor unseen rates, recalibrate regularly; consider monthly refresh.
	•	Trust: provide calibration/lift and path explanations; make backoff rates transparent.
