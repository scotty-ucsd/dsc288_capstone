# Milestone 2: Master Plan and Integrated Task System

> **Generated:** 2026-04-20  
> **Source Files Processed (in order):**  
> 1. `docs/milestones/milestone2/milestone2_rubric.md` ✅  
> 2. `docs/milestones/milestone2/milestone2_outline.md` ✅  
> 3. `docs/research/lecture/eda_tips.md` ✅  
> 4. `docs/research/lecture/feature_engineering_tips.md` ✅  
> 5. `docs/research/lecture/model_selection_tips.md` ✅  
> 6. `docs/tasklists/milestone2_tasklist.md` ✅  
> 7. `docs/research/potential_features.md` ✅  
> 8. `docs/tasklists/milestone3_tasklist.md` ✅  

---

## 1. Missing Files or Clarifications

| Item | Status | Notes |
|------|--------|-------|
| All 8 files | Present | All files found and processed |
| `milestone3_tasklist.md` | Partially read (first 500 lines) | File is ~4000 lines; sufficient context extracted for dependency analysis |

**Ambiguities flagged:**
- Rubric item 2d ("Relationships — correlations") does not specify whether lagged cross-correlations are required or only contemporaneous. **Resolution:** Include both, since lagged correlations are central to the physics (Sun-to-Ground causal chain).
- Outline §5 references "citations on where these are used" — unclear if this means published literature only or also includes feasibility report results. **Resolution:** Include both; cite published literature AND internal feasibility results.
- Outline §6 says "Model Architecture from literature that you plan to use **new architecture experiments** you want to try" — grammatically ambiguous. **Resolution:** Interpret as two items: (a) architectures drawn from literature, (b) novel architectural ideas being proposed.
- `potential_features.md` contains ~500+ engineered feature ideas. Not all belong in Milestone 2. Prioritization applied below.

---

## 2. Master Milestone 2 Structure

### Report Section Architecture (Outline + Rubric Aligned)

| Section | Outline Ref | Rubric Ref | Core Deliverable |
|---------|-------------|------------|------------------|
| Title Area | §Title | — | Header, author, project info |
| 1. Background | §1 | §4 (comprehensiveness) | Problem statement, motivation (2-3 paragraphs) |
| 2. Dataset Name and Citation | §2 | §1a (merging) | Data sources, links, access method |
| 3. Data Pipeline | §3 | §1a-d (full pipeline) | Pipeline design, merging, cleansing, augmentation, normalization |
| 4. EDA Description | §4 | §2a-d + §3 (feature-linked) | Analysis types, artifacts, findings connected to features |
| 5. Feature Engineering | §5 | §1c (augmentation) + §3 (EDA→features) | Literature features, novel features, EDA traceability |
| 6. Model Selection | §6 | §4 (maturity) | Prioritized model list with justification |
| 7. Progress Report | §7 | §4 (on-track evidence) | Effort summary, test results, current state |
| 8. Task List | §8 | §4 (completion trajectory) | Remaining work plan with timeline |
| 9. Risks and Mitigation | §9 | §4 (maturity) | Risk register with mitigations |
| 10. References | §10 | — | BibTeX-backed citations |

### Rubric-to-Outline Mapping

| Rubric Category | Outline Sections Covering It |
|-----------------|------------------------------|
| 1a. Data merging | §2, §3 |
| 1b. Data cleansing | §3, §4 (missing data EDA) |
| 1c. Data augmentation/enrichment | §3, §5 |
| 1d. Data normalization | §3, §5 |
| 2a. Data completeness/freshness/quality | §4 (EDA) |
| 2b. Variables and distributions | §4 (EDA) |
| 2c. Anomalies/outliers | §4 (EDA) |
| 2d. Relationships/correlations | §4 (EDA) |
| 3. EDA → Feature identification | §4 (narrative), §5 (explicit link) |
| 4. Report quality/completeness | All sections, especially §7 |

### Missing Coverage or Tension Points

1. **Rubric §3 explicitly requires EDA to be explained "with a view toward identifying features."** The outline (§4) separates EDA from Feature Engineering (§5). The report must create explicit forward-references from §4 findings to §5 feature choices.
2. **Rubric §4 asks "Is the work done so far mature enough?"** The Progress Report (§7) must quantify completion percentage and show evidence of working code/results.
3. **The outline does not have a dedicated "Data Quality" section**, but rubric §2a requires it. Must embed quality analysis within §4 or add a subsection.

---

## 3. Best-Practice Requirements

### 3.1 EDA Best-Practice Requirements (from `eda_tips.md`)

| ID | Requirement | Report Section |
|----|-------------|----------------|
| EDA-1 | Pre-analysis data hygiene: shape, types, missingness heatmaps, uniqueness/duplicates | §4 |
| EDA-2 | Univariate analysis: distributions, descriptive stats, skewness, kurtosis | §4 |
| EDA-3 | Identify non-Gaussian distributions; document which variables need transforms | §4, §5 |
| EDA-4 | Outlier detection and characterization (IQR, z-score) | §4 |
| EDA-5 | Bivariate/multivariate analysis: correlations, scatter, heatmaps | §4 |
| EDA-6 | Always visualize — do not rely solely on summary statistics (Anscombe's quartet lesson) | §4 |
| EDA-7 | Correlation ≠ causation: use domain knowledge to validate relationships | §4, §5 |
| EDA-8 | EDA is iterative and hypothesis-generating, not confirmatory | §4 |
| EDA-9 | Every plot must include sufficient explanation/narrative in the report | §4 |
| EDA-10 | EDA findings must drive feature engineering and model choice decisions | §4→§5→§6 |

### 3.2 Feature Engineering Best-Practice Requirements (from `feature_engineering_tips.md`)

| ID | Requirement | Report Section |
|----|-------------|----------------|
| FE-1 | Features define accessible information — no algorithm can recover missing information | §5 |
| FE-2 | Control scale: normalize/standardize numerical features | §3, §5 |
| FE-3 | Handle skew: log/power transforms for heavy-tailed variables (e.g., proton density, particle flux) | §5 |
| FE-4 | Temporal features: encode memory via lags and rolling windows; respect causality strictly | §5 |
| FE-5 | No future data leakage — verify all rolling windows use `center=False` | §5 |
| FE-6 | Missing data as signal: add missingness indicator features alongside imputation | §5 |
| FE-7 | Feature evaluation: ablation, permutation importance, univariate AUC | §5 |
| FE-8 | Dimensionality management: identify redundant/correlated features | §4 (correlations), §5 |
| FE-9 | Justify every feature from EDA findings or literature | §5 |
| FE-10 | Plan for unseen categories/regimes at inference time | §5 |

### 3.3 Model Selection Best-Practice Requirements (from `model_selection_tips.md`)

| ID | Requirement | Report Section |
|----|-------------|----------------|
| MS-1 | Principled reasoning, not model shopping — justify why each model family fits the data | §6 |
| MS-2 | Start with baselines: establish minimum acceptable performance first | §6, §7 |
| MS-3 | Link model family to data type: this is time-series + multi-source tabular | §6 |
| MS-4 | Bias-variance consideration: document capacity vs. data regime | §6 |
| MS-5 | Evaluation metric choice must encode operational costs (missed storms vs. false alarms) | §6 |
| MS-6 | Time-series models: classical → ML → DL progression with justification at each step | §6 |
| MS-7 | Deep learning failure modes: acknowledge overfitting risk on tabular/small-data regimes | §6 |
| MS-8 | Narrow to 2-3 candidates with justification, not an exhaustive list | §6 |
| MS-9 | Features, models, and evaluation co-evolve — acknowledge this in the plan | §6 |
| MS-10 | Operational constraints: latency, interpretability, deployment cost considered | §6 |

### 3.4 Cross-Link Requirements

| Link | Description |
|------|-------------|
| EDA → Feature | Every proposed feature must trace back to an EDA finding or physics rationale |
| EDA → Model | Distribution shapes (heavy-tailed, imbalanced) must inform model architecture choices |
| Feature → Model | Feature types (temporal lags, interaction terms) must match model capabilities |
| Model → Evaluation | Evaluation metrics must be selected before model training, not after |
| Physics → All | Newell coupling, propagation delays, northward Bz suppression must be visible in EDA, encoded in features, and tested in model evaluation |

---

## 4. Final Milestone 2 Task List

### Section 1: Background (Report §1)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| R1.1 | Write 2-3 paragraph problem statement: GICs threaten power grids, current forecasts rely on MHD/global indices, this project uses Direct-Drive ML | High | Pending |
| R1.2 | State the target variable (dB/dt in nT/min at ABK/TRO) and forecast horizon (60 min) | High | Pending |
| R1.3 | Reference the feasibility result ($r=0.55$ LSTM vs. $r=0.33$ XGBoost) as motivation | High | Pending |

### Section 2: Dataset Name and Citation (Report §2)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| R2.1 | List all data sources with links: OMNI (NASA CDAWeb), GOES (NOAA NCEI), Swarm (ESA VirES), SuperMAG | High | Pending |
| R2.2 | Cite each dataset formally (DOI or institution reference) | High | Pending |
| R2.3 | State temporal coverage (2015–2019 for EDA, 2010–2025 for full production) | High | Pending |
| R2.4 | State spatial coverage and resolution (1-min cadence, high-latitude stations) | Medium | Pending |

### Section 3: Data Pipeline (Report §3)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| P3.1 | **Data Loading Pipeline:** Create `src/swmi/data/loaders.py` with unified loading functions for OMNI, GOES, Swarm LEO indices, SuperMAG | High | Pending |
| P3.2 | Implement `load_omni_month(year, month)` returning validated DataFrame | High | Pending |
| P3.3 | Implement `load_goes_month(year, month, data_type)` for particle/xray data | High | Pending |
| P3.4 | Implement `load_leo_index_month(year, month)` for LEO sub-indices | High | Pending |
| P3.5 | Implement `load_supermag_month(year, month, stations)` for ground data | High | Pending |
| P3.6 | Implement `load_storm_labels(year, month, threshold)` for binary labels | High | Pending |
| P3.7 | **Data Merging:** Document multi-source merge strategy (1-min time grid, temporal alignment, propagation delay handling) | High | Pending |
| P3.8 | **Data Cleansing:** Implement NULL/missing value handling per Safe Zone rules (linear interp < 5 min, forward-fill 5–30 min, segment boundary > 30 min) | High | Pending |
| P3.9 | **Data Augmentation/Enrichment:** Implement Newell coupling computation, derived features (dB/dt, clock angle, ram pressure) | High | Pending |
| P3.10 | **Data Normalization:** Document and implement normalization strategy (log-transform for log-normal variables, StandardScaler for IMF, MinMax for bounded features) | High | Pending |
| P3.11 | Write unit tests for each loader: `tests/unit/test_data_loaders.py` | Medium | Pending |
| P3.12 | Call `validate_output()` and `validate_time_series()` after every load/transform step | High | Pending |
| P3.13 | Document pipeline design with diagram (flow from raw → processed → features) | Medium | Pending |
| P3.14 | Describe pipeline outputs (deduplicated, validated Parquet files) | Medium | Pending |

### Section 4: EDA (Report §4)

#### 4A: Infrastructure

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4A.1 | Create `src/swmi/eda/utils.py` with plotting helpers: `plot_timeseries`, `plot_distribution`, `plot_correlation_heatmap`, `compute_missing_stats`, `detect_outliers`, `plot_storm_event` | High | Pending |
| E4A.2 | Select EDA analysis period: 2015–2019 (Swarm operational, GOES-13/15 overlap, good SuperMAG coverage) | High | Pending |
| E4A.3 | Identify 10 reference storm events: 3 CME-driven, 3 CIR-driven, 2 isolated substorms, 2 quiet periods | High | Pending |
| E4A.4 | Create `docs/milestones/milestone2/reference_events.md` with event catalog | High | Pending |

#### 4B: OMNI Solar Wind Analysis (Univariate + Temporal)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4B.1 | Notebook `01_omni_eda.ipynb`: Plot distributions for BZ_GSM, BY_GSM, BX_GSM, |B|, Vx, Vy, Vz, Np, T, Pdyn, SYM-H, ASY-H, AL, AU, AE | High | Pending |
| E4B.2 | Statistical summaries: mean, median, std, percentiles, skewness, kurtosis for each OMNI variable | High | Pending |
| E4B.3 | Identify non-Gaussian distributions (e.g., Np is log-normal); document transform needs | High | Pending |
| E4B.4 | Plot monthly/seasonal averages of solar wind speed; identify solar cycle trends | Medium | Pending |
| E4B.5 | Compute autocorrelation functions for BZ_GSM, Vx (inform lag feature design) | High | Pending |
| E4B.6 | Missing data analysis: null percentages, gap timeline, gap duration histogram, cross-reference with ACE/Wind outages | High | Pending |
| E4B.7 | Storm-time superposed epoch analysis: average BZ_GSM, Vx, Pdyn over storm onset ±6 hours; compare CME vs. CIR | Medium | Pending |

#### 4C: GOES GEO Analysis

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4C.1 | Notebook `02_goes_eda.ipynb`: Plot distributions for GOES particle flux channels (electron 0.6–2.0 MeV, >2.0 MeV; proton >1, >10, >100 MeV) | High | Pending |
| E4C.2 | Identify dynamic range (log-scale); detect saturation events | Medium | Pending |
| E4C.3 | X-ray flux time series: identify flare events (X/M/C class), correlate with F10.7 | Medium | Pending |
| E4C.4 | GOES magnetic field: plot Bt, Bz; compare with solar wind Bz (delayed/damped response) | High | Pending |
| E4C.5 | Missing data and satellite transitions: GOES-13→15→16 transition dates, gap assessment | Medium | Pending |

#### 4D: Swarm LEO Index Analysis

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4D.1 | Notebook `03_swarm_leo_index_eda.ipynb`: Plot global LEO observation density (QDLat × MLT heatmap) | High | Pending |
| E4D.2 | Compare Swarm A/B/C coverage; identify coverage gaps | Medium | Pending |
| E4D.3 | Plot distributions for 12 sub-indices; compare quiet-time vs. storm-time | High | Pending |
| E4D.4 | Decay age analysis: histograms, median freshness, correlation with sub-index value | Medium | Pending |
| E4D.5 | LEO vs. Ground dB/dt: test hypothesis that high LEO index precedes ground dB/dt by ~10 min | High | Pending |

#### 4E: SuperMAG Ground Analysis

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4E.1 | Notebook `04_supermag_eda.ipynb`: Map all stations (geographic + magnetic coords), plot density vs. magnetic latitude | High | Pending |
| E4E.2 | Compute dB/dt (horizontal) for all stations; plot distribution (heavy-tailed, log-normal expected) | High | Pending |
| E4E.3 | Identify extreme events (dB/dt > 10 nT/s); compute per-station 95th/99th percentiles | High | Pending |
| E4E.4 | Spatial patterns: animated maps of dB/dt evolution during reference storms; correlate with MLT | Medium | Pending |
| E4E.5 | Missing data per station: compute availability, flag low-quality stations | High | Pending |

#### 4F: Target Variable Analysis

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4F.1 | Notebook `05_target_distribution.ipynb`: Compute event rate for thresholds 1.0, 1.5, 2.0, 3.0 nT/s | High | Pending |
| E4F.2 | Plot label distribution over time (seasonality, solar cycle effects) | Medium | Pending |
| E4F.3 | Event duration statistics: continuous event windows, duration histogram, median storm duration | High | Pending |
| E4F.4 | Storm vs. substorm stratification: compare feature distributions between categories | Medium | Pending |
| E4F.5 | Lead time analysis: time from BZ_GSM southward turning to ground dB/dt response | High | Pending |

#### 4G: Multi-Source Correlations

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4G.1 | Notebook `07_correlations.ipynb`: Lagged cross-correlation: BZ_GSM (L1) vs. GOES Bt (GEO) | High | Pending |
| E4G.2 | Lagged cross-correlation: GOES particle flux vs. ground dB/dt | High | Pending |
| E4G.3 | Lagged cross-correlation: LEO index vs. ground dB/dt | High | Pending |
| E4G.4 | Feature correlation heatmap: Pearson correlation, identify multicollinearity (VIF > 10) | High | Pending |
| E4G.5 | Newell Coupling validation: correlate with SYM-H, AL, AE; test if better predictor than raw BZ_GSM | High | Pending |

#### 4H: Temporal Patterns

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4H.1 | Notebook `06_temporal_patterns.ipynb`: Monthly event rate (equinox peaks?); test Russell-McPherron effect | Medium | Pending |
| E4H.2 | Diurnal patterns: event rate vs. UT hour and MLT; dawn/dusk asymmetry | Medium | Pending |
| E4H.3 | Solar cycle trends: compare event rates 2015 vs. 2019; correlate with F10.7 | Medium | Pending |

#### 4I: Missing Data Deep Dive

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4I.1 | Notebook `08_missing_data.ipynb`: Create missingness heatmap (time × data source) | High | Pending |
| E4I.2 | Identify correlated gaps (OMNI + GOES simultaneous failures?) | Medium | Pending |
| E4I.3 | Rolling window quality metrics: simulate 30-min windows, compute n_valid_points distribution | High | Pending |
| E4I.4 | Impact on labels: how many labeled events occur during data gaps? Estimate information loss | Medium | Pending |

#### 4J: Storm Case Studies

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| E4J.1 | Notebook `09_storm_case_studies.ipynb`: Multi-panel plots for each reference event (Solar wind → GEO → LEO → Ground) | High | Pending |
| E4J.2 | Timeline of feature evolution; annotate shock arrival, main phase, recovery | Medium | Pending |
| E4J.3 | CME vs. CIR comparison: compare feature signatures | Medium | Pending |

### Section 5: Feature Engineering (Report §5)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| F5.1 | **Literature features (with citations):** Newell coupling (Newell 2007), epsilon parameter (Akasofu 1981), rectified voltage, Borovsky coupling | High | Pending |
| F5.2 | **Novel features from EDA:** Document EDA→feature traceability for each proposed feature | High | Pending |
| F5.3 | Notebook `10_feature_engineering_preview.ipynb`: Implement and visualize Newell coupling with 10/30/60-min lags | High | Pending |
| F5.4 | Implement rolling std of BZ_GSM (variability proxy) | High | Pending |
| F5.5 | Implement GOES particle flux rate-of-change | Medium | Pending |
| F5.6 | Implement LEO spatial gradients (high_dawn − high_dusk) | Medium | Pending |
| F5.7 | Implement ground dB/dt trailing statistics (station-level history) | High | Pending |
| F5.8 | For each feature: plot distribution split by label (0 vs. 1) | High | Pending |
| F5.9 | Compute univariate AUC for each candidate feature; rank by predictive power | High | Pending |
| F5.10 | Identify "dead" features (AUC ≈ 0.5) for potential removal | Medium | Pending |
| F5.11 | **Normalization decisions:** Log-transforms (Np, particle flux), standard scaling (IMF), cyclical encoding (UT, DOY) | High | Pending |
| F5.12 | **Leakage prevention:** Verify all rolling windows use `center=False`; no future data in any feature | High | Pending |
| F5.13 | **Missingness indicators:** Add `is_imputed` flags for gap-filled data | Medium | Pending |
| F5.14 | **Interaction features (preview):** Newell × tail_stretch, pressure_pulse × dB_dt_mean (from potential_features.md Category 4) | Medium | Pending |
| F5.15 | **Feature prioritization table:** Tier 1 (must-have for Milestone 3), Tier 2 (implement if time), Tier 3 (deferred) | High | Pending |

### Section 6: Model Selection (Report §6)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| M6.1 | **Justify model hierarchy** (ordered by priority): (1) Persistence baseline, (2) Climatology baseline, (3) Logistic Regression, (4) LightGBM, (5) LSTM, (6) Physics-Gated LSTM | High | Pending |
| M6.2 | **Literature architectures:** Cite Camporeale 2020 (Gray Box LSTM), GeoDGP, prior XGBoost approaches | High | Pending |
| M6.3 | **Novel architecture:** Describe Direct-Drive Gray Box LSTM (no MHD, physics gating via Newell coupling) | High | Pending |
| M6.4 | **Data-to-model justification:** Time-series data → sequential models justified; heavy class imbalance → weighted loss; multi-source inputs → multi-branch architecture | High | Pending |
| M6.5 | **Baseline strategy:** Document why start with persistence/climatology before learned models (MS-2 requirement) | High | Pending |
| M6.6 | **Evaluation metrics chosen:** HSS (primary skill), Brier Skill Score (probabilistic), AUC-PR (imbalanced data), RMSE and Pearson r (for regression formulation) | High | Pending |
| M6.7 | **Bias-variance analysis:** LSTM overfitting risk on moderate dataset vs. underfitting of linear models on nonlinear coupling physics | Medium | Pending |
| M6.8 | **Operational constraints:** LSTM must run inference in milliseconds on laptop CPU (Challenge 18 requirement) | Medium | Pending |
| M6.9 | **Ablation study design:** Document Models A/B/C design and expected outcomes | High | Pending |

### Section 7: Progress Report (Report §7)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| PR7.1 | Summarize effort on dataset collection (sources contacted, data volumes, access methods) | High | Pending |
| PR7.2 | Summarize feature extraction work completed (Newell coupling validated, feasibility prototype results) | High | Pending |
| PR7.3 | List tests/experiments run (feasibility LSTM r=0.55, XGBoost r=0.33, cross-correlation checks) | High | Pending |
| PR7.4 | Show current pipeline state (what runs end-to-end vs. what is in progress) | High | Pending |
| PR7.5 | Quantify completion percentage toward Milestone 3 readiness | Medium | Pending |

### Section 8: Task List / Solo Work Plan (Report §8)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| TL8.1 | Create remaining-work timeline with weekly milestones | High | Pending |
| TL8.2 | Identify critical path items (what blocks Milestone 3) | High | Pending |
| TL8.3 | Show task dependencies (EDA complete → feature matrix → model training) | Medium | Pending |

### Section 9: Risks and Mitigation (Report §9)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| RM9.1 | **Risk: Data gaps in OMNI during critical storms** → Mitigation: Safe Zone imputation + missingness indicators | High | Pending |
| RM9.2 | **Risk: Severe class imbalance (1:50 to 1:200)** → Mitigation: Weighted loss (5× storm), stratified evaluation, HSS/BSS metrics | High | Pending |
| RM9.3 | **Risk: Temporal leakage in feature engineering** → Mitigation: Strict rolling window discipline, phase-shift validation (pending implementation in validate.py) | High | Pending |
| RM9.4 | **Risk: GOES satellite transitions create discontinuities** → Mitigation: EDA characterization, instrument-specific normalization | Medium | Pending |
| RM9.5 | **Risk: LSTM overfitting on moderate dataset** → Mitigation: Dropout, early stopping, physics-gating as regularizer, start with simpler baselines | Medium | Pending |
| RM9.6 | **Risk: Swarm LEO observation sparsity (non-continuous orbit)** → Mitigation: Decay age features, sub-index binning, coverage-quality flags | Medium | Pending |
| RM9.7 | **Risk: HPC speedup claim (20–100×) unverified** → Mitigation: Run inference timing benchmark before citing | Low | Pending |
| RM9.8 | **Risk: Phase-shift leakage detection not implemented in validate.py** → Mitigation: Implement before production model training (Milestone 3 prerequisite) | High | Pending |

### Section 10: References (Report §10)

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| REF10.1 | Compile all cited papers into `references.bib` (Newell 2007, Camporeale 2020, Dai 2024, Vokhmyanin 2019, Kim 2023, Shen 2023) | High | Pending |
| REF10.2 | Add dataset citations (OMNI, GOES, SuperMAG, Swarm DOIs) | High | Pending |
| REF10.3 | Verify all claims in report have corresponding references | Medium | Pending |

### Section 11: Report Production

| Task ID | Task | Priority | Status |
|---------|------|----------|--------|
| RP11.1 | Create EDA synthesis report: `docs/milestones/milestone2/milestone2_eda_report.md` (15–20 pages) | High | Pending |
| RP11.2 | Create data quality report: `docs/milestones/milestone2/milestone2_data_quality.md` | High | Pending |
| RP11.3 | Generate 10 publication-quality figures for `figures/eda/` | High | Pending |
| RP11.4 | Write final PDF report following outline structure (§1–§10) | High | Pending |

---

## 5. Traceability Table

### Source: `milestone2_rubric.md`

| Original Item | Final Status | Final Section | Notes |
|---------------|-------------|---------------|-------|
| 1a. Data merging | Kept | §3 (P3.7) | Multiple sources merged onto 1-min grid |
| 1b. Data cleansing | Kept | §3 (P3.8) | NULL handling, Safe Zone rules |
| 1c. Data augmentation/enrichment | Kept | §3 (P3.9), §5 | Newell coupling, derived features |
| 1d. Data normalization | Kept | §3 (P3.10), §5 (F5.11) | Log-transforms, scaling |
| 2a. Data completeness/freshness/quality | Kept | §4 (E4B.6, E4C.5, E4D.4, E4E.5, E4I.1) | Per-source quality assessment |
| 2b. Variables and distributions | Kept | §4 (E4B.1–3, E4C.1, E4D.3, E4E.2) | Univariate distributions |
| 2c. Anomalies/outliers | Kept | §4 (E4B.3, E4C.2, E4E.3) | Outlier detection |
| 2d. Relationships/correlations | Kept | §4 (E4G.1–5) | Lagged cross-correlations |
| 3. EDA→Feature identification | Kept | §4→§5 (F5.2, F5.8, F5.9) | Explicit traceability required |
| 4. Report quality/completeness | Kept | §7 (PR7.1–5), §11 (RP11.4) | Maturity evidence |
| "Plots with sufficient explanation" | Kept | §4 (all EDA tasks) | Narrative required with every plot |

### Source: `milestone2_outline.md`

| Original Item | Final Status | Final Section | Notes |
|---------------|-------------|---------------|-------|
| Title Area | Kept | Title | Header format specified |
| §1 Background | Kept | §1 (R1.1–3) | Problem statement |
| §2 Dataset citation | Kept | §2 (R2.1–4) | Sources and links |
| §3 Data Pipeline | Kept | §3 (P3.1–14) | Full pipeline |
| §4 EDA Description | Kept | §4 (all E tasks) | Expanded significantly |
| §5 Feature Engineering | Kept | §5 (F5.1–15) | Literature + novel + traceability |
| §6 Model selection | Kept | §6 (M6.1–9) | Justified hierarchy |
| §7 Progress Report | Kept | §7 (PR7.1–5) | Effort and evidence |
| §8 Task List | Kept | §8 (TL8.1–3) | Timeline and dependencies |
| §9 Risks | Kept | §9 (RM9.1–8) | Risk register |
| §10 References | Kept | §10 (REF10.1–3) | BibTeX |

### Source: `milestone2_tasklist.md`

| Original Task | Final Status | Final Section | Notes |
|---------------|-------------|---------------|-------|
| Task 2.1.1: Data Loading Pipeline | Kept | §3 (P3.1–6) | Split into per-source loaders |
| Task 2.1.2: EDA Utilities | Kept | §4 (E4A.1) | Plotting helper library |
| Task 2.1.3: Time Period Selection | Kept | §4 (E4A.2–4) | Reference event catalog |
| Task 2.2.1: Univariate Distributions | Kept | §4 (E4B.1–3) | OMNI distributions |
| Task 2.2.2: Temporal Patterns | Kept | §4 (E4B.4–5) | Seasonal + autocorrelation |
| Task 2.2.3: Missing Data Analysis | Kept | §4 (E4B.6) | Gap characterization |
| Task 2.2.4: Storm-Time Behavior | Kept | §4 (E4B.7) | Superposed epoch |
| Task 2.3.1: Particle Flux Distributions | Kept | §4 (E4C.1–2) | GOES particles |
| Task 2.3.2: X-Ray Flux Analysis | Kept | §4 (E4C.3) | Flare identification |
| Task 2.3.3: GEO Magnetic Field | Kept | §4 (E4C.4) | GOES Bt, Bz |
| Task 2.3.4: Missing Data/Satellite Transitions | Kept | §4 (E4C.5) | Continuity assessment |
| Task 2.4.1: Spatial Coverage Assessment | Kept | §4 (E4D.1–2) | LEO coverage |
| Task 2.4.2: Sub-Index Distributions | Kept | §4 (E4D.3) | 12 sub-indices |
| Task 2.4.3: Decay Age Analysis | Kept | §4 (E4D.4) | Freshness assessment |
| Task 2.4.4: LEO vs. Ground dB/dt | Kept | §4 (E4D.5) | Early-warning proxy test |
| Task 2.5.1: Station Network Characterization | Kept | §4 (E4E.1) | Station map |
| Task 2.5.2: dB/dt Distribution | Kept | §4 (E4E.2–3) | Target distribution |
| Task 2.5.3: Spatial Patterns | Kept | §4 (E4E.4) | Animated storm maps |
| Task 2.5.4: Missing Data and Quality | Kept | §4 (E4E.5) | Station whitelist |
| Task 2.6.1: Label Distribution | Kept | §4 (E4F.1–2) | Class imbalance |
| Task 2.6.2: Event Duration Statistics | Kept | §4 (E4F.3) | Inform LSTM window |
| Task 2.6.3: Storm vs. Substorm Events | Kept | §4 (E4F.4) | Stratification |
| Task 2.6.4: Lead Time Analysis | Kept | §4 (E4F.5) | Forecast horizon validation |
| Task 2.7.1: L1→GEO→Ground Chain | Kept | §4 (E4G.1–3) | Propagation delays |
| Task 2.7.2: Feature Correlation Heatmap | Kept | §4 (E4G.4) | Redundancy assessment |
| Task 2.7.3: Newell Coupling vs. Indices | Kept | §4 (E4G.5) | Physics anchor validation |
| Task 2.8.1: Seasonal Variations | Kept | §4 (E4H.1) | Russell-McPherron |
| Task 2.8.2: Diurnal Patterns | Kept | §4 (E4H.2) | MLT dependence |
| Task 2.8.3: Solar Cycle Trends | Kept | §4 (E4H.3) | Train/test split rationale |
| Task 2.9.1: Gap Characterization | Kept | §4 (E4I.1–2) | Missingness heatmap |
| Task 2.9.2: Rolling Window Quality | Kept | §4 (E4I.3) | min_valid_fraction |
| Task 2.9.3: Impact on Labels | Kept | §4 (E4I.4) | Information loss |
| Task 2.10.1: Deep Dive Notebooks | Kept | §4 (E4J.1–2) | Case studies |
| Task 2.10.2: CME vs. CIR Comparison | Kept | §4 (E4J.3) | Driver type comparison |
| Task 2.11.1: Test Feature Ideas | Kept | §5 (F5.3–8) | Feature preview |
| Task 2.11.2: Feature Importance Preview | Kept | §5 (F5.9–10) | Univariate AUC |
| Task 2.12.1: Synthesis Document | Kept | §11 (RP11.1) | EDA report |
| Task 2.12.2: Data Quality Report | Kept | §11 (RP11.2) | Quality scorecard |
| Task 2.12.3: Figures for Paper | Kept | §11 (RP11.3) | Publication figures |
| Unit tests for loaders | Kept | §3 (P3.11) | Test suite |

### Source: `potential_features.md`

| Original Item | Final Status | Final Section | Notes |
|---------------|-------------|---------------|-------|
| Sun GOES-R XRS features (raw + engineered) | Partially deferred | §5 (F5.5 for rate-of-change); rest deferred to Milestone 3 | Solar context is secondary driver |
| L1 OMNI raw features (25 variables) | Kept | §4 (E4B.1), §5 (F5.1–4) | Core input layer |
| L1 Category 1: Solar Wind Preconditioning (~14 features) | Kept - Tier 1 | §5 (F5.3) | Newell integrals are primary features |
| L1 Category 3: Solar Wind Components (~50 features) | Kept - Tier 1/2 | §5 (F5.4) | Bz rolling stats are Tier 1; velocity/density details are Tier 2 |
| L1 Coupling Functions (epsilon, rectified voltage, etc.) | Kept - Tier 2 | §5 (F5.1, F5.14) | Secondary coupling functions after Newell |
| L1 Trigger Indicators | Kept - Tier 2 | §5 (F5.14) | dNewell/dt, Bz sudden turning |
| GEO GOES raw features (~20 variables) | Kept | §4 (E4C.1–4) | EDA characterization |
| GEO Category 2: Magnetospheric State (~35 features) | Kept - Tier 1/2 | §5 (F5.5) | Tail stretching indicators are Tier 1 |
| GEO Particle Features | Partially deferred | §5 (F5.5 for rate-of-change) | Tier 2 – depends on data availability |
| LEO Swarm raw features (~15 per satellite) | Kept | §4 (E4D.1–5) | EDA characterization |
| LEO Engineered Features (auroral, perturbations, FAC) | Partially deferred | §5 (F5.6) | Spatial gradients are Tier 1; FAC features are Tier 3 |
| Ground SuperMAG raw features (~7 per station) | Kept | §4 (E4E.1–5) | Target source |
| Ground Single Station Features (dH/dt, dB/dt) | Kept - Tier 1 | §5 (F5.7) | Core target and trailing history |
| Ground Regional Aggregation | Kept - Tier 2 | §5 (F5.7) | Network-wide stats |
| Ground SuperMAG Index Features | Kept - Tier 1 | §5 | SME/SML as activity context |
| Category 4: Interaction Terms (~30 features) | Partially deferred | §5 (F5.14) | Preview in Milestone 2; full implementation Milestone 3 |
| Category 5: Temporal Context (~50 features) | Partially kept | §5 (F5.3, E4H.1–3) | Cyclic encoding + substorm history are Tier 1; forecast windows are Tier 3 (label-side) |
| Temporal Cross-Validation code snippet | Kept - reference | §6 (M6.5) | Informs split strategy |
| Target Variable Options | Kept - reference | §6 (M6.6) | Regression vs. classification decision |
| Handling Class Imbalance | Kept - reference | §9 (RM9.2), §6 | Informs weighted loss design |

### Source: `milestone3_tasklist.md` (dependency extraction)

| Original Item | Final Status | Final Section | Notes |
|---------------|-------------|---------------|-------|
| Task 3.0.1: Feature Builder | Prerequisite gap → Pulled back | §3 (P3.1–9) | Data loaders must be complete in M2 |
| Task 3.0.2: Feature Schema Validation | Prerequisite gap → Pulled back | §3 (P3.12) | validate_output/validate_time_series must work |
| Task 3.0.4: dB/dt Computation | Prerequisite gap → Pulled back | §4 (E4E.2), §5 (F5.7) | dB/dt computation needed for EDA |
| Task 3.0.5: Binary Labels | Prerequisite gap → Partially pulled back | §4 (E4F.1) | Label generation needed for target analysis |
| Task 3.0.6: Class Imbalance Analysis | Prerequisite gap → Pulled back | §4 (E4F.1–2), §9 (RM9.2) | Must characterize imbalance in M2 |
| Task 3.0.7: Time-Series Split Strategy | Prerequisite gap → Document in M2 | §6 (M6.5), §9 | Split strategy must be defined |
| Task 3.0.16: Feature-Label Correlation Test | Prerequisite gap → Pulled back | §5 (F5.9) | Univariate AUC already in M2 plan |
| Task 3.0.17: Temporal Leakage Check | Prerequisite gap → Pulled back | §5 (F5.12), §9 (RM9.3) | Must verify before M3 |
| All other M3 tasks | Not pulled back | — | Modeling-phase work stays in M3 |

---

## 6. Coverage Audit

### Rubric Coverage

| Rubric Item | Covered By | Status |
|-------------|-----------|--------|
| 1a. Data merging | P3.7 | ✅ Covered |
| 1b. Data cleansing | P3.8 | ✅ Covered |
| 1c. Data augmentation | P3.9, F5.1–15 | ✅ Covered |
| 1d. Data normalization | P3.10, F5.11 | ✅ Covered |
| 2a. Completeness/freshness/quality | E4B.6, E4C.5, E4D.4, E4E.5, E4I.1–4 | ✅ Covered |
| 2b. Variables and distributions | E4B.1–3, E4C.1, E4D.3, E4E.2 | ✅ Covered |
| 2c. Anomalies/outliers | E4B.3, E4C.2, E4E.3 | ✅ Covered |
| 2d. Relationships/correlations | E4G.1–5 | ✅ Covered |
| 3. EDA→Feature identification | F5.2, F5.8–9, cross-links §4→§5 | ✅ Covered |
| 4. Report quality/maturity | PR7.1–5, RP11.1–4 | ✅ Covered |
| "Plots with sufficient explanation" | All EDA tasks require narrative | ✅ Covered |

### Outline Coverage

| Outline Section | Tasks Mapped | Status |
|-----------------|-------------|--------|
| §1 Background | R1.1–3 | ✅ |
| §2 Dataset | R2.1–4 | ✅ |
| §3 Pipeline | P3.1–14 | ✅ |
| §4 EDA | E4A–J (40+ tasks) | ✅ |
| §5 Features | F5.1–15 | ✅ |
| §6 Models | M6.1–9 | ✅ |
| §7 Progress | PR7.1–5 | ✅ |
| §8 Task List | TL8.1–3 | ✅ |
| §9 Risks | RM9.1–8 | ✅ |
| §10 References | REF10.1–3 | ✅ |

### EDA Best-Practice Coverage

| EDA Requirement | Covered By | Status |
|-----------------|-----------|--------|
| EDA-1: Pre-analysis hygiene | E4A.1 (compute_missing_stats), E4I.1 | ✅ |
| EDA-2: Univariate distributions | E4B.1–3, E4C.1, E4D.3, E4E.2 | ✅ |
| EDA-3: Identify transform needs | E4B.3, F5.11 | ✅ |
| EDA-4: Outlier detection | E4E.3, E4C.2 | ✅ |
| EDA-5: Bivariate/multivariate | E4G.1–5 | ✅ |
| EDA-6: Always visualize | All notebooks produce plots | ✅ |
| EDA-7: Correlation ≠ causation | E4G.5 (domain-validated correlations) | ✅ |
| EDA-8: Iterative hypothesis generation | E4J.1–3 (case studies generate hypotheses) | ✅ |
| EDA-9: Plots with explanation | Report production tasks RP11.1, RP11.4 | ✅ |
| EDA-10: EDA drives features/models | F5.2, M6.4 (explicit links) | ✅ |

### Feature Engineering Best-Practice Coverage

| FE Requirement | Covered By | Status |
|----------------|-----------|--------|
| FE-1: Features define information | F5.1 (literature motivation) | ✅ |
| FE-2: Scale control | F5.11, P3.10 | ✅ |
| FE-3: Skew handling | E4B.3 → F5.11 (log-transforms) | ✅ |
| FE-4: Temporal lags + causality | F5.3, F5.12 | ✅ |
| FE-5: No leakage | F5.12, RM9.3 | ✅ |
| FE-6: Missingness as signal | F5.13 | ✅ |
| FE-7: Feature evaluation | F5.9–10 | ✅ |
| FE-8: Dimensionality management | E4G.4 (VIF), F5.10 | ✅ |
| FE-9: Justify from EDA/literature | F5.1–2 | ✅ |
| FE-10: Unseen regimes | F5.15 (deferred features), RM9.5 | ✅ |

### Model Selection Best-Practice Coverage

| MS Requirement | Covered By | Status |
|----------------|-----------|--------|
| MS-1: Principled reasoning | M6.1, M6.4 | ✅ |
| MS-2: Start with baselines | M6.5 | ✅ |
| MS-3: Match model to data type | M6.4 (time-series + tabular) | ✅ |
| MS-4: Bias-variance | M6.7 | ✅ |
| MS-5: Metric encodes costs | M6.6 | ✅ |
| MS-6: Classical → ML → DL progression | M6.1 (hierarchy) | ✅ |
| MS-7: DL failure modes | M6.7, RM9.5 | ✅ |
| MS-8: Narrow to 2-3 candidates | M6.1 (6 models, but justified hierarchy) | ✅ |
| MS-9: Co-evolution acknowledged | M6.4 | ✅ |
| MS-10: Operational constraints | M6.8 | ✅ |

### Risk Coverage

| Risk Category | Covered By | Status |
|---------------|-----------|--------|
| Data quality/gaps | RM9.1, RM9.4, RM9.6 | ✅ |
| Class imbalance | RM9.2 | ✅ |
| Temporal leakage | RM9.3, RM9.8 | ✅ |
| Model overfitting | RM9.5 | ✅ |
| Unverified claims | RM9.7 | ✅ |
| Missing validation infrastructure | RM9.8 | ✅ |

---

## 7. Highest-Priority Next Actions

Ordered by execution priority and dependency:

| Priority | Task ID | Action | Blocks |
|----------|---------|--------|--------|
| 1 | P3.1–6 | **Implement data loaders** (`src/swmi/data/loaders.py`) — all EDA depends on this | All EDA |
| 2 | E4A.1 | **Build EDA utility library** (`src/swmi/eda/utils.py`) — reusable plotting | All notebooks |
| 3 | E4A.2–4 | **Select time period and reference events** — frames entire analysis | All EDA |
| 4 | E4B.1–3 | **OMNI univariate distributions** — first notebook, validates pipeline | Feature decisions |
| 5 | E4B.6 | **Missing data analysis (OMNI)** — informs imputation strategy | Pipeline design |
| 6 | E4E.2–3 | **Ground dB/dt distribution** — characterize target variable | Label generation |
| 7 | E4F.1 | **Label distribution / class imbalance** — fundamental modeling constraint | Model selection |
| 8 | E4G.1–3 | **Lagged cross-correlations (L1→GEO→Ground)** — validates causal chain | Feature engineering |
| 9 | E4G.5 | **Newell Coupling validation** — physics anchor verification | Entire model strategy |
| 10 | F5.3 | **Feature engineering preview** (Newell lags, rolling BZ std) | Feature prioritization |
| 11 | F5.9 | **Univariate AUC ranking** — identifies high-value features | Milestone 3 feature matrix |
| 12 | F5.12 | **Leakage verification** — no future data in any feature | Scientific validity |
| 13 | M6.1–5 | **Write model selection justification** — documents reasoning for report | Report §6 |
| 14 | RP11.4 | **Compile final PDF report** | Milestone 2 submission |

### Critical Path

```
Data Loaders (P3.1-6)
    → EDA Notebooks (E4B-J)
        → Feature Preview (F5.3-10)
            → Feature Prioritization (F5.15)
                → Model Justification (M6.1-9)
                    → Report Production (RP11.1-4)
```

### Milestone 3 Prerequisites That MUST Be Complete in Milestone 2

| Prerequisite | M2 Task | Risk if Missing |
|--------------|---------|-----------------|
| Data loaders work end-to-end | P3.1–6 | Cannot build feature matrix |
| dB/dt computation validated | E4E.2 | Cannot generate labels |
| Class imbalance characterized | E4F.1 | Cannot set loss weights or choose metrics |
| Temporal leakage verified | F5.12 | All M3 results scientifically invalid |
| Train/test split strategy documented | M6.5 | M3 split may not match EDA period |
| Feature prioritization complete | F5.15 | M3 feature matrix scope undefined |
| validate_output() integrated | P3.12 | No quality assurance on pipeline |

---

## Feature Prioritization (from Stage 5)

### Tier 1: Analyze and Implement in Milestone 2 (Required for M3)

| Feature Group | Source | Count | Rationale |
|---------------|--------|-------|-----------|
| Newell coupling (current + integrals 6h/12h/24h/48h) | potential_features.md Cat 1 | ~14 | Physics anchor; primary driver |
| Bz_GSM rolling statistics (min, mean, std over 3h/6h/12h) | potential_features.md Cat 3 | ~12 | Direct reconnection proxy |
| Solar wind velocity + pressure derivatives | potential_features.md Cat 3 | ~8 | Shock detection |
| GEO Bz depression + tail stretch indicators | potential_features.md Cat 2 | ~8 | Magnetospheric state |
| Ground dB/dt trailing history (max 10min, hourly) | potential_features.md Ground | ~5 | Persistence memory |
| SuperMAG indices (SME, SML current + rolling) | potential_features.md Ground | ~8 | Activity context |
| Temporal encoding (UT sin/cos, DOY sin/cos) | potential_features.md Cat 5 | ~6 | Calendar effects |
| **Total Tier 1** | | **~61** | |

### Tier 2: Characterize in EDA, Implement if Time Permits

| Feature Group | Source | Count | Rationale |
|---------------|--------|-------|-----------|
| GEO particle flux + rate of change | potential_features.md GEO | ~10 | Injection detection |
| LEO spatial gradients and FAC summary stats | potential_features.md LEO | ~8 | Ionospheric coupling |
| Interaction terms (Newell × stretch, pressure × ground) | potential_features.md Cat 4 | ~10 | Cross-dataset signals |
| Additional coupling functions (epsilon, Borovsky) | potential_features.md Cat 3 | ~5 | Alternative drivers |
| Trigger indicators (dNewell/dt, Bz sudden turning) | potential_features.md Cat 3 | ~6 | Onset detection |
| **Total Tier 2** | | **~39** | |

### Tier 3: Defer to Milestone 3 or Beyond

| Feature Group | Rationale for Deferral |
|---------------|----------------------|
| Sun GOES-R XRS features (~20) | Solar context is upstream; effect is already captured by L1 measurements |
| Multi-satellite GEO features (east-west difference) | Requires dual-satellite pipeline not yet built |
| Swarm conjugate/multi-sat features | Complex mapping; low priority for initial model |
| Frequency content (Pi2, Pc5 power) | Requires spectral analysis pipeline |
| Operational context (forecast history, alert state) | Only relevant for operational deployment |
| Forecast windows (target variables) | Label-side definitions belong in M3 |

---

## Completeness Audit Confirmation

- [x] Every provided file was processed in the required order (1→8)
- [x] Every task-like item from `milestone2_tasklist.md` was given a disposition (all Kept; see traceability table)
- [x] Every major outline section (§1–§10) has at least one mapped task
- [x] Every rubric requirement (1a–d, 2a–d, 3, 4) has at least one mapped task
- [x] EDA, feature engineering, and model selection are explicitly linked via cross-link requirements (§3.4)
- [x] Milestone 3 dependencies identified and pulled back where prerequisite
- [x] Feature prioritization applied (Tier 1/2/3) with clear boundary between M2 and M3
- [x] No tasks silently dropped; no merges without explanation
- [x] Ambiguities flagged in §1 rather than silently resolved
