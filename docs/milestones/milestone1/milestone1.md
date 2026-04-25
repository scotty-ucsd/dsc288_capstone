# DSC 288: Capstone Project

## Milestone 1: Abstract

### Title: Physics-Informed Forecasting of Regional Geomagnetic Perturbations

### Author: Scotty Rogers

### Project Goal

* Apply physics informed ML models to improve regional forcasting of hazardous geomagnegtic perturbations ($dB/dt$) via end to end multimodal dataset (Solar/L1/GEO/LEO/Earth)

* *Regional forcasting of $dB/dt$ can be broken down into two supervised learning tasks.*

    * **Classification Task:** Is a substorm immeninet (yes/no)?
        * Baseline Model:
        * Primary Model: LightGBM

    * **Regression Task:** At some time $t$, how intense will $dB/dt$ be at a specific *SuperMAG* ground station
        * Baseline Model:
        * Primary Model:

---

### Project Details
* **Description:** Background, Problem Definition, Motivation, Literature Review, Approach
* Note: this needs to be updated but some could be helpful

---

#### Background
* **Description:** *domain specifics and the broad problem for which you are planning to apply data science*
Coronal Mass Ejections (CMEs) and high-speed solar wind streams drive dynamic deformations of Earth's magnetosphere, triggering substorms that generate Geomagnetically Induced Currents (GICs). These currents pose severe risks to electrical power grids, potentially causing voltage collapse, transformer damage, and significant economic loss. Despite this threat, current operational forecasting remains limited to broad global indices, failing to provide the regional environmental characterization demanded by the electrical power sector. [INSERT TRANSISTION SENTENCE] This research utilizes a continuous, high-fidelity multi-point data fusion strategy. This data fusion is achieved by integrating GOES X-ray flux, L1 solar wind measurements (OMNI), magnetospheric state data from GEO/LEO satellites, and ground magnetometer arrays (SuperMAG) from 2015 to 2024 into a comprehensive, standalone training dataset. This temporal range covers the end of Solar Cycle 24 and the rising phase of Cycle 25, mitigating the quiet-time bias prevalent in current literature. By embedding physics-derived features, specifically the Newell Coupling function representing magnetic reconnection efficiency,into the learning process, where model generalizability and physical consistency are enhanced.

---

#### Problem Definition
* **Description:** 
    * What specifically is the problem you will solve? 
    * Describe the expected inputs and outputs of ML models
    * **IMPORTANT:** Be precise and succinct. 
        * (1 point) *Clarity of a problem statement what are you trying to predict precisely and what are the inputs to this?*
* **Overview:** This project is structured around three connected modeling problems
  within a single Sun-to-ground forecasting pipeline.
* **Problem 1 (LightGBM):** Binary classification using a label derived from
  station-level $dB/dt$ time series from SuperMAG. The task is to predict whether
  regional ground magnetic perturbation will exceed a chosen threshold at a fixed
  lead time of 60 minutes.
* **Problem 2 (Regime / Context Model):** Solar wind classification or context
  estimation using OMNI-derived upstream features. The goal is to infer soft
  regime probabilities or disturbance-state context that can be appended to
  downstream forecasting models.
* **Problem 3 (LSTM):** Regional $dB/dt$ prediction using temporal sequence
  modeling. The goal is to test whether explicit temporal memory improves
  one-hour-ahead forecasting relative to engineered lag features in tabular models.

---

#### Motivation
* **Description:**
    * Why is this a problem where ML will work well? 
    * Are there datasets (of sufficient size) you can use to train and test your model?
This is a strong machine learning problem because the Sun-to-ground system is
nonlinear, multiscale, and only partially observed, making simple analytical
mappings insufficient while still offering abundant historical observational data.
Prior work has shown that neural networks, probabilistic forecasting systems, and
gray-box approaches can extract predictive skill from solar wind and geomagnetic
observations even when first-principles forward modeling remains computationally
expensive or incomplete (Camporeale; 2020).

Machine learning is especially appropriate here because the forecast target is
operational rather than purely explanatory: the practical goal is not to simulate
every microphysical process, but to predict elevated risk of local ground
disturbance with enough lead time and calibration quality to be useful.

---

#### Literature Review
* **Description:**
    * How people have solved this problem in the research community, providing references. 
    * Summarize the differences between the problem you are solving and what others have solved
    * Summarize the approaches you intend to take vs what others have taken.

---

#### Approach 
* **Description:**
    * a few lines on what approach you intend to take to solve the problem
    * the reasons why you are taking this approach. 
    * If you are planning to implement an existing idea please explicitly state so and cite 

---

### Technical Details
* **Description** Dataset, feature extraction/engineering, Models

#### Dataset
* current fused columns: 
    ```python
     ['timestamp', 'omni_bx_gse', 'omni_by_gsm', 'omni_bz_gsm', 'omni_f',
       'omni_vx', 'omni_proton_density', 'omni_pressure', 'omni_sym_h',
       'omni_al', 'omni_au', 'goes_bz_gsm', 'goes_bt', 'goes_bx_gsm',
       'goes_by_gsm', 'goes_satellite', 'goes_missing_flag', 'leo_high_lat',
       'leo_high_lat_decay_age', 'leo_high_lat_is_fresh', 'leo_high_lat_count',
       'leo_mid_lat', 'leo_mid_lat_decay_age', 'leo_mid_lat_is_fresh',
       'leo_mid_lat_count', 'leo_dayside', 'leo_dayside_decay_age',
       'leo_dayside_is_fresh', 'leo_dayside_count', 'leo_nightside',
       'leo_nightside_decay_age', 'leo_nightside_is_fresh',
       'leo_nightside_count', 'ABK_b_e', 'BJN_b_e', 'TRO_b_e', 'ABK_b_n',
       'BJN_b_n', 'TRO_b_n', 'ABK_b_z', 'BJN_b_z', 'TRO_b_z',
       'ABK_dbdt_magnitude', 'BJN_dbdt_magnitude', 'TRO_dbdt_magnitude',
       'ABK_dbdt_missing_flag', 'BJN_dbdt_missing_flag',
       'TRO_dbdt_missing_flag', 'ABK_dbe_dt', 'BJN_dbe_dt', 'TRO_dbe_dt',
       'ABK_dbn_dt', 'BJN_dbn_dt', 'TRO_dbn_dt', 'ut_sin', 'ut_cos', 'doy_sin',
       'doy_cos', 'newell_phi', 'omni_bz_gsm_valid_points_10m',
       'omni_bz_gsm_mean_10m', 'omni_bz_gsm_std_10m',
       'omni_bz_gsm_valid_points_30m', 'omni_bz_gsm_mean_30m',
       'omni_bz_gsm_std_30m', 'omni_vx_valid_points_10m', 'omni_vx_mean_10m',
       'omni_vx_std_10m', 'omni_vx_valid_points_30m', 'omni_vx_mean_30m',
       'omni_vx_std_30m', 'newell_phi_valid_points_10m', 'newell_phi_mean_10m',
       'newell_phi_std_10m', 'newell_phi_valid_points_30m',
       'newell_phi_mean_30m', 'newell_phi_std_30m', 'omni_bz_gsm_missing',
       'omni_by_gsm_missing', 'omni_vx_missing', 'omni_proton_density_missing',
       'omni_pressure_missing', 'goes_bz_gsm_missing', 'l1_any_missing',
       'geo_any_missing', 'omni_vx_ffill_applied',
       'omni_proton_density_ffill_applied', 'omni_pressure_ffill_applied',
       'year', 'month']
    ```
* will add solar and remove any leaky features(`omni_al`,`omni_ae`)
* 1. Solar: GOES Xray
    * **Source:**
    * **Description:**
    * **Why:**
    * **Features:**

* 2. L1 (First Lagrange Point): OMNI
    * Source:
    * **Description/Why:** 
    * Features:

* 3. GEO (Geosyncrinous Earth Orbit): GOES
    * Source:
    * **Description/Why:** 
    * Features:

* 4. LEO (Low Earth Orbit): Swarm
    * **Source:**
    * **Description**
    * **Why:** 
    * **Features:**
        * Comment on `ChaosMagPy` use to handel sparse observation

* 5. Ground (*Target $dB/dt$*): SuperMAG
    * **Source:**
    * **Description**
    * **Why:** 
    * **Features:**

#### Data Acquisition and Preparation Pipeline
* **Description:** just comment on existing pipeline
    ```
    ============================================================
    Pipeline complete: 120 months attempted
    Succeeded: 118
    Failed:    2
    Failed entries logged to: logs/failed_months.log
    2015-03     retrieve_supermag
    2015-03     retrieve_supermag
    2015-03     retrieve_supermag
    2015-03     retrieve_supermag
    2015-03     validate_sources
    2020-01     retrieve_supermag
    2020-01     retrieve_supermag
    2017-09     retrieve_supermag
    2012-03     validate_sources
    2015-07     retrieve_swarm_A
    2017-01     retrieve_swarm_A
    ============================================================
    ```
    * **Note:** *I think these dates failed since these dates already existed while 2015 to 2024 was processing*


#### Features Engineering
* **Description:** Newell Coupling, Carrington rotation period, MLT, ect.
    * **Explainitation of Feature:**
    * **Why/Reasoning**

#### Target Models and Model Success Criteria 

##### Model 1: Persistence Baseline
* **Description:** Predict that the future disturbance label at $t+60$ matches
  the current or most recent observed label.
* **Model:** Persistence rule
* **Model Reasoning:** Establishes the minimum operational baseline. Any trained
  model that fails to beat persistence provides no operational value.
* **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score
* **Success Criteria:** All learned models should exceed persistence on the
  held-out test set

##### Model 2: Climatology Baseline
* *Description:** Predict the unconditional event probability derived from the
  training period event rate.
* **Model:** Climatology (constant forecast equal to training-set base rate)
* **Model Reasoning:** Required reference for Brier Skill Score. BSS is defined
  relative to climatology, so this must be explicitly computed and reported.
* **Evaluation Metric:** Brier Score (absolute); denominator for BSS of all other
  models
* **Success Criteria:** Serves as the probabilistic calibration floor

#### Model 3: Logistic Regression
* **Description:** Linear probabilistic classifier on the full engineered feature
  matrix with L2 regularization.
* **Model:** Logistic Regression (`sklearn`, `solver='saga'`,
  `class_weight='balanced'`, regularization strength tuned on validation HSS)
* **Model Reasoning:** Establishes the linear ceiling and provides an interpretable
  coefficient-level view of which features contribute. If LightGBM dramatically
  outperforms logistic regression, it confirms that nonlinear interactions are
  necessary for this problem.
* **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score, Precision/Recall at
  HSS-optimal threshold
* **Success Criteria:** Outperform persistence (Model 1) and provide a stable
  linear comparison point for Models 5 and 6

#### Model 4: Regime-Context Model
* **Description:** Infer solar wind or geomagnetic regime probabilities from
  OMNI-derived features and append the resulting soft probability vector as context
  features to Models 5 and 6.
* **Model:** Gaussian Mixture Model (unsupervised) or weakly supervised LightGBM
  classifier with regime labels derived from SYM-H thresholds and solar wind speed
  percentiles; inspired by Camporeale et al. probabilistic solar wind classification
  (Camporeale; 2020)
* **Model Reasoning:** Regime context encodes system state - quiet, CIR-driven,
  CME-driven, recovery — explicitly, rather than forcing the main classifier to
  infer it implicitly from instantaneous solar wind values. The soft probability
  output also supports stratified evaluation of model skill by storm driver type,
  which is scientifically informative for the source ablation analysis.
* **Evaluation Metric:** Silhouette score or BIC for the unsupervised path;
  downstream ΔHSS of Models 5 and 6 when regime features are included vs. excluded
* **Success Criteria:** Regime probabilities produce ΔHSS > 0.02 in the main
  model ablation, or provide scientifically meaningful stratification of forecast
  errors by event type

#### Model 5: LightGBM Main Model
* **Description:** Primary nonlinear classifier on the fused Sun/L1/GEO/LEO/ground
  feature matrix including rolling statistics, physics-derived features (Newell
  $\Phi$), and regime-context probabilities from Model 4. False-positive reduction
  is an explicit design goal.
* **Model:** LightGBM `LGBMClassifier`
* **Model Reasoning:** Best-in-class for mixed-frequency tabular data with class
  imbalance and natively handles missing values, which is directly relevant given
  the LEO freshness flags and GOES missing indicators already in the dataset.
  Interpretable via SHAP for source attribution. Computationally efficient enough
  to support iterative threshold and weight tuning. Consistent with model choices
  in recent operational space weather forecasting literature (Ferdousi; 2025)
  (Keebler; 2025).
* **False-positive mitigation strategy:**
  * `scale_pos_weight` tuned in range 5–20, not set to the full imbalance ratio
    which would aggressively maximize recall at the cost of precision
  * decision threshold swept on validation Precision-Recall curve; selected at
    maximum HSS rather than defaulted to 0.5
  * isotonic probability calibration post-training (`CalibratedClassifierCV`),
    addressing calibration gaps documented in operational product evaluations
    (Camporeale; 2025)
  * evaluate on HSS as primary tuning metric rather than accuracy or AUC
* **Evaluation Metric:** HSS (primary), ROC AUC, Brier Skill Score,
  Precision/Recall at operational threshold, reliability diagram, confusion matrix
* **Success Criteria:** HSS > 0.35 on 2024 test set; BSS > 0.10 relative to
  climatology; Precision > 0.50 at HSS-optimal threshold; outperforms all baseline
  models (Models 1–3)

#### Model 6: LSTM
* **Description:** Temporal sequence model trained on fixed-length lookback windows
  for one-hour-ahead binary classification, without relying on hand-engineered lag
  statistics.
* **Model:** LSTM with dropout and recurrent dropout (Hochreiter; 1997); trained
  with focal loss (γ = 2) to concentrate gradient updates on rare positive events
* **Model Reasoning:** LightGBM operates on hand-crafted lag statistics (10-minute
  and 30-minute rolling means and standard deviations). An LSTM learns the temporal
  structure end-to-end and may capture higher-order dependencies — for example, the
  rate of change of southward Bz excursion, or the temporal shape of a pressure
  impulse — that fixed-window statistics miss. Prior work confirms that LSTM
  architectures extract useful temporal structure from solar wind sequences for
  geomagnetic forecasting (Collado-Villaverde; 2021) (Arxiv LSTM; 2025).
* **Evaluation Metric:** Same as Model 5; additionally compare against Model 5
  using DeLong test on AUC to assess statistical significance of any skill
  difference
* **Success Criteria:** HSS ≥ Model 5 on 2024 test set (demonstrates that sequence
  modeling adds skill beyond tabular lag features); informative even if it does not
  exceed LightGBM, by characterizing whether temporal memory is the binding
  constraint

---

### Success Criteria
* **Description:**
    * **Metrics:**What quantitative metrics will you measure to define the success of your project?
        * Literature Reasonsing
        * Theortical Reasoning
    * **Past Results:** What is your expectation of success?
        * ex: $db/dt$ hazardous is skewed(i.e. lots of false positives)
    * GEM dates from *SuperMAG*

---

