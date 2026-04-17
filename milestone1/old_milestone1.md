# Project Proposal

**1. Title:** [Insert Title Here]  
**2. Authors:** [Insert Author Names]

---

### 3. Abstract Details


**a) Background** Provide a few lines on the domain and the broad problem for which you are planning to apply data science.

**b) Problem Definition** What specifically is the problem you will solve? Describe the expected inputs and outputs of the "black box" that you will build out during this quarter. Be precise and succinct.

**c) Motivation** * **Suitability:** Why is this a problem where ML will work well?  
* **Data Availability:** Are there datasets (of sufficient size) you can use to train and test your model?

**d) Literature Review** * **Prior Research:** A few lines on how people have solved this problem in the research community, providing references.  
* **Comparison:** Summarize the differences between the problem you are solving and what others have solved and/or the approaches you intend to take vs. what others have taken.

**e) Your Approach** * **Methodology:** A few lines on what approach you intend to take to solve the problem and the reasons why you are taking this approach.  
* **Attribution:** If you are planning to implement an existing idea, please explicitly state so and cite prior work.

---

### 4. Technical Implementation Details
Include specific details regarding:
* **Dataset:** Source and characteristics of the data. Is there sufficient data to separate out training and test groups (preferably more than one test group)?
* **Feature Extraction:** Methods for processing raw data and identifying relevant variables.
* **Algorithm:** The specific models you intend to use (e.g., SVM, Random Forest, CNN, GAN, etc.) and the justification for selecting these models.

---

### 5. Success Criteria
What quantitative metrics will you measure to define the success of your project? What is your expectation of success?

---

## Grading Rubric

| Criterion | Description | Points |
| :--- | :--- | :--- |
| **Problem Statement** | Clarity of prediction goal; precise definition of inputs/outputs. | 1.0 |
| **Motivation & Lit Survey** | Importance of the problem, quality of research review, and clear identification of novelty. | 1.0 |
| **Dataset Identification** | Understanding of data depth; size sufficiency; availability of ground truth for supervised learning. | 1.5 |
| **Solution Approaches** | Justification for specific models and the technical methodology intended. | 1.0 |
| **Success Criteria** | Quantitative metrics and clear expectations of project success. | 0.5 |
| **Total** | | **5.0** |

---

### Use PCA for
- checking whether Sun/L1/GEO/LEO/ground blocks occupy different variance structure,
- seeing storm vs nonstorm separation,
- inspecting whether solar additions create meaningful latent structure.

### Use ablation for
- **L1 only**
- **L1 + GEO**
- **L1 + GEO + LEO**
- **L1 + GEO + LEO + ground-history**
- **Sun + L1 + GEO + LEO + ground-history**
- optional **full model + solar-wind regime probabilities**

That ablation will tell a much stronger story.

---

# Title: Physics-Informed Forecasting of Regional Geomagnetic Perturbations
# Author: Scotty Rogers

## Abstract

Coronal Mass Ejections (CMEs) and high-speed solar wind streams drive dynamic
deformations of Earth's magnetosphere, triggering substorms and large magnetic
perturbations that can produce Geomagnetically Induced Currents (GICs). These
disturbances pose significant risks to power systems through transformer damage,
voltage instability, and broader infrastructure disruption. Despite this threat,
most operational forecasting remains focused on broad global geomagnetic indices
or computationally expensive physics-based models, limiting the availability of
low-latency, region-specific hazard guidance. Prior gray-box work has shown that
machine learning can improve probabilistic estimates of regional ground magnetic
perturbations, but those systems often remain coupled to expensive simulation
pipelines rather than operating as direct end-to-end observational forecasters
(Camporeale; 2020).

This study proposes a physics-informed machine learning framework for forecasting
regional ground magnetic perturbations, expressed as $dB/dt$ in nT/min, using a
continuous Sun-to-ground observation chain. The feature pipeline fuses compact
solar context, upstream solar wind observations at L1, magnetospheric measurements
at GEO, ionospheric context at LEO, and regional ground magnetic response from
SuperMAG stations. The central scientific question is whether intermediate
observational layers between upstream solar forcing and ground response improve
one-hour-ahead regional forecasting beyond L1-only baselines. The project
emphasizes operationally relevant classification, calibrated probabilistic
outputs, source ablation, and interpretable machine learning, with the broader aim
of testing a system-based heliophysics framing for regional space weather hazard
monitoring (Tasistro-Hart; 2021) (Billcliff; 2026) (Heliophysics Decadal Survey;
2025).

---

## Background

Space weather describes the chain of physical processes linking solar forcing,
solar wind propagation, magnetospheric dynamics, ionospheric response, and ground
magnetic perturbations. The coupling begins at the Sun, where flares and CMEs
launch energetic plasma into the heliosphere. That plasma is then measured in situ
at L1, roughly 1.5 million km upstream of Earth, providing the primary real-time
driver signal used in most operational space weather systems. However, the L1
observation is only a single-point sample of a structured and time-varying solar
wind front, and the propagation from L1 to Earth introduces additional uncertainty
(Milan; 2022) (Vokhmyanin; 2019).

Once the solar wind reaches the magnetosphere, dayside magnetic reconnection
drives magnetopause erosion and controls the rate of energy entry into the coupled
magnetosphere-ionosphere system. The Newell coupling function quantifies this
energy transfer as a function of solar wind speed and IMF clock angle, and has
become a standard physics-informed feature in data-driven forecasting (Newell;
2007). Reconnection at the dayside magnetopause compresses the magnetosphere,
deposits open magnetic flux in the tail lobes, and eventually triggers explosive
tail reconnection events known as substorms (Fuselier; 2010) (Nagai; 2023)
(Nature Communications; 2024). Substorms release stored magnetic energy impulsively
into the ionosphere, producing rapid electrojet current variations that are
directly observed at ground-based magnetometer networks as large $dB/dt$
excursions (Gjerloev; 2012).

For electric power applications, the most operationally relevant environmental
quantity is not simply a broad planetary index, but localized rapid magnetic field
variation, because $dB/dt$ is directly related to induced geoelectric fields and
therefore to GIC risk. Reviews of GIC impacts globally have confirmed that local
response can differ substantially from global storm summaries, making regional
prediction an important applied data science problem (Carter; 2016). Recent
heliophysics strategy documents also argue for more integrated Sun-to-Earth and
system-level approaches rather than isolated single-domain studies, and explicitly
identify regional ground magnetic perturbation forecasting as an area where
data-driven methods are underexplored (Heliophysics Decadal Survey; 2025).
Satellite-based measurements at GEO and LEO provide intermediate context between
upstream L1 forcing and ground response. GEO observations can capture local
magnetopause dynamics, including erosion events that precede substorm-level
disturbances (Kim; 2024), while LEO particle precipitation data serves as a proxy
for auroral energy input and hemispheric power (Shen; 2023). These intermediate
layers are rarely fused together in a single direct observational pipeline,
motivating the multi-layer architecture proposed here (Ferdousi; 2025)
(Keebler; 2025).

---

## Problem Definition

- **Overview:** This project is structured around three connected modeling problems
  within a single Sun-to-ground forecasting pipeline.
- **Problem 1 (LightGBM):** Binary classification using a label derived from
  station-level $dB/dt$ time series from SuperMAG. The task is to predict whether
  regional ground magnetic perturbation will exceed a chosen threshold at a fixed
  lead time of 60 minutes.
- **Problem 2 (Regime / Context Model):** Solar wind classification or context
  estimation using OMNI-derived upstream features. The goal is to infer soft
  regime probabilities or disturbance-state context that can be appended to
  downstream forecasting models.
- **Problem 3 (LSTM):** Regional $dB/dt$ prediction using temporal sequence
  modeling. The goal is to test whether explicit temporal memory improves
  one-hour-ahead forecasting relative to engineered lag features in tabular models.

The black-box inputs consist of synchronized multimodal observations at 1-minute
cadence, including compact solar context, L1 solar wind drivers, GEO observations,
LEO context, and recent ground-history features. The outputs are either calibrated
probabilities of threshold exceedance, regime-context probabilities, or
sequence-based regional disturbance predictions depending on the model stage.

---

## Motivation

### Suitability for Machine Learning

This is a strong machine learning problem because the Sun-to-ground system is
nonlinear, multiscale, and only partially observed, making simple analytical
mappings insufficient while still offering abundant historical observational data.
Prior work has shown that neural networks, probabilistic forecasting systems, and
gray-box approaches can extract predictive skill from solar wind and geomagnetic
observations even when first-principles forward modeling remains computationally
expensive or incomplete (Tasistro-Hart; 2021) (Camporeale; 2020).

Machine learning is especially appropriate here because the forecast target is
operational rather than purely explanatory: the practical goal is not to simulate
every microphysical process, but to predict elevated risk of local ground
disturbance with enough lead time and calibration quality to be useful. This
aligns with broader arguments in Earth and space science that data-driven models
can complement or partially replace slower simulation workflows when the objective
is fast, decision-oriented prediction (Ziheng; 2022) (Bodnar; 2025) (Billcliff;
2026). Space weather specifically has been identified as a domain where ML methods
are maturing rapidly and offer near-term operational value, provided that
uncertainty quantification and calibration are treated as first-class requirements
alongside raw predictive skill (Abduallah; 2024) (Camporeale; 2021)
(Space Weather Review; 2025).

### Data Availability

This project has access to a sufficiently large and diverse multimodal dataset
spanning 2015–2024 at 1-minute cadence, integrating Sun, L1, GEO, LEO, and ground
sources into a unified feature matrix. The existing pipeline already supports
millions of timestamped rows and can be split chronologically into train,
validation, and held-out test periods, which is essential for avoiding leakage in
strongly autocorrelated time-series problems. OMNI data quality has been
characterized in the literature, including documented gap patterns and
interpolation artefacts that are handled by the existing missingness flags in the
dataset (Vokhmyanin; 2019). L1 data quality and the representativeness of the
OMNI multi-mission composite have also been assessed in the context of forecasting
applications (Milan; 2022).

There is also a viable source of supervised ground truth because the target labels
are derived directly from observed ground magnetic perturbations processed using
the SuperMAG standardized pipeline (Gjerloev; 2012), rather than inferred proxy
classes alone. Auxiliary labels such as substorm onset catalogs or regime labels
can be used as supporting context features or weak supervision without replacing
the primary target.

### Importance and Novelty

The importance of the problem is straightforward: regional $dB/dt$ forecasting is
directly relevant to GIC risk and therefore to grid operations and satellite
operations, while currently available broad geomagnetic indices are often too
coarse for local hazard characterization (Carter; 2016) (Mostafa; 2025). The
novelty of this project lies in combining a system-based Sun/L1/GEO/LEO/ground
fusion architecture with regional $dB/dt$ forecasting in a direct observational
pipeline, rather than relying solely on L1 inputs or using machine learning only
as a post-processing layer over large geospace simulations (Camporeale; 2020)
(Tasistro-Hart; 2021) (Billcliff; 2026). User needs assessments for operational
space weather products have also confirmed that regional, calibrated, probabilistic
hazard guidance is among the most frequently cited gaps in current forecast
capability (SWAG; 2024), further motivating the applied framing of this project.

---

## Literature Review

### Prior Research

Prior machine learning studies in space weather have largely focused on forecasting
global or quasi-global indices such as Dst, Kp, SYM-H, or related storm measures
using solar wind observations from L1. Tasistro-Hart et al. showed that combining
L1 data with solar-disk observations can support probabilistic geomagnetic storm
forecasting, but also found that direct solar inputs alone did not fully solve
storm-onset prediction and that L1 observations remained dominant (Tasistro-Hart;
2021). Collado-Villaverde et al. demonstrated that deep neural network
architectures trained on solar wind sequences can produce competitive forecasts of
global geomagnetic activity, reinforcing the value of sequence-based feature
extraction for this class of problem (Collado-Villaverde; 2021).

Camporeale et al. developed a gray-box approach for probabilistic regional ground
magnetic perturbation forecasting by enhancing the NOAA operational Geospace model
with machine learning, demonstrating the value of combining physical structure with
data-driven prediction. However, that work still depends on the upstream simulation
stack rather than operating as a direct observational forecaster (Camporeale;
2020). A subsequent evaluation of NOAA's operational forecast product found
persistent calibration deficiencies and skill gaps for rare high-amplitude events,
which motivates the calibration-first design emphasis in this project (Camporeale;
2025). Billcliff et al. extended lead-time geomagnetic storm forecasting by using
solar-wind ensembles and machine learning, arguing that near-Sun context becomes
most useful when transformed into propagation-aware future solar-wind scenarios
rather than simply appended as raw solar indicators (Billcliff; 2026).

Related work on solar wind classification has shown that probabilistic context
vectors can be useful representations of system state. Camporeale et al. used OMNI
data with a Gaussian Process classifier to output a probability vector over solar
wind categories, demonstrating a useful intermediate representation between raw
drivers and downstream prediction tasks (Camporeale; 2020). Abduallah et al.
demonstrated that uncertainty quantification integrated directly into the
forecasting architecture can improve operational reliability for geomagnetic index
prediction, providing a template for the calibration emphasis in this project
(Abduallah; 2024). Recent overviews of the current state of geomagnetic
disturbance forecasting — including both physics-based and ML approaches — confirm
that direct observational fusion across multiple layers of the coupled system
remains underexplored relative to single-domain L1-to-index mappings (Ferdousi;
2025) (Keebler; 2025).

Sequence modeling for geomagnetic applications has also been explored through
cosmic ray and geomagnetic storm prediction using LSTM architectures, confirming
that recurrent networks can extract temporal structure from solar wind time series
that is not captured by static lag features (Hochreiter; 1997) (Arxiv LSTM; 2025).
Chen et al. introduced GeoDGP, a deep Gaussian process framework for geomagnetic
disturbance prediction that places uncertainty quantification at the center of the
forecast rather than as a post-processing step, highlighting the gap this project
also aims to address through calibrated probabilistic outputs (Chen; 2025).
Advances in AI-based Earth science models more broadly also suggest that
multi-domain sensor fusion is becoming a standard architectural pattern for
prediction problems that span spatially and temporally heterogeneous data sources
(Bodnar; 2025) (Ziheng; 2022).

AI and ML applications in space weather operations have been reviewed in the
context of readiness for real-time deployment, with evaluations noting that most
published models lack operational validation against independent benchmark periods
(Thaker; 2025) (Camporeale; 2021). This motivates the explicit use of a held-out
2024 test set and GEM benchmark intervals in this project's evaluation design. The
impact of geomagnetic storms on satellites in LEO has also been documented,
reinforcing that the downstream consequences of regional disturbances extend beyond
ground-based infrastructure to orbital assets (Mostafa; 2025).

### Comparison to Prior Work

This project differs from most prior work in three main ways. First, the target is
regional ground magnetic perturbation exceedance rather than only a global storm
index. Second, the feature design explicitly spans multiple observational layers of
the coupled Sun–Earth system, including GEO and LEO context that are often omitted
in L1-centric studies. Third, the framework is intended to operate as a direct
observational forecasting system rather than as a machine-learning wrapper around
an expensive MHD simulation (Camporeale; 2020) (Tasistro-Hart; 2021) (Billcliff;
2026).

The project does not claim to replace the scientific value of physics-based
simulation. Instead, it tests whether a system-based observational fusion model can
deliver useful one-hour-ahead probabilistic forecasts for regional $dB/dt$, while
also quantifying which source layers actually contribute predictive value through
ablation (Heliophysics Decadal Survey; 2025). The evaluation design is also
explicitly calibration-aware, addressing gaps identified in operational product
assessments (Camporeale; 2025) and space weather user needs surveys (SWAG; 2024).

---

## Approach

### Methodology

The project will construct a unified 1-minute multimodal feature matrix covering
Sun, L1, GEO, LEO, and ground observations across 2015–2024. The modeling plan is
structured around three connected tasks: a main binary classifier for threshold
exceedance, an auxiliary solar wind regime-context model, and an LSTM-based
temporal prediction model. The core modeling ladder will include persistence and
climatology baselines, logistic regression, a regime-context model, a LightGBM
classifier as the main tabular model, and an LSTM as the main temporal comparison
model.

The main scientific test is whether adding intermediate observational layers
improves forecasting skill relative to simpler baselines. This will be evaluated
not only through overall metrics, but through structured ablation of source blocks,
calibration assessment, and feature attribution using SHAP. Stretch analyses may
include PCA for latent-structure inspection, while more ambitious temporal or
spatial architectures such as TFT or GNN will remain explicitly out of core scope
unless time permits.

### Attribution

This project is not implementing a single existing published architecture verbatim.
Instead, it combines several ideas from prior work:

- probabilistic space weather forecasting using solar and L1 observations
  (Tasistro-Hart; 2021)
- gray-box regional perturbation motivation and NOAA product calibration baseline
  (Camporeale; 2020) (Camporeale; 2025)
- probabilistic solar wind regime context (Camporeale; 2020)
- extended lead-time and uncertainty motivation from solar-wind ensemble
  forecasting (Billcliff; 2026)
- uncertainty quantification as a first-class output (Abduallah; 2024) (Chen; 2025)
- LSTM sequence modeling for geomagnetic time series (Hochreiter; 1997)
  (Collado-Villaverde; 2021)

The specific multimodal fusion design, multi-layer source ablation emphasis, and
direct observational pipeline framing are the project's own synthesis.

---

## Technical Implementation Details

### Dataset

The dataset is a multimodal 1-minute feature matrix spanning 2015–2024 and
integrating compact solar context, OMNI L1 solar wind, GOES GEO measurements, LEO
context, and SuperMAG ground observations processed using the standardized
SuperMAG pipeline (Gjerloev; 2012). The current pipeline supports station-level
targets for ABK, BJN, and TRO, with the option to add more stations if scope and
data quality permit. OMNI data quality flags and gap characteristics have been
incorporated following documented best practices for forecasting applications
(Vokhmyanin; 2019) (Milan; 2022). The time span is large enough for chronological
separation into train (2015–2021), validation (2022–2023), and held-out test
(2024) groups, with 2024 reserved as the primary evaluation period.

Current feature columns include:

- **L1 / OMNI:** `omni_bx_gse`, `omni_by_gsm`, `omni_bz_gsm`, `omni_f`,
  `omni_vx`, `omni_proton_density`, `omni_pressure`, `omni_sym_h`, `omni_al`,
  `omni_au`; rolling means and standard deviations at 10-minute and 30-minute
  windows for `omni_bz_gsm`, `omni_vx`, and `newell_phi`; per-variable
  missingness and forward-fill flags
- **GEO / GOES:** `goes_bz_gsm`, `goes_bt`, `goes_bx_gsm`, `goes_by_gsm`,
  `goes_satellite`, `goes_missing_flag`
- **LEO:** sector-level precipitation context for high-latitude, mid-latitude,
  dayside, and nightside zones, each with a value, `_decay_age`, `_is_fresh`, and
  `_count` field
- **Ground / SuperMAG:** per-station B-field components (`_b_e`, `_b_n`, `_b_z`),
  $dB/dt$ magnitude, $dBe/dt$, $dBn/dt$, and `_dbdt_missing_flag` for ABK, BJN,
  and TRO
- **Cyclical time:** `ut_sin`, `ut_cos`, `doy_sin`, `doy_cos`
- **Physics-derived:** `newell_phi` and its rolling statistics
- **Quality / provenance:** `year`, `month`, aggregate missingness flags
  (`l1_any_missing`, `geo_any_missing`)

Solar data (Task 0) is not yet present in the feature matrix and will be added
prior to modeling.

### Feature Extraction

Feature extraction includes:

- time alignment to a common 1-minute grid
- rolling statistics over recent windows (existing: 10-minute and 30-minute;
  to be extended to 60-minute and 120-minute windows for ground-history and
  solar wind blocks)
- freshness and decay indicators for intermittent data sources (LEO sector
  pattern already implemented; to be audited and extended)
- compact solar-context engineering (F10.7, X-ray flux, flare class encoding,
  minutes-since-last-M-plus-flare decay feature — Task 0)
- ground-history features (trailing max, std of $dB/dt$ per station over
  60-minute and 120-minute lookback windows)
- optional substorm-catalog summary variables (minutes-since-last-onset per
  catalog, onset density in trailing 3-hour window, multi-catalog agreement score)
- probabilistic regime-context outputs from the auxiliary Model 4 appended as
  features to Models 5 and 6
- leakage audit: variables that encode near-contemporaneous geomagnetic response
  (e.g., `omni_al`, `omni_au`, `omni_sym_h` at the target timestamp rather than
  lagged) will be assessed and removed or lagged explicitly before modeling

### Algorithms

The core algorithm set is:

- **Persistence baseline:** operational lower bound
- **Climatology baseline:** probability reference for Brier Skill Score
- **Logistic Regression:** interpretable linear benchmark
- **Regime-context model:** auxiliary probabilistic state representation
- **LightGBM:** primary nonlinear tabular model
- **LSTM:** primary temporal model with explicit sequence memory (Hochreiter; 1997)

Stretch algorithms, only if time permits, include:

- **Temporal Fusion Transformer (TFT):** mixed-frequency temporal modeling
- **Graph Neural Network (GNN):** spatial modeling across stations
- **PCA:** latent-structure analysis rather than predictive modeling

The model choices are justified by the structure of the data: mixed source types,
intermittent missingness, class imbalance, and strong temporal dependence favor
tree-based baselines and sequence models over more fragile high-complexity
architectures at the initial stage. The broader literature on AI-based Earth
science models confirms that this escalating-complexity design pattern is robust
for multi-domain sensor fusion problems (Ziheng; 2022) (Bodnar; 2025).

---

## Success Criteria

### Quantitative Metrics

Project success will be defined using:

- **Heidke Skill Score (HSS):** primary operational classification metric
- **ROC AUC:** discrimination
- **Precision / Recall:** rare-event performance
- **Precision-Recall curves:** threshold-sensitive evaluation under class imbalance
- **Brier Score and Brier Skill Score (BSS):** probabilistic performance relative
  to climatology
- **Reliability diagrams:** calibration quality

### Expected Success

A successful outcome is:

- LightGBM clearly outperforming persistence and logistic regression on the
  held-out 2024 test set
- calibrated probabilistic forecasts with positive Brier Skill Score relative to
  climatology, addressing gaps identified in operational product evaluations
  (Camporeale; 2025)
- measurable contribution from at least some non-L1 source blocks in ablation,
  supporting or refuting the system-based fusion hypothesis
- a clear conclusion about whether Sun, GEO, and LEO context provide real
  operational value for one-hour-ahead regional $dB/dt$ forecasting, contributing
  to the evidence base called for in current space weather modeling reviews
  (Ferdousi; 2025) (Keebler; 2025) (Space Weather Review; 2025)

---

## Tasks

### Task 0: Data Refinement
- **Description:** Finalize the multimodal feature matrix before formal modeling.
  - add a compact solar context block: F10.7 (daily, forward-filled with staleness
    flag), GOES X-ray flux 1–8 Å (1-minute), flare-class ordinal encoding, and a
    `minutes_since_last_Mplus_flare` decay feature following the same decay-age
    pattern already used in the LEO sector columns
  - add additional SuperMAG stations if feasible (expanding beyond ABK, BJN, TRO)
  - audit missingness and freshness flags across all source domains, paying
    particular attention to GOES satellite transitions (`goes_satellite` column)
    and LEO orbital coverage gaps (`*_is_fresh` flag distributions)
  - assess `omni_al`, `omni_au`, and `omni_sym_h` for target leakage — these
    encode near-contemporaneous geomagnetic response and should be used only in
    lagged form in the final feature matrix
  - finalize event label thresholds (candidate values: 6, 10, 20 nT/min) and
    compute class balance at each threshold across years and stations
  - verify timestamp alignment across all source domains and confirm that
    forward-fills do not introduce cross-period leakage at the train/validation
    and validation/test boundaries
  - optionally add substorm catalog summary features: `minutes_since_last_onset`
    (per catalog), `onset_density_3hr`, and `multicatalog_agreement_score` for
    use as context features and sample weights

### Task 1: Exploratory Data Analysis
- **Description:** Characterize the dataset before modeling and identify
  constraints, failure modes, and physically meaningful patterns.
  - coverage and null rates by source, column, and station; flag columns with
    persistent missingness or implausible values
  - event rates by threshold (6, 10, 20 nT/min) and by year, season (DOY), and
    UT; confirm class imbalance ratios for each target definition
  - inter-station correlation of ABK, BJN, and TRO $dB/dt$ to understand whether
    stations can be pooled or require station-specific models
  - feature distributions for key L1 variables (Bz, Vx, Pdyn, Newell $\Phi$,
    SYM-H, AL), GEO block (GOES Bt, Bz), LEO sector values, and ground-history
    statistics, colored by event vs. non-event label
  - solar block sanity checks after Task 0: cadence audit for F10.7 and X-ray
    flux, staleness distribution for forward-filled daily solar indices
  - chain plots for 3–5 representative storm intervals showing the full
    Sun/L1/GEO/LEO/ground sequence with substorm catalog onset times overlaid as
    vertical markers
  - autocorrelation of $dB/dt$ magnitude per station to understand temporal
    persistence and inform lookback window design
  - GOES satellite flag inspection: characterize whether the `goes_satellite`
    transition creates measurement discontinuities that require encoding or
    filtering

### Task 2: Outline Models and Studies
- **Description:** Define the full model ladder, evaluation plan, and supporting
  analyses before implementation.

#### Model 1: Persistence Baseline
- **Description:** Predict that the future disturbance label at $t+60$ matches
  the current or most recent observed label.
- **Model:** Persistence rule
- **Model Reasoning:** Establishes the minimum operational baseline. Any trained
  model that fails to beat persistence provides no operational value.
- **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score
- **Success Criteria:** All learned models should exceed persistence on the
  held-out test set

#### Model 2: Climatology Baseline
- **Description:** Predict the unconditional event probability derived from the
  training period event rate.
- **Model:** Climatology (constant forecast equal to training-set base rate)
- **Model Reasoning:** Required reference for Brier Skill Score. BSS is defined
  relative to climatology, so this must be explicitly computed and reported.
- **Evaluation Metric:** Brier Score (absolute); denominator for BSS of all other
  models
- **Success Criteria:** Serves as the probabilistic calibration floor

#### Model 3: Logistic Regression
- **Description:** Linear probabilistic classifier on the full engineered feature
  matrix with L2 regularization.
- **Model:** Logistic Regression (`sklearn`, `solver='saga'`,
  `class_weight='balanced'`, regularization strength tuned on validation HSS)
- **Model Reasoning:** Establishes the linear ceiling and provides an interpretable
  coefficient-level view of which features contribute. If LightGBM dramatically
  outperforms logistic regression, it confirms that nonlinear interactions are
  necessary for this problem.
- **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score, Precision/Recall at
  HSS-optimal threshold
- **Success Criteria:** Outperform persistence (Model 1) and provide a stable
  linear comparison point for Models 5 and 6

#### Model 4: Regime-Context Model
- **Description:** Infer solar wind or geomagnetic regime probabilities from
  OMNI-derived features and append the resulting soft probability vector as context
  features to Models 5 and 6.
- **Model:** Gaussian Mixture Model (unsupervised) or weakly supervised LightGBM
  classifier with regime labels derived from SYM-H thresholds and solar wind speed
  percentiles; inspired by Camporeale et al. probabilistic solar wind classification
  (Camporeale; 2020)
- **Model Reasoning:** Regime context encodes system state — quiet, CIR-driven,
  CME-driven, recovery — explicitly, rather than forcing the main classifier to
  infer it implicitly from instantaneous solar wind values. The soft probability
  output also supports stratified evaluation of model skill by storm driver type,
  which is scientifically informative for the source ablation analysis.
- **Evaluation Metric:** Silhouette score or BIC for the unsupervised path;
  downstream ΔHSS of Models 5 and 6 when regime features are included vs. excluded
- **Success Criteria:** Regime probabilities produce ΔHSS > 0.02 in the main
  model ablation, or provide scientifically meaningful stratification of forecast
  errors by event type

#### Model 5: LightGBM Main Model
- **Description:** Primary nonlinear classifier on the fused Sun/L1/GEO/LEO/ground
  feature matrix including rolling statistics, physics-derived features (Newell
  $\Phi$), and regime-context probabilities from Model 4. False-positive reduction
  is an explicit design goal.
- **Model:** LightGBM `LGBMClassifier`
- **Model Reasoning:** Best-in-class for mixed-frequency tabular data with class
  imbalance and natively handles missing values, which is directly relevant given
  the LEO freshness flags and GOES missing indicators already in the dataset.
  Interpretable via SHAP for source attribution. Computationally efficient enough
  to support iterative threshold and weight tuning. Consistent with model choices
  in recent operational space weather forecasting literature (Ferdousi; 2025)
  (Keebler; 2025).
- **False-positive mitigation strategy:**
  - `scale_pos_weight` tuned in range 5–20, not set to the full imbalance ratio
    which would aggressively maximize recall at the cost of precision
  - decision threshold swept on validation Precision-Recall curve; selected at
    maximum HSS rather than defaulted to 0.5
  - isotonic probability calibration post-training (`CalibratedClassifierCV`),
    addressing calibration gaps documented in operational product evaluations
    (Camporeale; 2025)
  - evaluate on HSS as primary tuning metric rather than accuracy or AUC
- **Evaluation Metric:** HSS (primary), ROC AUC, Brier Skill Score,
  Precision/Recall at operational threshold, reliability diagram, confusion matrix
- **Success Criteria:** HSS > 0.35 on 2024 test set; BSS > 0.10 relative to
  climatology; Precision > 0.50 at HSS-optimal threshold; outperforms all baseline
  models (Models 1–3)

#### Model 6: LSTM
- **Description:** Temporal sequence model trained on fixed-length lookback windows
  for one-hour-ahead binary classification, without relying on hand-engineered lag
  statistics.
- **Model:** LSTM with dropout and recurrent dropout (Hochreiter; 1997); trained
  with focal loss (γ = 2) to concentrate gradient updates on rare positive events
- **Model Reasoning:** LightGBM operates on hand-crafted lag statistics (10-minute
  and 30-minute rolling means and standard deviations). An LSTM learns the temporal
  structure end-to-end and may capture higher-order dependencies — for example, the
  rate of change of southward Bz excursion, or the temporal shape of a pressure
  impulse — that fixed-window statistics miss. Prior work confirms that LSTM
  architectures extract useful temporal structure from solar wind sequences for
  geomagnetic forecasting (Collado-Villaverde; 2021) (Arxiv LSTM; 2025).
- **Evaluation Metric:** Same as Model 5; additionally compare against Model 5
  using DeLong test on AUC to assess statistical significance of any skill
  difference
- **Success Criteria:** HSS ≥ Model 5 on 2024 test set (demonstrates that sequence
  modeling adds skill beyond tabular lag features); informative even if it does not
  exceed LightGBM, by characterizing whether temporal memory is the binding
  constraint

#### Ablation Study
- **Description:** Retrain Model 5 (LightGBM) while removing one feature block at
  a time and measuring the change in held-out validation skill.
- **Motivation:** Quantify the marginal value of each observational layer in the
  Sun-to-ground chain. If the project argues for a system-based approach, it should
  actually test the system rather than asserting its value from architecture alone.
- **Reasoning:** The ablation directly tests the central scientific question of
  this project: whether GEO, LEO, and solar context provide information about
  regional ground response beyond what L1 alone carries.
- **Blocks to Ablate:**
  - solar block (F10.7, X-ray flux, flare features — after Task 0)
  - L1 / OMNI block (Bz, Vx, Pdyn, Newell $\Phi$, SYM-H, AL)
  - GEO block (GOES Bt, Bz, particle flux)
  - LEO block (sector precipitation context, freshness, decay age)
  - ground-history block (trailing $dB/dt$ statistics per station)
  - regime-context block (soft probabilities from Model 4)
  - substorm catalog block (onset timing and density features, if included)
  - rolling-statistics block (10-minute and 30-minute means and standard
    deviations — tests whether raw instantaneous values alone are sufficient)
- **Evaluation Metric:** ΔHSS, ΔAUC, ΔPrecision, ΔRecall relative to the full
  Model 5

#### Stretch Goals
- **TFT:** Temporal Fusion Transformer for mixed-frequency attention-based temporal
  modeling, activated if ≥ 200 labeled storm intervals are available in training
- **GNN:** Graph Neural Network for spatial message passing across SuperMAG
  stations, encoding geographic and magnetic latitude proximity
- **PCA:** latent-structure analysis to assess feature-family separability and
  support interpretability rather than prediction

### Task 3: Build Models
- **Description:** Implement Models 1–6 using the finalized train/validation split
  and a common preprocessing pipeline.
  - train all core models on the same 2015–2021 chronological split
  - calibrate probabilistic outputs where appropriate (isotonic calibration for
    Models 3, 5, and 6)
  - store out-of-sample predictions, decision thresholds, and hyperparameter
    metadata for consistent evaluation across models
  - document modeling assumptions, lookback window choices, and class-weight tuning
    decisions explicitly

### Task 4: Evaluate Models
- **Description:** Compare discrimination, calibration, and operational usefulness
  across all core models on the 2024 held-out test set.
  - confusion matrix at the HSS-optimal operating threshold for each model
  - ROC and Precision-Recall curves overlaid across all models
  - reliability (calibration) diagram for Models 3, 5, and 6
  - Brier Score and Brier Skill Score relative to Model 2 climatology
  - DeLong test for AUC differences between Model 5 and Model 6
  - GEM benchmark intervals as a qualitative stress-test, not used for threshold
    tuning (following the evaluation design rationale in Camporeale; 2025 and
    Thaker; 2025)
  - note whether validation-period threshold selection holds on the 2024 test set
    or shows signs of threshold drift

### Task 5: Ablation Study
- **Description:** Run structured leave-one-block-out experiments on Model 5 across
  the validation set.
  - quantify how much each source block contributes to HSS, AUC, and Precision
  - identify which blocks improve recall vs. precision independently
  - assess whether the compact solar context block (Task 0) provides measurable
    skill beyond what the L1 block already captures — directly testing the
    forecasting value of near-Sun observations claimed in the heliophysics strategy
    literature (Heliophysics Decadal Survey; 2025) (Billcliff; 2026)
  - assess whether GEO and LEO blocks add value beyond L1 alone, which is the
    central scientific hypothesis of the project

### Task 6: Comprehensive Analysis
- **Description:** Interpret model behavior, summarize what the system learned, and
  characterize failure modes.
  - SHAP global feature importance from Model 5 across the full feature matrix
  - SHAP dependence plots for the top 5 features (expected: Newell $\Phi$, rolling
    Bz minimum, SYM-H, AL, ground history); inspect whether physics-consistent
    threshold behavior emerges (e.g., strong SHAP effect for Bz below −5 nT)
  - compare SHAP feature rankings across ABK, BJN, and TRO to assess whether
    station location modifies the relative importance of different source layers
  - characterize false-positive and false-negative tendencies by storm driver type
    using regime labels from Model 4 (do CME-driven events produce fewer false
    negatives than CIR-driven events?)
  - summarize whether intermediate orbital layers (GEO, LEO) add measurable
    operational value beyond L1-only input, framed as a direct answer to the
    central scientific question
  - optional PCA latent-structure analysis if time permits: project feature
    families into 2-D, overlay event labels and regime labels, assess separability

---

## References

Abduallah, Y., et al. (2024). Predicting geomagnetic indices with uncertainty
quantification. *IIS*.

Billcliff, S., et al. (2026). Extended lead-time geomagnetic storm forecasting
with solar wind ensembles and machine learning. *Space Weather*.

Bodnar, C., et al. (2025). AI-based Earth science models. *Nature*.

Camporeale, E., et al. (2020). Gray-box probabilistic geomagnetic storm
forecasting. *JGR Space Physics*.

Camporeale, E. (2021). Machine learning in space weather. *Space Weather Workshop*.

Camporeale, E., et al. (2025). Evaluation of NOAA operational geomagnetic
forecast. *Space Weather*.

Carter, B. A., et al. (2016). GIC around the world. *JGR Space Physics*.

Chen, Y., et al. (2025). GeoDGP: deep Gaussian process framework for geomagnetic
disturbance prediction. *Space Weather*.

Collado-Villaverde, A., et al. (2021). Deep neural network forecasting of
geomagnetic activity. *Space Weather*.

Ferdousi, B. (2025). Current state of geomagnetic disturbance forecasting:
overview of physics and ML approaches. *AFRL*.

Fuselier, S. A., et al. (2010). Antiparallel magnetic reconnection rates at the
Earth's magnetopause. *JGR Space Physics*.

Gjerloev, J. W. (2012). The SuperMAG data processing technique. *JGR Space Physics*.

Heliophysics Decadal Survey (2025). *National Academies Press*.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
Computation*.

Keebler, T., et al. (2025). Space weather model advances. *Space Weather*.

Kim, H., et al. (2024). Localized magnetopause erosion at GEO via reconnection.
*GRL*.

Milan, S. E., et al. (2022). L1 and OMNI data location considerations for
forecasting. *JGR Space Physics*.

Mostafa, A., et al. (2025). Machine learning for geomagnetic storm effects on
satellites. *Scientific Reports*.

Nagai, T., et al. (2023). Magnetic reconnection in the near-Earth magnetotail.
*JPR Space Physics*.

Nature Communications (2024). Global magnetospheric convection driven by dayside
magnetic reconnection.

Newell, P. T., et al. (2007). A nearly universal solar wind–magnetosphere coupling
function inferred from 10 magnetospheric state variables. *JGR Space Physics*.

Shen, X.-C., et al. (2023). Energetic electron flux dropouts at LEO and
ionospheric projection. *JGR Space Physics*.

Space Weather Review (2025). Annual review of advances in space weather science
and forecasting.

SWAG (2024). Space weather user needs assessment. *Space Weather Advisory Group*.

Tasistro-Hart, A., et al. (2021). Probabilistic geomagnetic storm forecasting via
deep learning. *JGR Space Physics*.

Thaker, D., et al. (2025). Evaluation of AI for space weather applications.
*Space Habitat*.

Vokhmyanin, M. V., et al. (2019). OMNI data quality assessment for space weather
applications. *AGU Space Weather*.

Zheng, Y., et al. (2013). Space weather forecasting: propagation and evolution of
CMEs. *Space Weather*.

Ziheng, S. (2022). Machine learning in Earth science. *Computers and Geosciences*.

Arxiv (2025). Cosmic ray LSTM for geomagnetic storm prediction.

---

# Project Flow
* Use OSEMI (obtain,scrub,explore,model,interpret) as general narative flow
    * **obtain:** gather data from relevant sources
    * **scrub:** clean data
    * **explore:** find patterens
    * **model:** train models to predict and forecast
    * **interpret** explain model and results
1. Past research and offical org priorities to determine sources
2. Based on relevant sources from *1.* create data pipeline to collect and clean data
3. Use data from pipeline in *2.** perform exploratory data analysis(EDA)
4. Relevant Sources + EDA results motivate Problem 1 and Model 1
5. Limitations/Issues from Model 1 motivate Problem 2 and Model 2
6. Limitations/Issues from Model 2 motivate Problem 3 and Model 3
7. Comprehensive Analysis on results from all models
8. Use analysis results from *7.* to generate Interpretation
9. Generate final remarks/conclusions based on intrepretation in *8.*

- **one main task,**
- **one optional context model,**
- **one optional temporal model,**
- **three strong analyses.**

