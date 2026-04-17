# Title: Physics-Informed Forecasting of Regional Geomagnetic Perturbations

# Author: Scotty Rogers

## Abstract
Coronal Mass Ejections (CMEs) and high-speed solar wind streams drive dynamic deformations of Earth’s magnetosphere, triggering substorms and large magnetic perturbations that can produce Geomagnetically Induced Currents (GICs). These disturbances pose significant risks to power systems through transformer damage, voltage instability, and broader infrastructure disruption. Despite this threat, most operational forecasting remains focused on broad global geomagnetic indices or computationally expensive physics-based models, limiting the availability of low-latency, region-specific hazard guidance. Prior gray-box work has shown that machine learning can improve probabilistic estimates of regional ground magnetic perturbations, but those systems often remain coupled to expensive simulation pipelines rather than operating as direct end-to-end observational forecasters (Camporeale; 2020). 

This study proposes a physics-informed machine learning framework for forecasting regional ground magnetic perturbations, expressed as \(dB/dt\) in nT/min, using a continuous Sun-to-ground observation chain. The feature pipeline fuses compact solar context, upstream solar wind observations at L1, magnetospheric measurements at GEO, ionospheric context at LEO, and regional ground magnetic response from SuperMAG stations. The central scientific question is whether intermediate observational layers between upstream solar forcing and ground response improve one-hour-ahead regional forecasting beyond L1-only baselines. The project is organized using the OSEMI framework—obtain, scrub, explore, model, interpret—and progresses through an escalating model ladder from binary classification to temporal sequence learning and finally to continuous-value regression. 

---

## Project Flow
This project follows the **OSEMI** narrative flow:

* **Obtain:** gather data from scientifically and operationally relevant sources
* **Scrub:** clean, align, and quality-control the data
* **Explore:** identify patterns, missingness, and physically meaningful structure
* **Model:** train models to classify and forecast regional disturbances
* **Interpret:** explain model behavior, source contributions, and scientific implications

The detailed project flow is:

1. Use past research and official organizational priorities to determine relevant sources.
2. Build a data pipeline to collect, align, and clean those sources.
3. Use the output of the pipeline to perform exploratory data analysis.
4. Use relevant sources plus EDA results to motivate **Problem 1** and **Model 1**.
5. Use limitations or issues from **Problem 1 / Model 1** to motivate **Problem 2** and **Model 2**.
6. Use limitations or issues from **Problem 2 / Model 2** to motivate **Problem 3** and **Model 3**.
7. Perform comprehensive analysis across all models.
8. Use those results to generate interpretation.
9. End with final remarks and conclusions based on the interpretation.

This flow is important because it prevents the project from reading like “here are six random models I felt like training” and instead frames each model as a response to the limitations of the previous one. 

---

## Background
Space weather describes the chain of physical processes linking solar forcing, solar wind propagation, magnetospheric dynamics, ionospheric response, and ground magnetic perturbations. The coupling begins at the Sun, where flares and CMEs launch energetic plasma into the heliosphere. That plasma is then measured in situ at L1, roughly 1.5 million km upstream of Earth, providing the primary real-time driver signal used in most operational space weather systems. However, the L1 observation is only a single-point sample of a structured and time-varying solar wind front, and the propagation from L1 to Earth introduces additional uncertainty (Milan; 2022) (Vokhmyanin; 2019). 

Once the solar wind reaches the magnetosphere, dayside magnetic reconnection drives magnetopause erosion and controls the rate of energy entry into the coupled magnetosphere-ionosphere system. The Newell coupling function quantifies this energy transfer as a function of solar wind speed and IMF clock angle and has become a standard physics-informed feature in data-driven forecasting (Newell; 2007). Reconnection at the dayside magnetopause compresses the magnetosphere, deposits open magnetic flux in the tail lobes, and eventually triggers explosive tail reconnection events known as substorms, which in turn drive rapid ionospheric current variation and large ground magnetic perturbations (Fuselier; 2010) (Nagai; 2023) (Gjerloev; 2012). 

For electric power applications, the most operationally relevant environmental quantity is not simply a broad planetary index, but localized rapid magnetic field variation, because \(dB/dt\) is directly related to induced geoelectric fields and therefore to GIC risk. Reviews of GIC impacts have confirmed that local response can differ substantially from global storm summaries, making regional prediction an important applied data science problem (Carter; 2016). Recent heliophysics strategy documents also argue for more integrated Sun-to-Earth and system-level approaches rather than isolated single-domain studies, motivating the multi-layer architecture used in this project (Heliophysics Decadal Survey; 2025) (Ferdousi; 2025) (Keebler; 2025). 

---

## Problem Definition

### Overview
This project is structured around three connected modeling problems within a single Sun-to-ground forecasting pipeline. Each problem is motivated by the limitations of the previous one, creating a natural progression from interpretable tabular classification to temporal sequence learning and finally to continuous regional forecasting. 

### Problem 1: Binary Threshold Forecasting
**Goal:** Predict whether station-level or regional \(dB/dt\) will exceed a chosen threshold at a fixed lead time of 60 minutes.

**Primary model:** LightGBM

**Input:** Synchronized multimodal features at 1-minute cadence, including compact solar context, L1 solar wind drivers, GEO context, LEO context, and recent ground-history features.

**Output:** Calibrated probability of threshold exceedance, later converted into an operational alert decision.

### Problem 2: Regime / Context Classification
**Goal:** Infer solar wind or geomagnetic regime probabilities from OMNI-derived upstream features.

**Primary model:** Regime-context model

**Input:** OMNI and related physics-derived upstream features such as IMF, flow speed, pressure, and Newell coupling.

**Output:** Soft regime probabilities representing disturbance context such as quiet, CME-driven, CIR-driven, or recovery-like conditions.

This auxiliary problem is motivated by the fact that the main classifier may be forced to learn hidden system state implicitly from raw inputs. The regime model provides an interpretable probabilistic context layer that can be fed into downstream models. 

### Problem 3A: Sequence-Based Classification
**Goal:** Test whether explicit temporal memory improves one-hour-ahead binary forecasting relative to engineered lag statistics.

**Primary model:** \(LSTM_A\)

**Input:** Fixed-length multivariate lookback windows from the same Sun/L1/GEO/LEO/ground feature set.

**Output:** Calibrated probability of threshold exceedance at \(t+60\).

This problem is motivated by a limitation of Problem 1: LightGBM relies on engineered rolling statistics and may miss temporal structure that a sequence model can learn directly. 

### Problem 3B: Continuous Regional Forecasting
**Goal:** Predict future continuous \(dB/dt\) magnitude rather than only a threshold exceedance label.

**Primary model:** \(LSTM_B\)

**Input:** Fixed-length multivariate lookback windows from the fused observation chain.

**Output:** Continuous forecast of future regional \(dB/dt\), such as station-level magnitude or regional aggregate response at \(t+60\).

This problem is motivated by the limitations of binary classification. Once classification performance is understood, regression provides a more demanding and physically informative extension by asking not only whether a disturbance will happen, but how large it may be.

---

## Motivation

### Suitability for Machine Learning
This is a strong machine learning problem because the Sun-to-ground system is nonlinear, multiscale, and only partially observed, making simple analytical mappings insufficient while still offering abundant historical observational data. Prior work has shown that neural networks, probabilistic forecasting systems, and gray-box approaches can extract predictive skill from solar wind and geomagnetic observations even when first-principles forward modeling remains computationally expensive or incomplete (Tasistro-Hart; 2021) (Camporeale; 2020).

Machine learning is especially appropriate here because the forecast target is operational rather than purely explanatory: the practical goal is not to simulate every microphysical process, but to predict elevated risk of local ground disturbance with enough lead time and calibration quality to be useful. This aligns with broader arguments in Earth and space science that data-driven models can complement or partially replace slower simulation workflows when the objective is fast, decision-oriented prediction (Ziheng; 2022) (Bodnar; 2025) (Billcliff; 2026). 

### Data Availability
This project has access to a sufficiently large and diverse multimodal dataset spanning 2015–2024 at 1-minute cadence, integrating Sun, L1, GEO, LEO, and ground sources into a unified feature matrix. The existing pipeline already supports millions of timestamped rows and can be split chronologically into train, validation, and held-out test periods, which is essential for avoiding leakage in strongly autocorrelated time-series problems. OMNI data quality, gap structure, and interpolation issues are documented in the literature and are already handled through missingness and provenance flags in the feature matrix (Vokhmyanin; 2019) (Milan; 2022).

There is also a viable source of supervised ground truth because the target labels are derived directly from observed ground magnetic perturbations processed through the SuperMAG pipeline rather than inferred proxy classes alone (Gjerloev; 2012). 

### Importance and Novelty
The importance of the problem is straightforward: regional \(dB/dt\) forecasting is directly relevant to GIC risk and therefore to grid operations, while currently available broad geomagnetic indices are often too coarse for local hazard characterization (Carter; 2016). The novelty of this project lies in combining a system-based Sun/L1/GEO/LEO/ground fusion architecture with regional \(dB/dt\) forecasting in a direct observational pipeline, rather than relying solely on L1 inputs or using machine learning only as a post-processing layer over large geospace simulations (Camporeale; 2020) (Tasistro-Hart; 2021) (Billcliff; 2026). 

---

## Literature Review

### Prior Research
Prior machine learning studies in space weather have largely focused on forecasting global or quasi-global indices such as Dst, Kp, SYM-H, or related storm measures using solar wind observations from L1. Tasistro-Hart et al. showed that combining L1 data with solar-disk observations can support probabilistic geomagnetic storm forecasting, but also found that direct solar inputs alone did not fully solve storm-onset prediction and that L1 observations remained dominant (Tasistro-Hart; 2021). Collado-Villaverde et al. similarly demonstrated that deep neural network architectures trained on solar wind sequences can produce competitive forecasts of global geomagnetic activity (Collado-Villaverde; 2021).

Camporeale et al. developed a gray-box approach for probabilistic regional ground magnetic perturbation forecasting by enhancing the NOAA operational Geospace model with machine learning, demonstrating the value of combining physical structure with data-driven prediction. However, that work still depends on the upstream simulation stack rather than operating as a direct observational forecaster (Camporeale; 2020). Billcliff et al. extended lead-time geomagnetic storm forecasting by using solar-wind ensembles and machine learning, arguing that near-Sun context becomes most useful when transformed into propagation-aware future solar-wind scenarios rather than simply appended as raw solar indicators (Billcliff; 2026). 

Related work on solar wind classification has shown that probabilistic context vectors can be useful intermediate representations of system state. This idea motivates the regime-context stage in the present project. More broadly, recent reviews of geomagnetic disturbance forecasting note that direct observational fusion across multiple layers of the coupled system remains underexplored relative to single-domain L1-to-index mappings (Ferdousi; 2025) (Keebler; 2025). 

### Comparison to Prior Work
This project differs from most prior work in four main ways:

1. It targets **regional ground magnetic perturbation exceedance** rather than only a global storm index.
2. It explicitly spans **multiple observational layers** of the coupled Sun–Earth system.
3. It uses a **direct observational pipeline** rather than an ML wrapper around an expensive simulation.
4. It includes a **progressive modeling ladder**, where tabular classification, context inference, sequence classification, and regression are treated as connected stages rather than unrelated experiments. 

The project therefore does not simply ask which model has the highest score. It asks what each modeling stage learns, what limitations remain, and whether additional observational layers actually contribute meaningful skill. 

---

## Approach

### Methodology
The project will construct a unified 1-minute multimodal feature matrix covering Sun, L1, GEO, LEO, and ground observations across 2015–2024. The modeling plan follows the OSEMI framework and is explicitly progressive:

* **Problem 1 / Model family:** threshold classification with LightGBM and baseline models
* **Problem 2 / Model family:** regime-context inference
* **Problem 3A / Model family:** sequence-based binary classification with \(LSTM_A\)
* **Problem 3B / Model family:** continuous forecasting with \(LSTM_B\)

This progression is deliberate. The goal is not to throw every architecture in the zoo at the data, but to use each model to answer a more refined version of the forecasting problem.

### Attribution
This project does not implement a single published architecture verbatim. Instead, it combines several ideas from prior work:

* probabilistic space weather forecasting using solar and L1 observations (Tasistro-Hart; 2021)
* gray-box regional perturbation motivation and operational calibration awareness (Camporeale; 2020)
* probabilistic solar wind regime context
* extended lead-time and uncertainty motivation from solar-wind ensemble forecasting (Billcliff; 2026)
* sequence learning for temporal dependencies (Hochreiter and Schmidhuber; 1997)

The multimodal fusion design, OSEMI framing, progressive problem structure, and source-ablation emphasis are the project’s own synthesis.

---

## Technical Implementation Details

### Dataset
The dataset is a multimodal 1-minute feature matrix spanning 2015–2024 and integrating compact solar context, OMNI L1 solar wind, GOES GEO measurements, LEO context, and SuperMAG ground observations. The current pipeline supports station-level targets for ABK, BJN, and TRO, with the option to add more stations if scope and data quality permit. The time span is large enough for chronological separation into train (2015–2021), validation (2022–2023), and held-out test (2024) groups. 

### Feature Extraction
Feature extraction includes:

* time alignment to a common 1-minute grid
* rolling statistics over recent windows
* freshness and decay indicators for intermittent data sources
* compact solar-context engineering
* ground-history features
* optional substorm-catalog summary variables
* probabilistic regime-context outputs appended as downstream features
* explicit leakage audits for geomagnetic-response-like upstream variables

This design supports both tabular models and sequence models while preserving source provenance and missingness information. 

### Algorithms
The core algorithm set is:

* **Model 1:** Persistence baseline
* **Model 2:** Climatology baseline
* **Model 3:** Logistic Regression
* **Model 4:** Regime-context model
* **Model 5:** LightGBM
* **Model 6A:** \(LSTM_A\) sequence classifier
* **Model 6B:** \(LSTM_B\) sequence regressor

Stretch algorithms, if time permits:

* Temporal Fusion Transformer (TFT)
* Graph Neural Network (GNN)
* PCA for latent-structure analysis 

---

## Success Criteria

### Quantitative Metrics
Success will be measured using:

**Classification**
* Heidke Skill Score (HSS)
* ROC AUC
* Precision / Recall
* Precision-Recall curves
* Brier Score and Brier Skill Score
* Reliability diagrams

**Regression**
* RMSE
* MAE
* Pearson correlation
* event-focused error analysis during elevated-disturbance intervals

### Expected Success
A successful outcome is:

* LightGBM clearly outperforming persistence and logistic regression on the held-out 2024 test set
* \(LSTM_A\) providing a meaningful comparison to LightGBM by testing whether learned temporal memory adds classification skill
* \(LSTM_B\) demonstrating that the framework can extend beyond binary alerts into continuous regional forecasting
* measurable contribution from at least some non-L1 source blocks in the ablation study
* a clear conclusion about whether Sun, GEO, and LEO context provide real operational value for one-hour-ahead regional forecasting 

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

Camporeale, E ,et al. (2017) Journal of Geophysical Research: Space Physics. Classification of Solar Wind With Machine Learning

---

## Tasks

### Task 0: Obtain + Scrub
**Description:** Finalize the multimodal feature matrix before formal modeling.

* add a compact solar context block
* add additional SuperMAG stations if feasible
* audit missingness and freshness flags across all source domains
* assess potential leakage variables and lag or remove them explicitly
* finalize event thresholds and regression targets
* verify timestamp alignment across all source domains

### Task 1: Explore
**Description:** Characterize the dataset before modeling and identify constraints, failure modes, and physically meaningful patterns.

* coverage and null rates by source and station
* event rates by threshold, year, season, and UT
* inter-station correlation structure
* feature distributions for key source blocks
* solar block cadence and staleness audit
* chain plots for representative intervals
* autocorrelation analysis to inform lookback design
* GOES transition artifact inspection

### Task 2: Model
**Description:** Define and train the full model ladder.

#### Model 1: Persistence Baseline
* **Description:** Predict that the future disturbance label at \(t+60\) matches the current or most recent observed label.
* **Model:** Persistence rule
* **Reasoning:** Establishes the minimum operational baseline.
* **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score
* **Success Criteria:** All learned models should exceed persistence

#### Model 2: Climatology Baseline
* **Description:** Predict the unconditional event probability derived from the training period event rate.
* **Model:** Climatology
* **Reasoning:** Required reference for Brier Skill Score.
* **Evaluation Metric:** Brier Score
* **Success Criteria:** Serves as the probabilistic floor

#### Model 3: Logistic Regression
* **Description:** Linear probabilistic classifier on the engineered feature matrix.
* **Model:** Logistic Regression
* **Reasoning:** Establishes the linear benchmark and interpretable baseline.
* **Evaluation Metric:** HSS, ROC AUC, Brier Skill Score, Precision/Recall
* **Success Criteria:** Outperform persistence and provide a stable linear comparison

#### Model 4: Regime-Context Model
* **Description:** Infer solar wind or geomagnetic regime probabilities from OMNI-derived features and append them as context features to downstream models.
* **Model:** GMM or weakly supervised classifier
* **Reasoning:** Encodes hidden system state explicitly rather than forcing downstream models to infer it from raw upstream inputs alone.
* **Evaluation Metric:** Separability diagnostics and downstream skill improvement
* **Success Criteria:** Improves calibration or downstream performance, or provides scientifically meaningful error stratification

#### Model 5: LightGBM Main Model
* **Description:** Primary nonlinear classifier on the fused Sun/L1/GEO/LEO/ground feature matrix.
* **Model:** LightGBM
* **Reasoning:** Best fit for mixed-frequency tabular data with class imbalance, nonlinear interactions, and missingness.
* **Evaluation Metric:** HSS (primary), ROC AUC, Brier Skill Score, Precision/Recall, reliability diagram
* **Success Criteria:** Best overall classification model on the held-out test set

#### Model 6A: \(LSTM_A\) Classification Model
* **Description:** Sequence model for one-hour-ahead binary classification using fixed lookback windows.
* **Model:** LSTM classifier
* **Reasoning:** Tests whether learned temporal memory improves on engineered lag statistics used by LightGBM.
* **Evaluation Metric:** Same as LightGBM
* **Success Criteria:** Matches or exceeds LightGBM, or clearly reveals whether temporal sequence learning adds classification value

#### Model 6B: \(LSTM_B\) Regression Model
* **Description:** Sequence model for one-hour-ahead continuous regional \(dB/dt\) forecasting.
* **Model:** LSTM regressor
* **Reasoning:** Extends the project from “will a disturbance happen?” to “how large will it be?” and provides a more physically informative downstream task.
* **Evaluation Metric:** RMSE, MAE, correlation, event-focused error analysis
* **Success Criteria:** Produces physically reasonable continuous forecasts and demonstrates feasibility of moving beyond threshold alerts

### Task 3: Evaluate
**Description:** Compare discrimination, calibration, and operational usefulness across all core models.

* confusion matrices for classification models
* ROC and Precision-Recall curves
* reliability diagrams
* Brier Score and Brier Skill Score
* RMSE / MAE / correlation for regression
* GEM benchmark intervals as qualitative stress tests
* threshold stability between validation and test periods

### Task 4: Ablation Study
**Description:** Retrain the main classification model while removing one feature block at a time.

* solar block
* L1 / OMNI block
* GEO block
* LEO block
* ground-history block
* regime-context block
* substorm-catalog block
* rolling-statistics block

**Motivation:** Quantify the marginal value of each observational layer in the Sun-to-ground chain.

**Evaluation Metric:** \(\Delta\)HSS, \(\Delta\)AUC, \(\Delta\)Precision, \(\Delta\)Recall relative to the full model. 

### Task 5: Interpret
**Description:** Explain model behavior and summarize what the system learned.

* SHAP global feature importance
* SHAP dependence plots
* compare feature rankings across stations
* summarize false positives and false negatives by regime
* evaluate whether GEO and LEO add measurable value beyond L1
* optional PCA if time permits

---

## Final Remarks
The central contribution of this project is not just a better score on a single benchmark. It is a structured test of whether a system-based Sun/L1/GEO/LEO/ground observational chain improves regional geomagnetic forecasting, and whether that value holds across classification, context modeling, and continuous prediction. If nothing else, this project should at least answer whether the satisfying end-to-end arrow diagram corresponds to real predictive information or just to excellent aesthetic instincts. 
