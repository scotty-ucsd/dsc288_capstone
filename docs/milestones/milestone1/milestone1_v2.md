# DSC 288: Capstone Project
## Milestone 1: Abstract

**Title:** Physics-Informed Machine Learning for Regional Geomagnetic Disturbance Forecasting

**Author:** Scotty Rogers

---

### Project Goal

Apply physics-informed ML models to improve regional forecasting of hazardous geomagnetic perturbations ($|dB/dt|$) using an end-to-end multimodal dataset spanning Solar / L1 / GEO / LEO / Ground observation layers.

Regional $|dB/dt|$ forecasting is structured around two supervised learning tasks:

| Task | Question | Lead Time | Target Stations |
|---|---|---|---|
| **Classification** | Will $\|dB/dt\|$ exceed a hazard threshold at a SuperMAG station? (yes/no) | 60 minutes | ABK, BJN, TRO (Scandinavian auroral zone) |
| **Regression** *(extension)* | How intense will $\|dB/dt\|$ be at a target station at time $t + 60$? | 60 minutes | ABK, BJN, TRO |

**Classification Task** — *Primary deliverable*

| | Model | Notes |
|---|---|---|
| Baseline | Persistence + Climatology | Predict label/rate from recent history; required floor for all skill scores |
| Linear reference | Logistic Regression | Establishes linear ceiling; coefficient-level interpretability |
| **Primary** | **LightGBM** | Nonlinear tabular classifier; handles imbalance and missing data natively; SHAP-compatible |
| **Secondary** | **LSTM** | Sequence classifier; tests whether learned temporal structure adds skill beyond LightGBM's engineered lag features |

**Regression Task** — *Extension, time permitting*

| | Model | Notes |
|---|---|---|
| Baseline | Persistence | Predict $\|dB/dt\|_{t+60} = \|dB/dt\|_t$ |
| Baseline | Climatology mean | Predict training-set mean $\|dB/dt\|$; denominator for skill scores |
| **Primary** | **LSTM** | Sequence-to-scalar regression; same architecture as classification LSTM with output head swapped to predict continuous $\|dB/dt\|$ intensity |

---

### 3. Abstract

#### a) Background

Coronal mass ejections (CMEs) and high-speed solar wind streams drive dynamic deformations of Earth's magnetosphere, triggering substorms and geomagnetically induced currents (GICs). GICs pose measurable risks to electrical power grids, including voltage instability and transformer saturation, with documented economic consequences during major events such as the March 1989 Quebec blackout. Despite this threat, current operational space weather forecasting is largely limited to broad global indices such as Kp and Dst, which do not resolve the regional ground-level magnetic perturbations that matter most to grid operators.

Forecasting the rate of change of the geomagnetic field, $|dB/dt|$, is widely used as a proxy for GIC hazard because it captures the inductive forcing on ground infrastructure more directly than slowly varying storm indices. However, $|dB/dt|$ is spatially structured, temporally intermittent, and driven by a nonlinear chain of processes spanning solar eruptions, L1 solar wind conditions, magnetospheric state, and ionospheric currents. No single upstream measurement captures all of this variability, motivating a multimodal observational approach that spans Solar/L1/GEO/LEO/ground data layers.

This project applies physics-informed machine learning to forecast regional $|dB/dt|$ hazard one hour in advance using a fused dataset spanning Solar Cycle 24 and the rising phase of Cycle 25 (2015–2024).

---

#### b) Problem Definition

**Primary Task — Regional Hazard Classification:**
The primary forecasting problem is binary classification: given a window of multimodal space environment observations at time $t$, predict whether the magnitude of the horizontal geomagnetic field rate of change, $|dB/dt|$, will exceed a regionally calibrated hazard threshold at a target high-latitude SuperMAG ground station within a 60-minute lead time.

- **Inputs:** A fixed-width feature vector constructed from (1) GOES X-ray flux (solar driver), (2) OMNI 1-minute solar wind and interplanetary magnetic field data (L1), (3) GOES geostationary magnetic field observations (GEO), (4) Swarm-derived electron flux and field-aligned current proxies by MLT zone (LEO), (5) SuperMAG ground station magnetic field components and recent $|dB/dt|$ history (ground), and (6) physics-informed features including the Newell coupling function $\Phi$, rolling IMF $B_z$ statistics, and cyclical time encodings.
- **Outputs:** A calibrated probability that $|dB/dt|$ exceeds threshold $\tau$ at a target station within the next 60 minutes, plus a binary alert derived at the HSS-optimal decision threshold.

**Secondary Task (Extension, Time Permitting):**
An LSTM sequence model trained on the same feature streams will be evaluated against the primary classifier to determine whether learned temporal structure improves skill beyond hand-engineered lag statistics.

---

#### c) Motivation

**Suitability for Machine Learning:**
This is a well-suited ML problem for several reasons. The Sun-to-ground causal chain is nonlinear and multiscale: simple analytical mappings from solar wind to ground $|dB/dt|$ are known to fail in practice, particularly at high latitudes during substorm onset and CME-driven storm main phases. At the same time, the system is not purely stochastic — structured upstream drivers such as sustained southward IMF $B_z$ and elevated solar wind dynamic pressure are consistently associated with enhanced ground disturbance, providing predictive signal that ML models can exploit.

The operational goal is not to simulate every microphysical process, but to deliver calibrated risk estimates with sufficient lead time and regional specificity to support grid operator decisions. This makes the problem tractable for discriminative ML approaches (Camporeale, 2020; Sun et al., 2022). The physics-informed framing — embedding domain-derived features such as the Newell coupling function alongside raw observational inputs — improves generalization by encoding known driver-response relationships rather than requiring the model to rediscover them from data alone.

**Data Availability:**
The project dataset spans January 2015 through December 2024 (120 months, 118 successfully processed), covering both the declining phase of Solar Cycle 24 and the rising phase of Cycle 25. This range mitigates the quiet-time bias present in many existing studies that sample predominantly from solar minimum. The dataset fuses five source layers at one-minute cadence: GOES X-ray flux, OMNI solar wind (CDAWeb), GOES geostationary field (GEO), Swarm electron flux proxies by MLT zone (LEO), and SuperMAG ground station magnetic components and $|dB/dt|$ for three Scandinavian high-latitude stations (ABK, BJN, TRO). The resulting feature matrix contains approximately 5.2 million one-minute records with over 80 input columns, sufficient to support separate training, validation, and temporally held-out test sets (2024 reserved as the primary test year).

---

#### d) Literature Review

**Prior Research:**
Regional geomagnetic disturbance forecasting has been approached through both physics-based and data-driven methods. Camporeale et al. (2020) developed a gray-box classifier that enhances NOAA's operational Geospace model with a boosted tree, achieving improved threshold-exceedance prediction at three high-latitude stations; this work is the most direct methodological precedent for the present project. Chen et al. (2025) introduced GeoDGP, a deep Gaussian process model for probabilistic global $dB_H$ forecasting one hour ahead, providing a strong benchmark for probabilistic evaluation and demonstrating that ML approaches can outperform the Geospace and DAGGER operational models on storm sets. Billcliff et al. (2026) demonstrate that probabilistic storm forecasting can be extended to longer lead times by combining solar wind ensemble propagation with logistic regression, though their approach targets global storm indices rather than regional $|dB/dt|$. For evaluation methodology, Camporeale et al. (2025) document systematic calibration failures in operational NOAA solar flare forecasts when compared against simple persistence and climatology baselines, reinforcing the need for rigorous imbalance-aware verification in any operational space weather product.

**Comparison:**
Relative to prior work, this project advances in three directions: 

1. It uses a richer multimodal input architecture, spanning Solar/L1/GEO/LEO/ground layers, rather than relying on OMNI-only or simulation-assisted features. 

2. This targets regional $|dB/dt|$ threshold exceedance rather than smooth global indices ($Dst$, $Kp$, $dB_H$), which are less directly tied to GIC risk. 

3. Lastly, it covers a longer and more solar-active period than many existing studies, reducing quiet-time data dominance. The approach does not attempt to replace physics-based operational models; rather, it complements them by providing a data-driven, multimodal hazard classifier with calibrated probability outputs and source-attribution capability via SHAP.

---

#### e) Your Approach

**Methodology:**
The primary model is a LightGBM binary classifier trained on the fused feature matrix described above. LightGBM is selected for its native handling of class imbalance, missing-value tolerance, computational efficiency, and compatibility with SHAP-based feature attribution — properties that are directly relevant given the irregular LEO data coverage and the severe class imbalance in hazardous-$|dB/dt|$ events. This model choice is consistent with recent operational space weather forecasting literature (Ferdousi et al., 2025; Keebler et al., 2025).

The approach is physics-informed in two ways: (1) physics-derived features — specifically the Newell coupling function $\Phi$, rolling IMF $B_z$ statistics, and cyclical solar time encodings — are embedded as engineered inputs to the classifier, encoding known driver-response relationships; (2) model evaluation is structured around operationally meaningful thresholds and imbalance-aware metrics rather than aggregate accuracy.

The evaluation protocol uses two baselines: a persistence baseline (predict that the future label matches the most recent observed label) and a climatology baseline (predict the unconditional training-set event rate). Any primary model that fails to exceed both baselines provides no operational value. The primary metric is the Heidke Skill Score (HSS), with secondary reporting of ROC AUC, Brier Skill Score, precision/recall at the HSS-optimal threshold, and a reliability diagram.

**Attribution:**
The gray-box framework is adapted from Camporeale et al. (2020). The Newell coupling function follows Newell et al. (2007). The LightGBM architecture follows Ke et al. (2017). The LSTM secondary model, if implemented, follows Hochreiter & Schmidhuber (1997) with focal loss weighting following Lin et al. (2017).

---

### 4. Technical Implementation Details

**Dataset:**
The training dataset fuses five observation layers at one-minute cadence over January 2015 – December 2024:

- *Solar:* GOES X-ray flux (1–8 Å channel), capturing flare activity as a solar driver proxy.
- *L1:* OMNI 1-minute solar wind data (CDAWeb), including IMF components ($B_x$, $B_y$, $B_z$ in GSM), solar wind speed ($V_x$), proton density, dynamic pressure, and geomagnetic indices (SYM-H, AL, AU). Note: AL and AU are included as contextual features but will be scrutinized for label leakage before final training.
- *GEO:* GOES geostationary magnetic field ($B_x$, $B_y$, $B_z$ in GSM, $B_T$), capturing near-Earth magnetospheric state.
- *LEO:* Swarm-derived electron flux proxies aggregated by MLT zone (high-latitude, mid-latitude, dayside, nightside), with freshness flags and decay-age tracking to handle the sparse and irregular orbital sampling pattern. Field-aligned current proxies are computed using the ChaosMagPy library.
- *Ground (Target):* SuperMAG magnetic field components ($B_N$, $B_E$, $B_Z$) and computed $|dB/dt|$ for three Scandinavian high-latitude stations: ABK (Abisko), BJN (Bjørnøya), and TRO (Tromsø). These stations form a regional cluster in the auroral zone, a region of high GIC relevance.

The dataset spans 118 successfully processed months with approximately 5.2 million one-minute records. Year 2024 is reserved as a temporally held-out test set; 2022–2023 serves as a validation set; 2015–2021 is used for training.

**Feature Engineering:**
Physics-informed features are central to the input representation:

- *Newell Coupling Function* ($\Phi = V^{4/3} |B_T \sin^3(\theta/2)|$): Encodes solar wind–magnetosphere coupling efficiency via magnetic reconnection rate. Included in both instantaneous and rolling (10-minute and 30-minute) forms.
- *Rolling IMF statistics:* 10-minute and 30-minute rolling means and standard deviations of $B_z$ and $V_x$, capturing sustained southward field excursions rather than instantaneous values.
- *Cyclical time encodings:* Universal time and day-of-year encoded as sine/cosine pairs to represent diurnal and seasonal structure in ionospheric conductance without ordinal leakage.
- *Missing-value and freshness flags:* Binary indicators for GOES gaps and LEO orbital coverage, allowing the model to condition on data quality rather than treating imputed values as fully observed.
- *Recent ground state:* Lagged $|dB/dt|$ from the target stations, providing short-memory persistence features consistent with the gray-box approach of Camporeale et al. (2020).

**Models:**

| Model | Role | Key Justification |
|---|---|---|
| Persistence | Baseline | Minimum operational bar; any useful model must exceed it |
| Climatology | Baseline | Denominator for Brier Skill Score |
| Logistic Regression | Linear reference | Establishes linear ceiling; interpretable coefficient view |
| LightGBM | **Primary model** | Handles imbalance, missing data, and tabular heterogeneity natively; SHAP-compatible |
| LSTM | Optional extension | Tests whether learned temporal structure adds skill beyond lag features |

---

### 5. Success Criteria

The primary quantitative target is **HSS > 0.35** on the 2024 held-out test set for the LightGBM classifier. This threshold is chosen based on reported performance in comparable operational geomagnetic disturbance classifiers (Camporeale, 2020; Chen, 2025). Secondary targets are Brier Skill Score > 0.10 relative to the climatology baseline, ROC AUC > 0.80, and precision > 0.50 at the HSS-optimal decision threshold. The last criterion is operationally motivated: a false alarm rate that dominates true alerts provides no value to grid operators and erodes trust in automated warning systems (Camporeale, 2025).

Calibration will be evaluated with a reliability diagram and expected calibration error (ECE). The model will also be evaluated stratified by storm intensity (quiet, moderate, intense) using GEM substorm event dates from SuperMAG where available. 

The project is considered successful if the primary LightGBM model consistently outperforms both baseline models on the held-out test set on HSS and BSS, and produces a reliability diagram indicating that forecast probabilities are well-calibrated. Reaching the LSTM extension is a secondary goal that will be pursued if the primary model pipeline is complete and stable before the end of the quarter.

---

