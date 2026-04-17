# Extended Lead‐Time Geomagnetic Storm Forecasting With Solar Wind Ensembles and Machine Learning

## Space Weather: Billcliff (2026)

**TL;DR:** Extends geomagnetic storm lead time by combining near-Sun solar-wind ensembles, HUXt propagation, and logistic regression for probabilistic global storm forecasting.

* **Target:** $Hp30_{MAX} \geq 4.66$ within a 24-hour forecast window, focusing on global storm occurrence rather than regional $dB/dt$.
* **Inputs:** MAS-derived Carrington maps, HUXt velocity ensembles, OMNI comparison features, and historical $Hp30$.
* **Method:** Ensemble of logistic regressions weighted by historical OMNI agreement.

### Key Takeaways
* **Strengths:** Demonstrates that incorporating pre-L1 context can significantly extend lead times and highlights the importance of probabilistic calibration.
* **Limitations for Project:** Lacks regional targets and direct Sun/L1/GEO/LEO/ground fusion. It does not include CME-resolved magnetic-field forecasting; the velocity-only ambient modeling fails to capture strong events where $B_z$ orientation is the primary driver.

**Citation Use:**
Useful as a contrast paper showing a lightweight lead-time extension path based on solar-wind propagation rather than direct multimodal observational fusion.

### Feature Engineering
* **Rolling and derivative forms** of $B_z$ and the Newell coupling function $\Phi$.
* **Threshold-duration counters** and event-onset transition markers.
* **Source-block ablations** to evaluate the individual contribution of each feature set to model performance.

---
# GeoDGP: One-Hour Ahead Global Probabilistic Geomagnetic Perturbation Forecasting Using Deep Gaussian Process

## Space Weather: Chen (2025)

**TL;DR:** Builds a global probabilistic forecast model for ground magnetic perturbations using a Deep Gaussian Process (DGP). Demonstrates that one-hour-ahead $dB_H$ forecasts can outperform Geospace and DAGGER, though the model still struggles with auroral/high-latitude regimes and localized effects.

* **Target:** Global and regional forecasting of $dB_H$ at 1-minute cadence. Lead times include $T_S$ and $T_S + 1$ hour (where $T_S$ is L1-to-Earth propagation time); also models $dB_N$ and $dB_E$.
* **Inputs:** OMNI-derived solar wind and IMF ($V_x, N_p, T, B_x, B_y, B_z$), $F10.7$, $Dst$, location encoding in solar magnetic (SM) coordinates, and dipole tilt. Includes 1-hour history for solar wind/IMF (5-minute medians) and 12-hour $Dst$ history.
* **Method:** Deep Gaussian Process regression with spatial conditioning in SM coordinates. Trained jointly across stations to produce probabilistic forecasts and prediction intervals at arbitrary global locations.

### Key Takeaways
* **Strengths:** Provides a strong benchmark for probabilistic global $dB_H$ forecasting; offers direct comparisons against Geospace and DAGGER; utilizes clear regional evaluation by magnetic latitude and reusable threshold-based scoring.
* **Limitations for Project:** Targets $dB_H$ rather than regional $|dB/dt|$; relies on L1/state/location inputs rather than full multimodal fusion (Sun/L1/GEO/LEO/ground). Auroral-zone predictions remain challenging, characterized by underpredicted peaks in $dB_E$, overpredicted troughs in $dB_H$, and below-nominal uncertainty coverage.

**Citation Use:**
Primary benchmark for regional threshold selection and probabilistic evaluation. Highlights the limitations of global smooth models in the high-latitude and localized settings critical to this project.

### Feature Engineering & Methodology
* **Regional Threshold Framing:** Evaluation uses magnetic-latitude bins. Low/mid latitudes ($< 50^\circ$) use 50, 100, and 200 nT; high latitudes ($\geq 50^\circ$) use 50, 200, 300, and 400 nT. These are linked to GEM-style $dB/dt_H$ thresholds via a power-law relation.
* **Storm-State Inputs:** A **NoDst** ablation study confirms that removing $Dst$ consistently degrades $dB_H$ performance. While $Dst$-like state variables are mathematically beneficial, operational alignment/leakage must be monitored.
* **History Windows:** Supports short upstream memory (1 hour for solar wind/IMF) and longer memory for slow-varying storm-state variables ($Dst$).
* **Regional Evaluation Segments:** Explicitly splits results into low/mid latitude ($< 50^\circ$), high latitude ($\geq 50^\circ$), and auroral latitude ($60^\circ$–$70^\circ$).
* **DAGGER Comparison:** GeoDGP outperforms DAGGER on storm sets while extending the horizon to $T_S + 1$ hr; DAGGER notably underpredicts high-level disturbances at larger thresholds.
* **Uncertainty Calibration:** While prediction intervals are provided, empirical coverage drops below nominal levels in auroral latitudes, indicating that uncertainty quality degrades in high-interest regions.

---

# Verification of the NOAA Space Weather Prediction Center Solar Flare Forecast 1998–2024

## Space Weather: Camporeale (2025)

**TL;DR:** A long-horizon verification study showing that NOAA/SWPC probabilistic forecasts for M- and X-class solar flares are often matched or beaten by simple baselines (persistence, climatology, and lightweight statistical models), revealing significant calibration and false-alarm issues in operational settings.

* **Target:** Daily probabilistic prediction of at least one M-class or X-class flare occurring within a 1, 2, or 3-day forecast window.
* **Inputs:** NOAA/SWPC issued flare probabilities compared against historical records. Baselines utilize simple real-time features: number of consecutive flare-free days, same-day sunspot number, and persistence from the previous day.
* **Method:** Rolling operational-style verification (1998–2024) comparing SWPC against persistence, empirical climatology, Naive Bayes, logistic regression, and a baseline-average ensemble using binary metrics and probabilistic calibration analysis.

### Key Takeaways
* **Strengths:** Excellent methodology reference for rigorous operational verification. Emphasizes rolling retraining to prevent data leakage, calibration checks, reliability diagrams, and the necessity of benchmarking against strong trivial baselines.
* **Limitations for Project:** Focuses on solar flare forecasting rather than geomagnetic disturbance; does not directly address regional ground perturbation or multimodal Sun/L1/GEO/LEO/ground fusion.

**Citation Use:**
Useful as a standards paper for evaluation logic. It justifies the requirement that any new operational space-weather model must be compared against persistence, judged via imbalance-aware metrics, and audited for high-stakes edge-case performance.

### Feature Engineering
* **Persistence-Style Baselines:** Explicitly include persistence features, as trivial temporal memory remains a high bar for operational performance.
* **Quiet-Time Memory Features:** Track metrics such as "time since last event" or "time since threshold crossing" to expose weaknesses in "storm after the calm" scenarios.
* **Imbalance-Aware Evaluation:** Prioritize CSI, HSS, Precision/Recall, FAR, and calibration metrics over Accuracy or Brier scores, which can be deceptive in rare-event settings.
* **Edge Scenario Stress-Testing:** Specifically evaluate the first event following a long quiet period and "all-clear" conditions, as these are operationally decisive.
* **Ensemble Benchmarking:** Compare against a simple ensemble average of baseline models, which often proves more robust than individual complex models or official forecasts.

---

# Classification of Solar Wind With Machine Learning

## JGR Space Physics: Camporeale (2017)

**TL;DR:** Uses a Gaussian Process classifier to probabilistically label hourly OMNI solar wind into four physically motivated source categories. Demonstrates that probabilistic regime identification is accurate, reliable, and provides a valuable baseline context layer for downstream space-weather forecasting.

* **Target:** Four-way solar wind classification at 1-hour cadence: ejecta, coronal hole origin plasma, sector reversal origin plasma, and streamer belt origin plasma (following the Xu and Borovsky (2015) scheme).
* **Inputs:** Seven OMNI-derived features: solar wind speed $V_{sw}$, proton temperature standard deviation, sunspot number, $F10.7$, Alfvén speed $v_A$, proton specific entropy $S_p$, and the temperature ratio $T_{exp}/T_p$.
* **Method:** Probabilistic multiclass Gaussian Process classification trained on labeled OMNI events. Includes ROC analysis, reliability analysis, and confidence-based thresholding to allow for an "undecided" class.

### Key Takeaways
* **Strengths:** Conceptually relevant for treating upstream states as probabilistic regimes rather than raw variable streams. Explicitly demonstrates how confidence-thresholding can trade coverage for higher purity in classification.
* **Limitations for Project:** Focuses on nowcasting/classification rather than regional ground-perturbation forecasting. Uses hourly OMNI and broad regime labels rather than high-cadence multimodal drivers or direct $dB/dt$ targets.

**Citation Use:**
Useful for feature-logic justification: instead of raw continuous inputs, the model can utilize regime-aware context variables or transition-state features to summarize whether the environment is ejecta-like, coronal-hole-like, streamer-belt-like, or sector-reversal-like.

### Feature Engineering
* **Regime Class Probabilities:** Supports adding upstream solar-wind regime probabilities as features for downstream geomagnetic forecasting, moving beyond raw plasma variables.
* **Confidence Thresholding:** Demonstrates that higher probability thresholds improve purity; this maps directly to alerting logic and the ability to selectively trust high-confidence forecasts.
* **Transition Matrix Features:** The provided transition matrix between solar wind categories offers a climatological persistence baseline, suggesting that regime-transition markers (e.g., entering a sector reversal) are highly informative.
* **Persistence Dominance:** Replaces the assumption of simple continuity with measured transition probabilities, reinforcing the need for models to outperform temporal persistence.
* **Multi-Parameter Encoding:** Explicitly argues that speed alone is an insufficient classifier, justifying the inclusion of coupling functions (Newell), $B_z$, and thermodynamic ratios over velocity-only inputs.

---

# A Gray-Box Model for a Probabilistic Estimate of Regional Ground Magnetic Perturbations: Enhancing the NOAA Operational Geospace Model With Machine Learning

## JGR Space Physics: Camporeale (2020)

**TL;DR:** A hybrid "gray-box" model that enhances NOAA’s operational Geospace forecasts by combining physics-model outputs with machine learning. It predicts the probability that regional $dB/dt$ will exceed station-specific thresholds, with a heavy focus on calibration and operational verification.

* **Target:** Probabilistic prediction that the maximum horizontal $dB/dt$ over a 20-minute interval exceeds a specified threshold at a given station. Uses overlapping 20-minute windows with 1-minute strides.
* **Inputs:** Observations, OMNI drivers, geomagnetic indices, and Geospace outputs. Top features include: past $dB/dt$, Geospace-predicted 20-minute range of $B_n$ and $B_e$, IMF $B_z$, and $Sym\text{-}H$.
* **Method:** A gray-box classifier utilizing RobustBoosted decision trees. Compared against a "white-box" baseline (Geospace alone) and a "black-box" ML model (without Geospace-derived features).

### Key Takeaways
* **Strengths:** Highly relevant as a direct precedent for regional geomagnetic disturbance forecasting. It uses threshold exceedance, combines physics with ML, and treats probability calibration as a primary requirement.
* **Limitations for Project:** Validated at only three stations; uses station-specific models rather than a single regional/global architecture. Relies heavily on persistence (past $dB/dt$), which may limit lead-time extension.

**Citation Use:**
Methodological ancestor for hybrid forecasting. Supports a strategy where physics models are corrected or enhanced by ML, particularly for threshold-based operational products relevant to GIC (Geomagnetically Induced Current) risk.

### Feature Engineering & Methodology
* **Gray-Box Fusion:** Demonstrated that Geospace alone underestimates large $dB/dt$ at high latitudes; adding ML consistently improves TPR, TSS, and HSS.
* **Local Thresholding:** Defines station-specific thresholds based on the 60th–95th percentiles of a 19-year dataset. Justifies why regional alerting should be localized rather than universal.
* **Feature Importance:** Ranks past $dB/dt$ as the strongest predictor, followed by Geospace-derived magnetic-range terms, supporting the combination of state memory with simulation-informed structure.
* **Recalibration:** Boosted tree probabilities are explicitly identified as miscalibrated and corrected using reliability-based recalibration from the training set—a critical operational lesson.
* **Operational Evaluation:** Uses temporally disjoint train/test periods and benchmarks white-box, black-box, and gray-box approaches using ROC, TPR, FPR, TSS, and HSS.

### Project Alignment
* **Primary Citation:** Acts as the strongest precedent for "physics + ML + regional threshold exceedance + operational metrics."
* **Leverage:** Borrow station/regional thresholding techniques and the framing of simulation outputs as informative features rather than final answers.
* **Advancement:** Aim to move beyond three-station classifiers toward a unified regional model and reduce reliance on persistence-heavy features to maximize true lead-time gains.

---

# A Foundation Model for the Earth System

## Nature: Bodnar (2025)

**TL;DR:** Introduces Aurora, a large-scale Earth-system foundation model pretrained on massive, heterogeneous geophysical datasets. It demonstrates that a single pretrained architecture can be fine-tuned to outperform specialized operational systems across multiple tasks (weather, air quality, ocean waves) at a significantly lower computational cost.

* **Target:** A general-purpose Earth-system forecasting framework capable of downstream fine-tuning for air quality, ocean waves, tropical cyclone tracking, and high-resolution atmospheric weather.
* **Inputs:** Heterogeneous multi-source geophysical data spanning reanalysis, forecasts, reforecasts, and climate simulations. Designed to ingest varying variables, pressure levels, and spatial resolutions.
* **Method:** A foundation-model pipeline utilizing large-scale pretraining followed by task-specific fine-tuning. The architecture features a flexible encoder–processor–decoder built around a 3D Swin Transformer with Perceiver-based modules.

### Key Takeaways
* **Strengths:** Provides high-level justification for the "pretrain-then-fine-tune" paradigm and heterogeneous data fusion. Proves that transfer learning across Earth-system tasks is now competitive with operational physics-based systems.
* **Limitations for Project:** Not a space-weather paper; does not address geomagnetic disturbances or $dB/dt$. Relies on traditional data-assimilation products for initialization. Best used for motivational and methodological framing.

**Citation Use:**
Supports the shift in Earth-system forecasting toward reusable pretrained models and multimodal ingestion rather than building isolated systems from scratch.

### Feature Engineering & Methodology
* **Pretraining for Extremes:** Demonstrates that pretraining on diverse datasets significantly improves downstream performance on extreme values—validating the use of broad upstream context for specialized forecasting.
* **Handling Heterogeneity:** Aurora’s design for varied resolutions and missing-data patterns is directly applicable to the challenges of a multimodal Sun/L1/GEO/LEO/ground data setup.
* **Extreme-Event Evaluation:** Utilizes thresholded RMSE and targeted case studies for rare events, reinforcing that aggregate error metrics are insufficient for operational hazard forecasting.
* **Physics-AI Complementarity:** While data-driven, the model still relies on traditional assimilation-based initial conditions, supporting a "gray-box" framing where AI enhances existing operational pipelines.
* **Scaling Law Justification:** Argues that diverse data and model scale improve performance on extremes, providing a roadmap for starting with specialized milestones while planning for future scale-up.
* **Operational Feasibility:** Frames task-specific fine-tuning as computationally inexpensive compared to full numerical systems, strengthening the argument for the practical deployment of ML in Earth-science contexts.

---

# A Review of Earth Artificial Intelligence

## Computers & Geosciences: Sun et al. (2022)

**TL;DR:** A comprehensive review of AI/ML applications across Earth-system sciences. It synthesizes common algorithms, workflows, and tooling while emphasizing the critical roles of data preparation, uncertainty quantification, explainability, and the integration of physical principles into machine learning.

* **Target:** Narrative review of AI usage across the geosphere, hydrosphere, atmosphere, cryosphere, oceanography, and biosphere (not a specific forecasting model).
* **Inputs:** Synthesis of methods, domain applications, cyberinfrastructure, and operational challenges spanning modern Earth-science AI practices.
* **Method:** Organized around AI techniques, workflow stages (from data prep to deployment), and cross-cutting themes such as generalization, reproducibility, and physics-model integration.

### Key Takeaways
* **Strengths:** Provides high-level academic language for workflow design, MLOps-style operationalization, and project standards. Excellent for framing the "professionalism" and rigor of a geoscientific ML pipeline.
* **Limitations for Project:** Broad scope; lacks specific space-weather baselines or domain-specific geomagnetic constants. Use this for methodology framing rather than direct model comparison.

**Citation Use:**
Justifies the requirement that Earth-science ML projects must address data preparation, uncertainty, generalization, explainability, and physics integration alongside raw prediction accuracy.

### Feature Engineering & Methodology
* **Data Preparation Bottleneck:** Explicitly identifies data harmonization (formats, temporal preprocessing, metadata) as the primary effort-sink in Earth AI, validating the complexity of multimodal space-weather pipelines.
* **Baseline Hierarchy:** Recommends a "simple-first" approach, justifying the inclusion of persistence and classical statistical baselines before moving to deep learning.
* **Sensitivity Analysis:** Emphasizes that feature importance and sensitivity analysis are prerequisites for scientific trust, supporting the use of ablation studies.
* **Uncertainty Quantification (UQ):** Distinguishes between aleatoric and epistemic uncertainty. Provides a framework for arguing why geomagnetic forecasts must be probabilistic.
* **Physics-ML Integration:** Highlights the trend of hybrid modeling (AI-enhanced physics models or physics-constrained ML), which aligns with current space-weather "gray-box" research.
* **Generalization Challenges:** Warns that Earth-science models often fail when moved across different temporal or spatial regimes, a critical consideration for regional forecasting across different latitudes and solar cycles.
* **Workflow Orchestration:** Stresses provenance, reproducibility, and reusability, providing a strong citation for why pipeline structure and data lineage are central to a project's success.

---

# 2024 SWAG and 2025 Solar & Space Physics Decadal

## Community / Strategic Guidance: SWAG (2024) + National Academies Decadal (2025)

**TL;DR:** These are not model papers, but they are highly valuable for project justification because they show that the operational community wants better regional measurements, local validation, and actionable 1-hour infrastructure-relevant forecasts, while the decadal frames space weather explicitly as a coupled “system of systems.”

* **Target:** Strategic and operational guidance for next-decade space weather capability, especially around regional validation, infrastructure impacts, and forecast products useful to operators.
* **Inputs:** Community/operator needs from SWAG-style user and stakeholder feedback, plus decadal-survey priorities spanning solar drivers, physical-system response, and infrastructure impacts.
* **Method:** Not a forecasting method paper; these sources function as motivation and requirements documents that define what outcomes matter, what gaps remain, and what operational capabilities are worth building.

### Key Takeaways
* **Strengths:** Extremely useful for milestone framing because they justify your project from the user-needs side, not just the modeling side; they support regional forecasting, local measurements, multimodal system coupling, and outcome-driven evaluation.
* **Limitations for Project:** These are not technical benchmark papers and do not provide train/test designs, model architectures, ablation studies, or implementation details, so they should support motivation and scope rather than replace methodological citations.

**Citation Use:**
Useful for arguing that space weather forecasting should be designed around operational outcomes and regional decision needs, especially for power-system relevance, instead of only optimizing generic global metrics.

### Operational Motivation
* **Regional data matters:** SWAG-style notes emphasize the value of better data access and sharing between operators, the need for more local measurements, and the importance of improved regional indices.
* **Threshold-based decisions are real:** Operator procedures are commonly triggered around disturbance thresholds such as G3/K7, with some lower-latitude stakeholders acting at stronger thresholds like G4/K8, which supports a threshold-based classifier framing.
* **Validation is constrained by geography:** Community feedback highlights practical validation challenges such as quiet-cycle conditions, lack of nearby magnetometers/variometers, and GIC monitors being placed where topology may suppress observed impacts, all of which support your emphasis on regionalization and careful station/site interpretation.

### System-of-Systems Framing
* **End-to-end coupling:** The decadal frames space weather as a coupled chain from solar eruptions to physical-system responses to infrastructure effects, which strongly supports your Sun/L1/GEO/LEO/ground feature architecture.
* **Physical-system responses:** The decadal explicitly identifies LEO density, ionospheric state, and magnetospheric state as key response layers, which aligns well with your use of multisource space-environment observations rather than only upstream solar-wind inputs.
* **Impact relevance:** It also links those responses to actionable outcomes such as geoelectric fields, radiation exposure, and infrastructure disruption, giving a clean rationale for forecasting geomagnetic disturbance proxies that matter to grid operations.

### Project Relevance
* **Why your problem is well chosen:** The decadal calls for operationally useful products such as a 1-hour geoelectric field variation forecast at roughly 200 km spatial resolution for power-system operators, which is closely aligned with your one-hour regional hazard forecasting direction.
* **Why multimodal fusion is defensible:** The strategic framing supports the idea that no single measurement layer is sufficient; useful operational forecasting needs information spanning solar drivers, near-Earth plasma state, and ground response.
* **How to use in writing:** These sources are best cited in the motivation and problem-definition sections to justify regional forecasting, lead-time goals, and multimodal architecture, while your technical papers should still carry the methods and benchmark discussion.
