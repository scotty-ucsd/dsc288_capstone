# Lecture Tips: Model Selection
* Goal: choose the right inductive bias for the data and task
    * Inductive bias determines what patterns a model can represent
    * Model choice impacts accuracy, robustness, and stability
    * Also affects interpretability, latency, and deployment cost
* Focus: principled reasoning, **not** model shopping

## Why Model Seletion Matters
* Different models fail in systematically different ways:
    * Underfitting: model too simple for data structure
    * Overfitting: model memorizes noise instead of signal
    * Mismatch between data regime and model capacity
* **Example:** A deep NN fit training churn data well but collapsed on unseen customers.

## Model Seclection vs Feature Engineering
* Feature engineering shapes the representation space
    * Determines which relationships are expressible
* Model selection shapes the hypothesis space
    * Determines how relationships are learned
* **Example:** Linear + interaction features rival tree
models

## Model Families and Core Assumptions
* Linear models:
    * Global additive effects
    * Limited interaction modeling
    * Strengths:
        * Assume linear decision boundary in feature space
            * Perform well with strong feature engineering
            * Coefficients directly interpretable
            * Fast training and inference 
        * **Example:** Credit risk models under regulation
            * Regulators preferred transparent coefficients over black-box gains.
    * Regularization:
        * L1 regularization:
            * Encourages sparsity
            * Performs implicit feature selection
        * L2 regularization:
            * Stabilizes correlated feature
        * **Example:** Marketing attribution modeling
            * L1 revealed that only a few channels drove marketing impact.
* Tree models:
    * Hierarchical, rule-based splits
    * Single trees are unstable
        * Ensembles improve robustness
    * Strengths:
        * Automatically capture nonlinear thresholds
        * Naturally model feature interactions
        * Robust to monotonic feature transforms
    * **Example:** Fraud Detection
        * Tree splits mirrored expert-defined fraud rules.
* Neural nets:
    * Compositional feature hierarchies
    * High-capacity universal approximators
        * Learn hierarchical representations
        * Scale with data and compute
        * Require careful regularization
    * **Example:** Image recognition with CNNs 
        * removed the need for manual image feature design.

## Start with Basic Baseline
* Baselines provide a sanity check
    * Catch leakage, bugs, and label noise early
    * Establish minimum acceptable performance
    * Easier to interpret and debug
* **Example:** Logistic regression nearly matches XGBoost

## Bias and Variance
* Bias error:
    * Model too constrained to capture structure
* Variance error:
    * Model overly sensitive to training data
* Note: Model complexity controls balance
* **Example:** Polynomial regression degree tuning 
    * Validation error clearly identified the optimal polynomial degree.

## Bagging vs Boosting
* Bagging
    * Trains models independently
    * Reduces variance via averaging
* Boosting
    * Sequentially focuses on errors
    * Reduces bias
* **Example:** Random Forest vs XGBoost 
    * Boosting improved recall on hard-to-detect defaulters.

## GBDTs as Tabular Data Workhorse
* Strong default for structured data
    * Handles mixed feature types
    * Captures complex interactions
    * Requires careful hyperparameter tuning
* **Example:** Kaggle tabular benchmarks 
    * GBDTs consistently topped leaderboards on tabular benchmarks.

## Kernel Methods (SVMs)
* Implicit feature expansion via kernels
    * Effective in high-dimensional spaces
    * Strong margins improve generalization
    * Training scales poorly
* **Example:** Small-sample text classification 
    * SVMs excelled when labeled text data was scarce.

## Distance-Based Models (k-NN)
* Prediction based on similarity
    * No explicit training phase
    * Sensitive to scaling and distance metrics
    * Curse of dimensionality degrades performance
* **Example:** Cold-start recommendation 
    * k-NN gave reasonable recommendations before sufficient history accumulated

## Probabilistic Models
* Explicit probabilistic assumptions
    * Model uncertainty directly
    * Often data-efficient
    * Strong independence assumptions
* **Example:** Naive Bayes for spam filtering
    * Naive Bayes remained robust despite noisy text features.

## Deep Learning Failure Modes
* Overfitting on small or medium datasets
    * Especially severe for tabular data
* Optimization instability
    * Sensitive to initialization and learning rates
* **Example:** Finance datasets 
    * Performance varied wildly acrossrandom seeds.

## Time Series Models
* Classical models:
    * Encode stationarity and seasonality
* ML models:
    * Depend on engineered lag features
* DL models:
    * Capture long-range dependencies
* **Example:** ARIMA vs LSTM 
    * LSTM only helped once multiple correlated signals were added.

## Evaluation Drives Selecection
* Metrics encode business costs
    * Accuracy often misleading
* Cross-validation estimates robustness
    * Stratification for imbalance
* **Example:** PR-AUC vs ROC-AUC 
    * ROC masked poor minority-class performance.

## Operational and Ethical Constraints
* Latency constraints limit model complexity
* Explainability may be legally required
* Fairness considerations influence model choice
* **Example:** Real-time credit approval
    * Simpler models reduced audit and compliance risk.

## Deep Dives

### Customer Churn
* Dataset: 20k customers, 50 features
* Models evaluated:
    * Logistic regression (AUC ≈ 0.81)
    * XGBoost (AUC ≈ 0.83)
* Tradeoff:
    * Marginal accuracy vs interpretability
* Decision: Logistic regression deployed.
    * Business stakeholders trusted and adopted the simpler model faster.

### Fraud Detection
* Dataset: 5M transactions, extreme imbalance
* Key challenges:
    * Rare positives
    * Concept drift
* Models:
    * RF vs XGBoost
* Decision: Boosted trees + PR-AUC
    * Boosted trees improved recall without overwhelming investigators.

## When Models Break
* Deep neural networks:
    * Fail on small tabular data
* k-NN:
    * Fails in high dimensions
* SVMs:
    * Do not scale to massive datasets
* Lesson: assumptions must match data

## Model Selection Flowchart
* **Step 1:** Identify data type
    * Tabular, text, image, time series
* **Step 2:** Estimate data size
    * Small vs large regimes
* **Step 3:** Apply constraints
    * Interpretability, latency, cost
* **Outcome:** narrow to 2–3 candidates

## Conclusion
* No universally best model
* Model choice is data-dependent
* Start simple and justify complexity
* Features, models, and evaluation co-evolve
* Successful teams revisit model choice as data evolves.



