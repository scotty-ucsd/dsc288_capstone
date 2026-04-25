# Lecture Tips: Feature Engineering

## Feature Engineering: Model Choice
* Information
    * Features define what information a model can access
    * No learning algorithm can recover missing information
* Bias
    * Feature choice sets inductive bias implicitly
    * Bias–variance tradeoff is often controlled via representation

## Feature Engineering: ML Pipeline
* Raw data 
    * Noisy, high-entropy, and poorly structured
    * Logs, text, timestamps, IDs are not model-ready 
* Feature engineering 
    * compresses raw data into stable signals
    * Aggregation and normalization reduce variance

## Feature Taxonomy

### Numerical
* encode magnitudes and counts
* represent measurable quantities or counts
* They encode magnitude, frequency, or intensity
* **Principles:**
    * Control scale so features interact properly with regularization
    * Prefer normalized or ratio-based features
    * Reduce skew and heavy tails using transforms
        * Ensure features are stable over time and populations
    * **DO'S**
        * DO normalize or standardize features
        * DO use log or power transforms for heavy-tailed variables
        * DO clip extreme outliers deliberately
    * **DON'TS**
        * DON'T use raw magnitudes blindly
        * DON'T let outliers dominate decision boundaries
* **Examples:** income, account balance, transaction count
* **Example dataset:** credit scoring
    * Raw inputs: annual income, total debt, monthly transactions

#### Applying Do's and Don’ts: Credit Scoring
* Raw income $\rightarrow$ log(income)
    * Reduces dominance of very high earners
* Raw debt $\rightarrow$ debt-to-income ratio
    * Encodes repayment stress independent of wealth
* Raw transaction count $\rightarrow$ transactions per month
    * Removes exposure-time bias

### Categorical
* Categorical features represent discrete, unordered states
    * No numeric distance exists between categories
* **Principles:**
    * Avoid imposing artificial ordering
    * Control dimensionality as cardinality grows
    * Preserve signal while avoiding label leakage
        * Plan for unseen categories at inference time
    * **DO'S**
        * DO use one-hot encoding for low cardinality
        * DO use CV-safe target encoding when appropriate
        * DO handle rare and unseen categories explicitly
    * **DON'TS**
        * DON'T apply naive target encoding
        * DON'T use raw magnitudes blindly
        * DON'T let outliers dominate decision boundaries
DON'T let high-cardinality explode dimensionality
* **Examples:** country, subscription plan, device type
* **Example dataset:** customer churn prediction
    * Raw input: subscription plan = Basic / Plus / Premium

#### Applying Do's and Don'ts: Churn
*  Subscription plan $\rightarrow$ one-hot encoding (3 plans)
    * Interpretable and low dimensional
* If plans grow $\rightarrow$ CV-safe target encoding
    * Encodes churn risk without leakage
* Unseen plan $\rightarrow$ mapped to 'other'
    * Prevents runtime failure

### Ordinal
* Ordinal features are categorical with inherent ordering
* Ordering matters; distances usually do not
* **Principles:**
    * Preserve ordering explicitly
    * Avoid encodings that destroy monotonicity
    * Check that model assumptions respect order
    * **DO'S**
        * DO encode ordinal levels as ordered integers
        * DO consider monotonic constraints when available
    * **DON'TS**
        * DON'T use one-hot encoding by default
        * DON'T assume equal spacing without justification
* **Examples:** credit rating, education level
* **Example dataset:** credit risk assessment
    * Raw input: credit rating = AAA, AA, A, BBB, ...

#### Applying Do's and Don'ts: Credit Rating
* AAA→1, AA→2, A→3, BBB→4,...
    * Preserves increasing risk ordering
* Tree models split naturally on thresholds
    * Linear models respect monotonic trend

### Temporal
* Temporal features encode time-dependent structure
* Observations are correlated across time
* **Principles:**
    * Encode memory explicitly using lags
    * Capture trends and volatility with rolling windows
    * Separate calendar effects from signal
    * Respect causality strictly
    * **DO'S**
        * DO use lagged and rolling features
        * DO align timestamps carefully
        * DO add calendar features
    * **DON'TS**
        * DON'T use future data
        * DON'T treat time as i.i.d.

* **Examples:** sensor readings, demand over time
* **Example dataset:** energy demand forecasting
    * Raw input: hourly electricity load

#### Applying Do's and Don'ts: Energy Forecasting
* Load(t−1), Load(t−24)
* Capture short-term and daily cycles
* Rolling mean (24h)
* Smooths transient spikes
* Hour-of-day, weekday features
* Explain systematic demand patterns

### Text and Unstructured
* Text features are unstructured and high dimensional
* Semantic meaning is implicit, not explicit
* **Principles:**
    * Reduce dimensionality while preserving meaning
    * Choose representation based on task semantics
    * Balance interpretability and expressiveness
    * **DO'S**
        * DO start with TF–IDF for classical models
        * DO use embeddings when semantics matter
        * DO monitor vocabulary drift
    * **DON'TS**
        * DON'T use massive raw bag-of-words
        * DON'T assume pretrained embeddings fit your domain
* **Examples:** reviews, support tickets, notes
* **Example dataset:** product review sentiment
    * Raw input: free-text customer reviews

#### Applying Do's and Don'ts: Sentiment Analysis
* TF–IDF captures sentiment-laden terms
    * Down weights common neutral words
* Embeddings capture paraphrases
    * Generalize across phrasing
* Periodic retraining handles language drift
    * Maintains performance over time

## Dimensionality Reduction
* What it is
    * Techniques that reduce feature dimensionality while preserving signal
    * Used when features are redundant or highly correlated
* **Example:** numerical + interaction features in credit scoring
    * Many correlated financial ratios encode similar information
    * PCA removes correlation and stabilizes linear models
* **Caution:**
    * Reduced features are harder to interpret and audit

## Feature Leakage
* What it is
    * Using information not available at prediction time
* **Example:** temporal features in energy demand forecasting
    * Rolling averages accidentally include future timestamps
    * Model shows unrealistically high offline accuracy
* **Caution:**
    * Leaks create models that fail catastrophically in production

## Missing Data as Signal
* Key idea
    * Missingness itself can carry information
* **Example:** credit scoring and transaction history
    * Missing income or sparse transaction data correlates with risk
    * Naive imputation destroys this signal
* Correct handling
    * Add missingness indicator features alongside imputation

## Evaluating Feature Quality
* Why evaluation matters 
    * Feature impact is often non-obvious
* **Example:** credit scoring features
    * Ablation: remove debt-to-income ratio → performance drops
    * Permutation importance confirms dependency
    * SHAP values explain contribution direction
* **Goal:** Detect spurious or leaked features early
