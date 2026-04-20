# Lecture Tips: EDA

## What is EDA
* A systematic process of understanding data beforemodeling
* Focuses on structure, quality, distributions, relationships, and anomalies
* Uses statistics, visualization, and domain reasoning
* **Goal:** Gain insights and generate hypotheses, not confirm them

## Importance
* Reveals data quality issues early (degree of missing data, noise,bias)
* Guides feature engineering and model choice
* Prevents invalid assumptions in downstream analysis
* Builds intuition and trust in the dataset
* Identifying outliers and missing values
* Identifying trends in time and space
* Uncover patterns related to the target (dependent variable)
* Creating hypotheses and testing them through experiments
* Enable data-driven decisions
* Test underlying assumptions (especially for implementing statistical models)
* Identify essential and unnecessary features
* Examine distributions such as examination of Normality
* Assess data quality
* Graphical representation of data for quick visual analysis

## Core Questions EDA Answers
* What data do we actually have?
* How clean, complete, and consistent is it?
* What patterns, trends, or anomalies exist?
* What relationships seem plausible or implausible?

## EDA vs Descriptive Analytics
* Descriptive analytics reports known metrics (means,KPIs)
* EDA asks open-ended questions about the data
* EDA is iterative and adaptive, not one-shot
* EDA often combines visuals with narrative reasoning

## EDA Workflow

### 1.Understand data sources
* completeness,freshness, quality, meaning etc.

### 2. The variables and their distributions
* continuous (distributions) or Categorical (unique values)
### 3. Describe Anomalies
* outliers, errors, duplicates

## Pre-Analysis
* Before visualization, you must assess data hygiene.
* Shape: How many rows (observations) and columns (features)?
* Types: Are numerical columns accidentally stored as strings?
* Missingness: Heatmaps of null values—is missing data random or systematic?
* Uniqueness: Identifying duplicate rows or columns with zero variance (single value).

## Types of EDA
* Statistics
    * Descriptive: organizing and summary
    * Inferential: draw conclusions
* Data Analysis
    * Univariate
    * Bivariate
    * Multivariate

## Univariate Non-Graphical Analysis
* One variable at a time

### Descriptive
* mean, median, mode,minimum and maximum,interquartile range, variance, coefficient of variance,standard deviation, quartiles etc.
* Skewness
* Kurtosis 

### Inferential
* One sample t-test
* Dependent t-test
* One-way ANOVA (F-test)
* Number of missing values
* Identification of outliers

## Univariate Graphical Analysis
* Histogram – how are values distributed?
* Frequency Bar/Pie Chart – Frequencies of different categories
* Box Plot – Helps visualize minimum, maximum, median, and quartiles. It is beneficial in identifying outliers in the data.
* Density curve - is the data is bimodal, normally distributed, skewed, etc.?

## Inferential Testing
* **Example:**
    * Checking if the price of sugar has statistically significantly risen from the generally accepted price by using sample survey data. Hypothesis tests such as the Z or T-test answer such questions.
* **Types:**
    * Z Test
    * Chi-Square Test
    * One-Sample T-Test
    * Kolmogorov-Smirnov Test

## Bi/Multi-variate Analysis
* Analyze two or more variables using descriptive and mainly inferential statistics to understand their relationship.

### Non-Graphical
* Aggregation and summarization using categorical and a numerical variable
* Regression
* Correlation Coefficient
* PCA
* Chi-Square Test

### Graphical
* Scatter Plot
* Heat Map
* Stacked / Dodged
* Bar Chart
* Multiline Chart

## Effectiveness
* Conveys real insights: ease of interpretation
* Accurate: Lie factor = size of visual effect/size of data effect.
* Efficient: minimize data-ink ratio and chart-junk
* Aesthetics: must not offend viewer's senses (e.g. moire patterns).
* Adaptable: can adjust to serve multiple needs.

## Anti-patterns

### Example 1
* The Mistake: Relying solely on the mean (average) to describe data.
* Scenario: "The average bill is $50."
* The Reality: Your data might be bimodal (two humps: lots of $20 bills and lots of $80 bills, but almost no $50 bills).
* The Fix: Always visualize the distribution (Histogram) along with the mean.

### Example 2 
* The Mistake: Assuming correlation equals causation.
* Scenario: "Ice cream sales correlate with drowning deaths."
* The Reality: Both are driven by a third variable: Summer (Heat).
* The Fix: Use domain knowledge to sanity-check strong correlations. 
    * Don'tblindly feature-engineer based on correlation matrices.

### Example 3
* The Concept: Four datasets that have identical descriptive statistics (mean, variance, correlation) but look completely different when plotted.
* The Lesson: If you only look at the summary table, you miss the outliers and non-linear patterns.
* The Fix: Always visualize. Summary stats are lossy compression; plots are the raw truth.

## Conclusion
* EDA is not just making charts; it is the phase where you build your intuition and validate your data quality.
* Next Steps: Proceed to Feature Engineering (Handling the outliers and missing values identified during EDA).
