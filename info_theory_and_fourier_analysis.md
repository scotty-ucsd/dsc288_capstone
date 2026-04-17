# info theory

Dependency is a key discriminating statistic that is commonly used to understand how systems operate. The standard tool used to identify dependency is cross correlation. Considering two variables, $x$ and $y$, the correlation analysis essentially tries to fit the data to a 2-D Gaussian cloud, where the nature of the correlation is determined by the slope and the strength of correlation is determined by the width of the cloud perpendicular to the slope.

By nature, the response of the radiation belts to solar wind variables is nonlinear [Reeves et al., 2011; Kellerman and Shprits, 2012] as evidenced by the triangle distribution in $J_e$ versus $V_{sw}$ seen in Figures 1a–1c. Such a distribution is not well described by a Gaussian cloud of points and is not well characterized by a slope.

For such distributions, it is better to use a statistical-based measure such as **mutual information (MI)** [Tsonis, 2001; Li, 1990; Darbellay and Vajda, 1999]. Mutual information between two variables, $x$ and $y$, compares the uncertainty of measuring variables jointly with the uncertainty of measuring the two variables independently. The uncertainty is measured by the entropy. In order to construct the entropies, it is necessary to obtain the probability distribution functions, which in this study are obtained from histograms of the data based on discretization of the variables (i.e., bins).

Suppose that two variables, $x$ and $y$, are binned so that they take on discrete values, $\hat{x}$ and $\hat{y}$, where:

$$x \in \{\hat{x}_1, \hat{x}_2, \dots, \hat{x}_n\} \equiv \aleph_1; \quad y \in \{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_m\} \equiv \aleph_2 \tag{1}$$

The variables may be thought of as letters in alphabets $\aleph_1$ and $\aleph_2$, which have $n$ and $m$ letters, respectively. The extracted data can be considered as sequences of letters. The entropy associated with each of the variables is defined as:

$$H(x) = -\sum_{\aleph_1} p(\hat{x}) \log p(\hat{x}); \quad H(y) = -\sum_{\aleph_2} p(\hat{y}) \log p(\hat{y}) \tag{2}$$

where $p(\hat{x})$ is the probability of finding the word $\hat{x}$ in the set of $x$ data and $p(\hat{y})$ is the probability of finding word $\hat{y}$ in the set of $y$ data. To examine the relationship between the variables, we extract the word combinations $(\hat{x}, \hat{y})$ from the data set. The joint entropy is defined by:

$$H(x, y) = -\sum_{\aleph_1\aleph_2} p(\hat{x}, \hat{y}) \log p(\hat{x}, \hat{y}) \tag{3}$$

where $p(\hat{x}, \hat{y})$ is the probability of finding the word combination $(\hat{x}, \hat{y})$ in the set of $(x, y)$ data. The mutual information is then defined as:

$$MI(x, y) = H(x) + H(y) - H(x, y) \tag{4}$$

In the case of Gaussian distributed data, the mutual information can be related to the correlation function; however, it also includes higher-order correlations that are not detected by the correlation function. Hence, MI is a better measure of dependency for variables having a nonlinear relationship [Johnson and Wing, 2005].

While MI is useful to identify nonlinear dependence between two variables, it does not provide information about whether the dependence is causal or coincidental. Herein, we use the working definition that if there is a transfer of information from $x$ to $y$, then $x$ causes $y$. In this case, it is useful to consider conditional dependency with respect to a conditioner variable $z$ that takes on discrete values, $\hat{z} \in \{z_1, z_2, \dots, z_n\} \equiv \aleph_3$.

The **conditional mutual information (CMI)** [Wyner, 1978]:

$$CMI(x, y | z) = \sum_{\aleph_1\aleph_2\aleph_3} p(\hat{x}, \hat{y}, \hat{z}) \log \frac{p(\hat{x}, \hat{y} | \hat{z})}{p(\hat{x} | \hat{z}) p(\hat{y} | \hat{z})} = H(x, z) + H(y, z) - H(x, y, z) - H(z) \tag{5}$$

determines the mutual information between $x$ and $y$ given that $z$ is known, where $p(\hat{x}|\hat{z})$ is the probability of finding the word $\hat{x}$ in the set of $x$-data given $\hat{z}$. In the case where $z$ is unrelated, $CMI(x, y|z) = MI(x, y)$, but in the case that $x$ or $y$ is known based on $z$, then $CMI(x, y|z) = 0$. CMI therefore provides a way to determine how much additional information is known given another variable. CMI can be seen as a special case of the more general conditional redundancy that allows the variable $z$ to be a vector [e.g., Prichard and Theiler, 1995; Johnson and Wing, 2014].

# Functional Precision: Fourier Analysis in Time Series Machine Learning

Fourier analysis facilitates the transformation of a discrete time-domain signal into a frequency-domain representation. In the context of machine learning (ML), this process acts as a deterministic feature extractor that identifies periodic regularities, reduces stochastic noise, and enhances the signal-to-noise ratio (SNR) prior to model training.

---

## 1. Operational Granularity: The Discrete Fourier Transform (DFT)

The fundamental mechanism is the **Discrete Fourier Transform (DFT)**. For a time series sequence $x[n]$ of length $N$, the transformation is governed by the following mathematical framework:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i \frac{2\pi}{N} kn}$$

### Mechanistic Clarity
* **Vector Decomposition:** The DFT treats the time series as a vector and projects it onto a basis of orthogonal complex sinusoids.
* **Complex Coefficients:** Each resulting $X[k]$ is a complex number where the magnitude ($|X[k]|$) represents the **Amplitude** (strength of the cycle) and the argument ($\angle X[k]$) represents the **Phase** (temporal offset).
* **Symmetry:** For real-valued time series, the output is conjugate symmetric, meaning only $N/2$ components are non-redundant for feature selection.



---

## 2. Procedural Transparency: Applications in the ML Pipeline

### A. Automated Feature Engineering via Spectral Density
Standard ML models (e.g., Random Forests, XGBoost) struggle to "learn" periodicity from raw timestamps. 
* **Logic:** By calculating the **Power Spectral Density (PSD)**, we identify frequencies with the highest energy.
* **Implementation:** The $k$ indices with the highest $|X[k]|^2$ values are extracted. These indices correspond to physical cycles (e.g., a spike at $k$ equivalent to 1/24 hours). 
* **Result:** These magnitudes are fed into the model as static features, providing the learner with explicit knowledge of seasonality that would otherwise require deep "look-back" windows.

### B. Denoising via Spectral Thresholding (Low-Pass Filtering)
Raw time series data often contains high-frequency thermal noise or measurement artifacts that lead to overfitting.
* **Logic:** If the underlying physical process is known to change slowly, high-frequency components are assumed to be non-informative noise.
* **Implementation:** A "Brick Wall" filter is applied in the frequency domain:
    1. Transform data to frequency domain via FFT.
    2. Set $X[k] = 0$ for all $k > k_{cutoff}$.
    3. Perform **Inverse FFT (IFFT)** to return to the time domain.
* **Result:** A smoothed "Global Trend" signal that allows the ML model to focus on the macro-movements rather than micro-oscillations.



### C. Dimensionality Reduction (Axiomatic Scaffolding)
In high-resolution sensor data, the input dimension $N$ can be massive, causing the "Curse of Dimensionality."
* **Logic:** According to the principle of **Sparsity**, most real-world signals are "compressible" in the frequency domain. 
* **Implementation:** Instead of using $N$ time steps as features, we use the first $m$ Fourier coefficients (where $m \ll N$).
* **Justification:** This preserves the global structure of the series while discarding redundant local fluctuations, reducing model complexity and training latency.

---

## 3. Comparative Validation: Fourier vs. Time-Lagged Features

| Parameter | Time-Lagged Approach (Lags) | Fourier Analysis Approach |
| :--- | :--- | :--- |
| **Dependency** | Relies on local autocorrelation. | Relies on global periodic structure. |
| **Interpretability** | High for immediate past ($t-1$). | High for cyclical patterns (Seasonality). |
| **Data Requirements** | Sensitive to missing values (NaNs). | Requires uniform sampling (Equidistant steps). |
| **Model Load** | Increases feature count linearly with lag. | Compresses long-term trends into few coefficients. |

### Justification of Parameters
Fourier analysis is chosen over simple moving averages (SMA) when the goal is **unbiased periodicity detection**. While an SMA can smooth data, it introduces a "phase lag" (time delay) into the features. Fourier filtering, when applied symmetrically, preserves the temporal alignment of peaks and troughs, ensuring the ML model’s predictions are not artificially shifted in time.
