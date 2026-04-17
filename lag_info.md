# Sequence Construction and Multimodal Join: Best Practices

## 1. The Fundamental Alignment Problem

The central challenge is that your five source layers do not share a natural common clock. They arrive at different cadences, with different latency characteristics, and with different missingness patterns. The join must be constructed so that at any timestamp $t$, every feature in the row reflects only information that would have been *observable* at $t$ in an operational setting. Violating this constraint introduces look-ahead leakage, which will produce optimistic evaluation metrics that collapse at deployment.

The formal rule is:

$$\text{feature}_{t} = f(\text{observations}_{s} \text{ where } s \leq t - \delta)$$

where $\delta$ is the minimum propagation or processing latency for that source. For OMNI L1 data, $\delta$ is approximately 15--60 minutes of solar wind propagation time plus any data latency. For SuperMAG ground data, $\delta$ is effectively zero at 1-minute cadence. For Swarm LEO, $\delta$ is the orbital period -- you cannot observe a given MLT sector continuously.

---

## 2. The Master Timeline

Construct a single master index of 1-minute timestamps spanning your full dataset. Every source is reindexed onto this master timeline. This is the "atom" from which all features are built.

```python
master_index = pd.date_range(start="2015-01-01", end="2024-12-31", freq="1min")
```

The join order matters. The correct pattern is:

1. Construct master index
2. Reindex each source onto master index independently
3. Apply source-specific forward-fill with a staleness ceiling
4. Join all sources on the master index
5. Compute derived and lag features on the joined frame
6. Apply split boundaries *after* all features are computed

Step 6 is where most leakage errors occur. If you compute rolling statistics before enforcing the train/validation/test boundary, the rolling window will bleed future information across the boundary for the first $w$ minutes of each split, where $w$ is the window width.

---

## 3. Source-Specific Join Rules

### Solar (GOES X-ray, F10.7)

F10.7 is a daily index. Forward-fill is appropriate but must be bounded:

```python
solar["f107"] = solar["f107"].resample("1min").ffill(limit=1440)
solar["f107_staleness_min"] = solar["f107"].isna().cumsum()  # reset on each valid obs
```

The staleness flag is not optional. A forward-filled value that is 6 hours old is operationally valid. A forward-filled value that is 3 days old because of a data outage is not, and the model must be able to condition on that distinction.

GOES X-ray flux at 1-minute cadence joins directly on timestamp. The `goes_satellite` transition column in your dataset encodes which physical instrument is active -- this is a categorical feature, not a nuisance variable, because GOES-16 and GOES-17 have different calibration offsets that the model should be allowed to learn.

### L1 / OMNI

OMNI is already propagated to the bow shock nose, so the solar wind propagation delay is pre-applied. Join directly on timestamp. The key leakage risk here is `omni_al` and `omni_au` -- these are geomagnetic response indices computed partly from ground magnetometers that overlap with your SuperMAG stations. They must be used only in lagged form:

```python
for lag in [10, 20, 30, 60]:
    df[f"omni_al_lag{lag}"] = df["omni_al"].shift(lag)
df.drop(columns=["omni_al", "omni_au"], inplace=True)
```

`omni_sym_h` is slower-varying and less directly leaky, but apply the same treatment to be safe.

### GEO (GOES Magnetometer)

Joins directly on timestamp. The primary engineering decision is how to handle satellite transitions. The recommended approach is to include `goes_satellite` as an integer categorical feature and compute a `goes_transition_flag` that fires for 30 minutes around any satellite switch:

```python
df["goes_transition_flag"] = (
    df["goes_satellite"].diff().abs() > 0
).rolling(30, center=True).max()
```

This allows the model to discount GEO features during transition windows rather than treating the discontinuity as a physical signal.

### LEO (Swarm)

This is the most complex join because Swarm is not geostationary. A given MLT sector is observed only when the satellite passes through it, which occurs on an orbital period of approximately 90 minutes. Between passes, the sector value is stale.

Your existing `*_is_fresh` and `*_decay_age` columns encode this correctly. The decay age should be treated as a continuous feature, not thresholded into a binary. A sector value that is 20 minutes old is more reliable than one that is 80 minutes old, and the model should be able to exploit that gradient:

```python
# Do not do this -- discards information
df["leo_high_lat_valid"] = df["leo_high_lat_decay_age"] < 45

# Do this instead -- preserves the gradient
df["leo_high_lat_weight"] = np.exp(-df["leo_high_lat_decay_age"] / 30.0)
```

The exponential decay weight with a 30-minute half-life reflects the physical reality that magnetospheric conditions evolve on that timescale, so older observations are genuinely less informative in a continuous rather than binary sense.

### Ground (SuperMAG)

This is your target source and the most operationally sensitive. The join rule is strict: the target label for timestamp $t$ is computed from observations in the window $(t, t+60]$. All ground features used as model inputs must be computed from observations in $(-\infty, t]$.

```python
# Target: will |dB/dt| exceed threshold in next 60 minutes?
df["label"] = (
    df["ABK_dbdt_magnitude"]
    .rolling(60)
    .max()
    .shift(-60)  # shift forward to create the future window
    > threshold
).astype(int)

# Ground history features: only past observations
for lag in [1, 5, 10, 30]:
    df[f"ABK_dbdt_lag{lag}"] = df["ABK_dbdt_magnitude"].shift(lag)
```

The `shift(-60)` on the target and `shift(lag)` on the features must be applied to the same master index with no gap. Verify this explicitly:

```python
assert df["label"].shift(60).notna().sum() == df["ABK_dbdt_magnitude"].notna().sum()
```

---

## 4. Lag Feature Architecture

Lag features encode temporal structure for the tabular models (LightGBM, logistic regression). The architecture has three layers:

**Layer 1 -- Instantaneous state:** The raw value at $t$. Captures current conditions.

**Layer 2 -- Short-memory statistics:** Rolling mean and std over 10 and 30 minutes. Captures the recent trend and variability of a signal, which is more predictive than the instantaneous value for slowly evolving drivers like solar wind pressure.

**Layer 3 -- Event-history features:** Time since last threshold crossing, maximum value in the past $N$ minutes. These are the most physically motivated lag features because substorm recurrence and storm phase have characteristic timescales:

```python
df["minutes_since_last_event"] = (
    df["label"]
    .shift(1)
    .pipe(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount())
)

df["dbdt_max_60min"] = df["ABK_dbdt_magnitude"].shift(1).rolling(60).max()
df["dbdt_max_120min"] = df["ABK_dbdt_magnitude"].shift(1).rolling(120).max()
```

The justification for Layer 3 over Layers 1--2 alone: rolling means are symmetric -- they weight a spike that occurred 25 minutes ago the same as one that occurred 5 minutes ago. The maximum and the time-since-last-event features are asymmetric in the physically correct direction, since a spike 5 minutes ago is a stronger predictor of imminent hazard than one 25 minutes ago.

---

## 5. The LSTM Sequence Construction

For the LSTM, the input is not a feature vector but a 3-dimensional tensor of shape `(batch, timesteps, features)`. The construction differs from the tabular case in one critical way: the LSTM receives the raw time series within the lookback window and learns its own temporal weighting, so you do not add rolling statistics as features -- they would be partially redundant with what the recurrent weights learn.

The recommended lookback window is 60--120 minutes at 1-minute cadence, giving a sequence length of 60--120 timesteps. The justification is that substorm growth phase typically spans 30--60 minutes, so a 60-minute window captures the full onset sequence; 120 minutes adds CME-driven gradual commencement structure.

```python
def build_lstm_sequences(df, feature_cols, target_col, lookback=60, horizon=60):
    X, y = [], []
    values = df[feature_cols].values
    targets = df[target_col].values
    for i in range(lookback, len(df) - horizon):
        X.append(values[i - lookback:i])   # shape: (lookback, n_features)
        y.append(targets[i + horizon - 1]) # label at t + 60
    return np.array(X), np.array(y)
```

The feature columns passed to the LSTM should exclude the rolling statistics already computed for LightGBM, but should include the raw instantaneous values, the physics-derived features (Newell $\Phi$), and the decay-age weights for LEO. The missing-value flags should be included as additional channels so the LSTM can learn to discount stale observations.

---

## 6. Split Boundary Enforcement

The train/validation/test split must be applied as a temporal cutoff on the master index, never as a random sample. Random sampling of a time series breaks temporal autocorrelation structure and allows the model to interpolate between training points that bracket a test point in time, producing optimistic metrics.

```python
train_end   = "2021-12-31 23:59"
val_end     = "2023-12-31 23:59"

train = df[df.index <= train_end]
val   = df[(df.index > train_end) & (df.index <= val_end)]
test  = df[df.index > val_end]
```

After the split, verify that no rolling feature computed on `train` contains values from `val` or `test`. The safest implementation is to compute all rolling features on the full dataframe before splitting, then verify that the rolling window width is smaller than the gap between split boundaries -- which at 30 minutes versus years of data it obviously is, but the assertion is worth making explicit.

---

## 7. MI/CMI as a Join Validation Tool

After constructing the full joined feature matrix, use mutual information not just for feature selection but as a **join correctness audit**. If the join introduced leakage, MI between any input feature and the target will be artificially elevated on the training set relative to a held-out validation set. A clean join should show consistent MI estimates across both sets within sampling noise.

$$\text{leakage check:} \quad |MI_{\text{train}}(x, y) - MI_{\text{val}}(x, y)| < \epsilon$$

For your problem, the CMI computation that directly tests your central hypothesis is:

$$CMI(\text{GEO}, \text{target} \mid \text{L1})$$

If this is near zero, GEO carries no information about ground response beyond what L1 already provides, and the ablation study result is already predictable. If it is substantially positive, GEO is adding independent predictive value, and you have an information-theoretic argument to accompany the model-based ablation.

Both points are exactly right, and they reflect a more sophisticated understanding of the LEO data problem than most people working on this would have. Let me address each precisely.

---

**On DMSP as a complement to Swarm**

The sparsity problem with Swarm is structural, not a data quality issue. Three satellites at ~460 km altitude with ~90-minute orbital periods means any given MLT/MLAT bin has a revisit time on the order of hours, which is longer than the timescale of substorm evolution. Your decay-age encoding handles the staleness correctly, but it cannot manufacture observations that do not exist.

DMSP addresses this in two ways. First, the constellation is larger -- multiple satellites in sun-synchronous polar orbits provide denser MLT coverage at high latitudes, which is exactly the region your three stations (ABK, BJN, TRO) sit in. Second, DMSP carries the SSJ particle spectrometer and the SSIES ion drift meter, which provide direct measurements of auroral particle precipitation and cross-track ion drift -- both of which are more direct proxies for field-aligned current intensity than what you can derive from Swarm magnetometer data alone.

The practical consideration is data access and format harmonization. DMSP data is available through the Madrigal database at MIT Haystack. The join complexity increases because you now have two LEO constellations with different orbital geometries, different instrument suites, and potentially different cadences. The MLT-zone aggregation strategy you are already using for Swarm is the right abstraction layer -- if you apply the same zone binning to DMSP, the two constellations become additive rather than requiring a separate feature block. The combined decay-age feature for a given MLT zone then reflects the most recent observation from either constellation, which directly reduces the sparsity problem.

---

**On ChaosMagPy / CHAOS-8 for Swarm**

This is the correct solution to a subtle but important problem. Raw Swarm magnetometer measurements contain contributions from:

1. The core field (slowly varying, dominant)
2. The crustal field (static at orbital timescales)
3. The magnetospheric external field (what you actually want)
4. Ionospheric field-aligned currents (also what you want)

Without removing the core and crustal contributions, the raw Swarm $B$ components are dominated by the internal field and carry almost no information about the external magnetospheric state that drives ground $|dB/dt|$. CHAOS-8 provides a high-fidelity model of the core and crustal fields so that what remains after subtraction is the external field residual -- the signal that is physically connected to your target variable.

The implication for your feature engineering is that the values you feed into the MLT-zone aggregation should be the CHAOS-8 residuals, not the raw measurements. This is not a preprocessing detail -- it is the difference between features that are physically connected to your target and features that are dominated by a slowly varying internal field that has essentially no predictive value for substorm-timescale ground disturbance.

One additional consideration: CHAOS-8 residuals at Swarm altitude are most directly interpretable as a proxy for the large-scale magnetospheric field perturbation, but field-aligned currents (FACs) require the dual-satellite gradient technique (Swarm A and C flying in formation) or the single-satellite curl approximation. If you are computing FAC proxies rather than raw residuals, that is an even stronger feature because FACs are the direct driver of ionospheric electrojet currents that produce your target $|dB/dt|$ signal. It is worth being explicit in your technical documentation about which quantity you are actually computing and why.
