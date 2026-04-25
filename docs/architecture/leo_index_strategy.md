# LEO index strategy and DMSP inclusion decision

This document records how the Swarm-derived LEO ionospheric index relates to ground truth and when DMSP remains out of scope for the SWMI pipeline.

## Role of the LEO index

The LEO index is built from Swarm magnetic residuals (reference field subtracted, gap-aware persistence, sub-indices by magnetic latitude and MLT sector). It is a **precursor-state signal** for ionospheric disturbance, not a replacement for SuperMAG dB/dt targets.

Ground truth for the forecasting task remains **per-station gap-aware** `dbdt_horizontal_magnitude` from SuperMAG NEZ components, as defined in the target architecture.

## Validation against SuperMAG (P0-C1)

Before adding any additional LEO missions, the pipeline checks whether the existing LEO features carry information that aligns with **global** SuperMAG variability over the same period.

Implementation: `src/swmi/evaluation/leo_index_validation.py` and `notebooks/03_leo_index_validation.ipynb`.

Summary of the procedure:

1. Load one or more monthly feature matrices (Parquet) that include LEO columns and per-station targets `dbdt_horizontal_magnitude_{station}`.
2. Form a **global SuperMAG target** as the mean of all per-station targets at each timestamp (NaNs ignored per station).
3. For each LEO feature family listed in code (`leo_high_lat`, `leo_mid_lat`, `leo_dayside`, `leo_nightside`, `leo_index_global` and related suffixed columns), scan **Pearson correlation** against that global target over a grid of **integer minute lags**.
4. **Lag sign:** positive lag means the LEO series is shifted **forward in time** relative to the target series, i.e. **LEO leads** the SuperMAG global mean by that many minutes (same convention as `pandas.Series.shift(lag)` on the LEO column).
5. The **best** (feature, lag) pair is the one with the largest absolute correlation; variance explained is reported as **R² = ρ²** for that pair.
6. **Per-station** diagnostics: for the best (feature, lag), correlations are also computed between shifted LEO and each station’s `dbdt_horizontal_magnitude_{station}`, with optional geographic context if `glat_`, `glon_`, `mlat_`, `mlon_` per-station columns exist.

Outputs are written under `results/validation/leo_index/` (CSV tables, figures, markdown report, JSON summary) when `run_leo_index_validation()` is used.

## DMSP defer / escalate criteria (P0-C2)

**Default policy:** DMSP is **not** implemented in P0. Escalation is a **governance decision** after the LEO validation above, not a silent code path.

| Outcome | Condition | Project implication |
|--------|------------|----------------------|
| **Defer DMSP** | Best global variance explained **R² ≥ 0.10** (i.e. at least **10%** of variance in the mean SuperMAG dB/dt target explained by the best lagged LEO feature) | Keep LEO-only architecture for P0; document any residual limitations; DMSP remains deferred unless a separate product decision reopens scope. |
| **Escalate DMSP review** | Best global **R² < 0.10** | **Does not** auto-implement DMSP. It triggers a **phase review**: justify extra retrieval complexity, data rights, and cadence match, or invest in LEO feature engineering and alternative validation before adding DMSP. |

The 10% threshold is the default agreed in the task register; it is encoded as `DMSP_DEFER_R2_THRESHOLD = 0.10` in `leo_index_validation.py`. If this threshold is ever changed, the constant and this document should be updated together, and any prior studies should be re-tagged to the old threshold for comparability.

## Warnings and scope

- **MMS** is out of scope for this pipeline (operational/contract reasons).
- **DMSP** is deferred unless the escalation review above justifies it after LEO proves insufficient under the agreed metric.
- The validation metric is **diagnostic** (correlation / explained variance on historical aligned data), not a replacement for a held-out sequence forecast metric; use the sequence builder and model evaluation for end-to-end skill.

## References

- Target definition: `docs/architecture/target_variable.md`
- LEO validation code: `src/swmi/evaluation/leo_index_validation.py`
- Notebook: `notebooks/03_leo_index_validation.ipynb`
