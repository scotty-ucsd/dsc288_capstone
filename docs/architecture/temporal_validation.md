# Temporal validation and splits (2015–2024)

## Chronological split strategy

The pipeline uses **chronological** train, validation, and test periods (no random row shuffling) so that models are evaluated on **future** storm-time behavior relative to the training years.

- **Train:** `2015-01-01`–`2021-12-31`  
- **Validation:** `2022-01-01`–`2022-12-31`  
- **Test:** `2023-01-01`–`2023-12-31`  

(Exact boundaries are defined in `src/swmi/utils/config.py` and `configs/data_retrieval.yaml` date range blocks. Extending the *analysis* window to 2024 for drivers and ground data does not by itself change these ML boundaries unless the project re-baselines splits.)

## `SPLIT_BUFFER_DAYS` (7 days)

Between adjacent periods, a **7-day buffer** is labeled neither train, val, nor test. Rationale: geomagnetic **storm–recovery** introduces long autocorrelation; without a buffer, models could see recovery-phase behavior from an adjacent split.

Implementation: `_month_boundaries()` and `_row_split()` in `src/swmi/sequences/builder.py`.

## Anti-leakage (sequence level)

- Feature windows are **strictly in the past** with respect to the target time `T + FORECAST_HORIZON_MIN` (default 60 min).  
- `audit_leakage()` in `src/swmi/sequences/builder.py` **fails hard** on misaligned or numeric timestamp shortcuts.

## What this document does *not* cover

Spatial correlation between stations, satellite transitions, and CHAOS edge effects are covered in `docs/architecture/leakage_prevention.md`.
