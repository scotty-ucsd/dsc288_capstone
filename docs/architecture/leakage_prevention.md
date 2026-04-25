# Leakage prevention

This file enumerates the main **information leakage** vectors in space-weather ML and how this pipeline mitigates them. It complements `docs/architecture/temporal_validation.md` (split design).

## 1) Future data in features

- **Feature windows** end at time `T`; **targets** are at `T + FORECAST_HORIZON_MIN` (e.g. 60 min dB/dt), encoded explicitly in the sequence builder.  
- **Audit:** `audit_leakage()` rejects wrong horizons or non-timestamp arrays.  
- **dB/dt** uses the **backward** difference (`DBDT_METHOD`) so no future in-window samples are used.

## 2) Overlapping storm events across splits

- **7-day `SPLIT_BUFFER_DAYS`** strips storm-recovery tails from split seams (see `temporal_validation.md`).  
- Sequences that would straddle a split label are **dropped** (`_valid_sequence_starts`).

## 3) Spatial autocorrelation (stations)

- Multi-station outputs are **wide** with per-station columns; the model may still *learn* spatial structure within a time slice — that is a **generalization** question, not a temporal label leak. Document when reporting “per-station” vs “global” metrics.  
- SuperMAG MLT/MLT metadata from the API is preferred over recomputing from apex for station context.

## 4) Satellite transitions (GOES)

- Merges are **deterministic** using `goes.satellite_priority[year]`. **GOES-19** (2024) follows the same rules as other GOES-R sats.  
- Source column `goes_source_satellite` records which sat supplied each minute.

## 5) CHAOS model and time

- **CHAOS-8.5** coefficients are fixed for a model generation; the pipeline uses a **single** CHAOS file period-wide and processes **1 calendar month** of Swarm at a time in `build_leo_index_month()` to avoid OOM, not to “peek” at future data. Sub-index state is still **causal** within the month (only past 1-Hz data for each minute’s aggregation).  
- If the reference field were switched mid-study, that would be a **science** break, not timestamp leakage; keep `REFERENCE_FIELD` constant across the training record.

## 6) NaN / missingness

- Missing stations use **mask-aware** training (`torch.nanmean` or boolean masks) so the optimizer is not **implicitly** told which stations are missing via fake zeros.  
- Global indices (SME, etc.) remain valid **fallbacks** if per-station columns fail.

## Rationale: 1-month CHAOS chunks

Large multi-year Swarm pulls exceed typical RAM; **month-by-month** LEO index builds match HPC practice and keep failures localized with explicit Dask error surfacing in `leo_index.py` and worker logs under `logs/`.

## See also

- `src/swmi/sequences/builder.py` — splits and `audit_leakage`  
- `src/swmi/preprocessing/validation.py` — `validate_sources`, physical range checks  
- `docs/architecture/leo_index_strategy.md` — DMSP / Phase 2 scope
