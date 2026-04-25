Here is the final, comprehensive, and corrected task list for Prompt 1B.

***

# Prompt 1B Input: Comprehensive Priority 0 Task List

## Project Context

**SWMI dB/dt Forecasting Pipeline**: Predict ground-level geomagnetic disturbance (dB/dt magnitude) at 60-minute horizon using 120-minute multi-source input window. Chain: **Sun (GOES X-ray precursor) → L1 (OMNI) → GEO (GOES mag) → LEO (Swarm) → Ground (SuperMAG)**. GOES X-ray is a **solar activity precursor** (leads L1 by 1–3 days), not a synchronous Earth measurement. **Target: horizontal dB/dt magnitude in NEZ coordinates as proxy for GICs at all available SuperMAG stations globally.**

***

## Approved Scientific Invariants (Do Not Change)

| Setting | Value | Consequence of Change |
|---|---|---|
| `REFERENCE_FIELD` | `"IGRF"` | Invalidates all LEO residuals if changed mid-study |
| `QDLAT_HIGH_LAT_MIN` | `55.0°` | Redefines auroral zone; all LEO parquet must be reprocessed |
| `DECAY_HALFLIFE_MIN` | `10.0` | Alters LEO persistence physics |
| `DBDT_METHOD` | `"backward"` | `"centered"` introduces future data leakage |
| `FORECAST_HORIZON_MIN` | `60` | Changes the scientific question |
| `MASTER_CADENCE` | `"1min"` | Requires full re-ingestion if changed |
| `SPLIT_BUFFER_DAYS` | `7` | Storm-recovery autocorrelation buffer |

***

## Approved Architecture Decisions

| Decision | Approved Choice |
|---|---|
| GOES primary satellite priority | GOES-16 primary, 15 backup for 2015–2016; best available modern for 2017+ |
| GOES X-ray inclusion | **P0-Must** — user explicitly requires; data product confirmed available |
| DMSP inclusion | Defer to Phase 2 unless LEO index proves weak (decision criteria in C2) |
| CHAOS vs. IGRF | IGRF fallback acceptable with version lock and documentation |
| SuperMAG target coordinate system | **NEZ** — N and E are horizontal, direct GIC proxy; no rotation needed |
| SuperMAG station scope | **All available stations** — dynamic inventory per month (~300–542 stations) |
| Target architecture | **Option A: Per-station dB/dt** — multi-output regression with NaN masking |
| SuperMAG API flags | Default NEZ + `"mlt,mag,geo"` for station context; never use `.mag` or `.geo` for dB/dt |
| Sequence gap threshold | Reject if >10% window gap |

***

## Complete Ordered Task List

### Phase 1: Critical Blockers (Execute First)

| Order | Task ID | Title | Priority | Status | Dependencies | Target Location | Definition of Done |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | **P0-A5** | Fix `retrieve_supermag.py` schema call | P0-Must | Ready now | None | `scripts/retrieve_supermag.py` | `validate_output_schema` call matches function signature; no `allow_duplicates` parameter; long-format station data passes validation |
| 2 | **P0-A3** | Fix SuperMAG dB/dt gap-aware computation for all stations | P0-Must | Ready now | None | `src/swmi/preprocessing/cleaners.py` | dB/dt uses actual time intervals between valid points; gaps >1 min produce NaN with `dbdt_gap_flag`; NEZ N and E components used for `dbdt_horizontal_magnitude = sqrt(dbdt_n² + dbdt_e²)` |
| 3 | **P0-S1** | Implement dynamic SuperMAG station inventory retrieval | P0-Must | Ready now | None | `src/swmi/api/supermag.py` | `SuperMAGGetInventory` called per month; available station list saved to `data/external/supermag_inventory_YYYYMM.json`; station count and geographic coverage logged |
| 4 | **P0-S2** | Design per-station vs. global index target architecture | P0-Must | Ready now | P0-S1 | `docs/architecture/target_variable.md` | Document comparing Option A (per-station dB/dt), Option B (indices), Option C (hybrid); **Option A approved**; user decision recorded with justification |
| 5 | **P0-A1** | Implement `validate_sources.py` and `validate_feature_matrix.py` | P0-Must | Ready now | None | `scripts/validate_sources.py`, `scripts/validate_feature_matrix.py` | Both scripts importable, callable with `(year, month, fail_on_error=True)`, return bool; `run_pipeline.py` executes without `ModuleNotFoundError` |
| 6 | **P0-A4** | Per-month data completeness report with per-station coverage | P0-Must | Ready now | P0-A1, P0-S1 | `scripts/validate_sources.py` → `src/swmi/preprocessing/validation.py` | Per-month JSON/CSV report: % valid per source, gap locations, **per-station completeness**, satellite availability, station count |
| 7 | **P0-S3** | Implement all-station SuperMAG data retrieval | P0-Must | Ready now | P0-S1, P0-S2 | `scripts/retrieve_supermag.py` or `src/swmi/api/supermag.py` | Iterates over all available stations from inventory; retrieves **NEZ components only** (`N.nez`, `E.nez`, `Z.nez`); flags include `"mlt,mag,geo"` for station context; handles API rate limits with exponential backoff; saves long-format Parquet with `timestamp, station, n_nez, e_nez, z_nez, mlt, mlat, mlon, glat, glon` |
| 8 | **P0-S4** | Implement multi-station dB/dt computation | P0-Must | Ready now | P0-S3, P0-A3 | `src/swmi/preprocessing/cleaners.py` | Gap-aware dB/dt on NEZ N and E for all stations; outputs long-format DataFrame with `timestamp, station, dbdt_n, dbdt_e, dbdt_z, dbdt_horizontal_magnitude`; `dbdt_gap_flag` for gap-affected points |
| 9 | **P0-G1** | Design unified GOES architecture | P0-Must | Ready now | None | `docs/architecture/goes_unified.md`, `src/swmi/api/goes.py` (skeleton) | Document specifying: class hierarchy (base retriever + product-specific subclasses), output schemas, merge strategy, X-ray integration point, era detection logic |
| 10 | **P0-G2** | Implement unified GOES magnetometer retrieval | P0-Must | Ready now | P0-G1, P0-B3 | `src/swmi/api/goes.py` | Single `retrieve_goes_mag()` with automatic satellite routing (legacy ≤15, modern ≥16); primary/backup merge per approved decision; deprecated scripts removed; outputs canonical Parquet with `goes_bz_gsm`, `goes_source_satellite`, `goes_mag_missing_flag` |
| 11 | **P0-A2** | GOES multi-satellite merge strategy | P0-Must | Ready now | P0-G2 | `src/swmi/api/goes.py` | Documented primary/backup priority per year; merge produces single `goes_bz_gsm` column with `goes_source_satellite` and `goes_missing_flag`; no duplicate timestamps; gap analysis included in completeness report |
| 12 | **P0-G3** | Implement unified GOES X-ray flux retrieval | P0-Must | Ready now | P0-G1, P0-B3 | `src/swmi/api/goes.py` (XRS class) | `retrieve_goes_xray_month()` with: (1) era detection from satellite number, (2) era-specific quality filtering (legacy: 4-flag `good_data=0`; modern: 4-flag + `electron_correction_flag` reject `1,4` and significant contamination bits `8,16,32,64,128,256`), (3) J2000 time decoding, (4) canonical Parquet output with `xrsa_flux`, `xrsb_flux`, flags, satellite; **do not apply `au_factor`** |
| 13 | **P0-G6** | Implement X-ray cross-satellite calibration normalization | P0-Must | Ready now | P0-G3 | `src/swmi/preprocessing/cleaners.py` or `src/swmi/features/goes_xray.py` | Pipeline: (1) era-specific quality filtering, (2) log10 transform, (3) per-satellite quiet-Sun baseline subtraction, (4) NOAA scale factor application for legacy→modern continuity, (5) validation plot showing GOES-15 vs GOES-16 overlap agreement within 10% or documented offset |
| 14 | **P0-D2** | Integrate GEO data in feature matrix | P0-Must | Blocked | P0-A2 | `src/swmi/features/builder.py` | `goes_bz_gsm` present with missingness flag; rolling stats computed; documented as magnetopause compression proxy |
| 15 | **P0-G4** | Integrate GOES X-ray precursor features into feature matrix | P0-Must | Blocked | P0-G3, P0-G6, P0-A2 | `src/swmi/features/builder.py` | X-ray features: (1) `goes_xray_long_log` instantaneous, (2) `goes_xray_long_dlog_dt`, (3) `goes_xray_time_since_last_c/m/x_flare`, (4) `goes_xray_cumulative_m_class_24h`, (5) `goes_xray_max_flux_24h`; all **event-driven accumulators** (not rolling windows); documented as precursor features leading OMNI by 1–3 days |
| 16 | **P0-S5** | Design feature matrix for multi-station targets | P0-Must | Ready now | P0-S2, P0-S4 | `src/swmi/features/builder.py` | Feature matrix remains **global** (one row per timestamp); target is **multi-column** `dbdt_horizontal_magnitude_{station}` for each station; missing stations handled with `dbdt_missing_flag_{station}`; station context features (MLT, QDLat) from SuperMAG API, not `apexpy` |
| 17 | **P0-S6** | Implement multi-station sequence generation | P0-Must | Blocked | P0-S5, P0-E1 | `src/swmi/sequences/builder.py` | `X` shape: (samples, timesteps, features); `y` shape: (samples, stations); missing station masking in loss; station context (MLT, QDLat) per sequence from precomputed SuperMAG metadata |
| 18 | **P0-S7** | Implement multi-output baseline model | P0-Must | Blocked | P0-S6, P0-E2 | `scripts/04_train_baseline.py` | M0: persistence per station; M1: log-linear with multi-output; M2: gradient boosting with multi-output; **NaN-aware loss** (ignore missing stations); metrics per station and global average; per-station skill scores |

***

### Phase 2: LEO Index & Validation

| Order | Task ID | Title | Priority | Status | Dependencies | Target Location | Definition of Done |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 19 | **P0-C1** | Validate LEO index physics against global SuperMAG field | P0-Must | Needs decision | P0-A4, P0-S4 | `notebooks/03_leo_index_validation.ipynb` | Notebook showing LEO index correlation with dB/dt during known substorm events across **all stations**; spatial correlation maps; documented decision on index utility; if weak, triggers DMSP escalation |
| 20 | **P0-C2** | Define DMSP defer/escalate criteria | P0-Should | Blocked | P0-C1 | `docs/architecture/leo_index_strategy.md` | Document: "If LEO index explains >10% variance in global dB/dt field, defer DMSP to Phase 2; else, escalate to P0-Must"; clear go/no-go thresholds |
| 21 | **P0-B3** | GOES operational period table | P0-Should | Needs inspection | None | `docs/architecture/goes_operational_periods.md`, `configs/data_retrieval.yaml` | Table: satellite, start_date, end_date, primary_coverage_years, data_quality_notes; covers 2015–2023 (or 2024+ if GOES-19 included) |

***

### Phase 3: Infrastructure & Config

| Order | Task ID | Title | Priority | Status | Dependencies | Target Location | Definition of Done |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 22 | **P0-B4** | Precompute station metadata for all SuperMAG stations | P0-Should | Ready now | P0-S1 | `data/external/station_metadata/supermag_station_coords.parquet` | Parquet: station, glat, glon, mlat, mlon, qdlat (from `apexpy` if needed), mlt_offset, operational_start, operational_end; **primary source: SuperMAG API `mag` and `geo` flags**; `apexpy` only for QDLat if required |
| 23 | **P0-B1** | Migrate config to YAML | P0-Should | Ready now | None | `configs/data_retrieval.yaml`, `configs/feature_engineering.yaml` | YAML files contain all scientific invariants; `src/utils/config.py` loader; `config.py` deprecated or removed |
| 24 | **P0-B2** | Align directory structure with target | P0-Should | Ready now | P0-B1 | `src/config.py` → `configs/data_retrieval.yaml` | Interim outputs: `data/interim/{omni,goes,swarm,supermag}/`; features: `data/processed/features/`; all scripts updated |
| 25 | **P0-F1** | Dask-aware logging | P0-Should | Ready now | None | `src/swmi/utils/logger.py` | Worker logs captured to per-worker files or forwarded to central; LEO index and SuperMAG task failures explicit; `config.LOGS_DIR` used consistently |
| 26 | **P0-F2** | Physical range validation in schema | P0-Should | Ready now | None | `src/swmi/preprocessing/validation.py` | Configurable range checks per source; raises on violation; ranges documented (e.g., `|Bz| < 100 nT`, `200 < Vsw < 2000 km/s`, `dbdt_horizontal < 10000 nT/min`) |

***

### Phase 4: Sequences & Models (Integrated)

| Order | Task ID | Title | Priority | Status | Dependencies | Target Location | Definition of Done |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 27 | **P0-E1** | Refactor `build_sequences.py` for multi-station production | P0-Must | Ready now | P0-A3, P0-A4, P0-S5 | `src/swmi/sequences/builder.py` | Validity mask vectorized; sequences with >10% internal gaps rejected; performance <30s for 1 year; `audit_leakage` numerical fallback **removed** (fails hard); **multi-station `y` array with NaN for missing stations** |
| 28 | **P0-E2** | Build multi-output production baseline model script | P0-Must | Ready now | P0-E1, P0-S6 | `scripts/04_train_baseline.py` | Reads NPZ sequences with multi-station `y`; trains M0/M1/M2 with NaN-aware loss; saves metrics to `results/experiments/exp001_baseline/`; per-station and global metrics; **do not refactor old `baseline_model.py`** — write new from scratch |
| 29 | **P0-G5** | Document GOES X-ray deferral criteria | P0-Should | Ready now | P0-G3 | `docs/architecture/data_pipeline.md` | Document states: "X-ray included as P0-Must per user requirement; precursor physics documented; normalization pipeline specified; **not deferred**" |

***

## Warnings for Prompt 1B (Mandatory)

| # | Warning | Violation Consequence |
|:---|:---|:---|
| 1 | **Do not implement MMS** | Scope creep; mission not operationally available |
| 2 | **Do not implement DMSP unless C1 criteria met** | Scope creep; significant new development |
| 3 | **Preserve `config.py` scientific invariants** | Any change requires full reprocessing and invalidates prior results |
| 4 | **Do not modify `newell_coupling.py` physics** | Implementation is correct and validated |
| 5 | **GOES merge must be deterministic** | Arbitrary merge produces scientifically invalid GEO stream |
| 6 | **SuperMAG dB/dt must be gap-aware** | `diff()/60.0` is a silent scientific bug |
| 7 | **Dask worker failures must be explicit** | Silent task drops hide data quality issues |
| 8 | **Sequence anti-leakage must not have fallbacks** | `audit_leakage` numerical fallback is dangerous; remove it |
| 9 | **Baseline model script is MVP-only** | Do not refactor `baseline_model.py`; write new production script |
| 10 | **Preserve schema validation contract** | All Parquet outputs must pass `validate_output_schema` before write |
| 11 | **GOES must be consolidated into single `goes.py`** | Legacy and modern scripts deprecated; X-ray is P0-Must |
| 12 | **X-ray requires cross-satellite normalization** | Raw flux values not comparable across instrument generations |
| 13 | **X-ray is a solar precursor, not local Earth measurement** | Leads L1 by 1–3 days; feature engineering must use event-driven accumulators, not rolling windows |
| 14 | **Modern XRS has dual-channel redundancy and electron correction** | Use merged `xrsa_flux`/`xrsb_flux`; apply era-specific quality filtering; reject `electron_correction_flag == 1, 4` or significant contamination |
| 15 | **Do not apply `au_factor`** | Earth-observed flux is correct for geomagnetic forecasting |
| 16 | **Use NEZ coordinates for dB/dt target** | N and E are already horizontal; no rotation needed for GIC proxy |
| 17 | **Do not use .mag or .geo for dB/dt computation** | Those require extraction of horizontal components via rotation; NEZ is ready-to-use |
| 18 | **Primary target is horizontal magnitude** | `sqrt(dbdt_n² + dbdt_e²)`; vertical component (dbdt_z) is secondary |
| 19 | **MLT/mag/geo flags are for station context features only** | They provide station location metadata, not the target variable |
| 20 | **Do not hard-code station lists** | Use `SuperMAGGetInventory` for dynamic discovery; station availability varies monthly |
| 21 | **Use SuperMAG-provided MLT/mag coords when available** | More accurate than `apexpy`; eliminates fallback approximation bug |
| 22 | **Handle API rate limits for 300+ stations** | SuperMAG API has rate limits; implement exponential backoff; consider chunking |
| 23 | **Multi-station targets require NaN-aware loss** | Not all stations available every month; model must ignore missing stations in training |
| 24 | **Per-station dB/dt is high-dimensional output** | 300+ stations × 60-min horizon = massive `y` array; consider sparse representations or station sampling if memory issues arise |
| 25 | **Global indices (Option B) are scientifically valid alternative** | If per-station proves infeasible, SME/SML/SMU/SMR are established metrics with 40+ years of validation |

***

## User Decisions Required

| Decision | Options | Default if No Response | Impact |
|---|---|---|---|
| GOES-13 operational data inclusion | Include for 2015–2017 / Exclude, start with GOES-15 only | Include | Data volume 2015–2017 |
| GOES-19 data inclusion | Extend to 2024+ / Cap at 2023 | Cap at 2023 | Study period |
| CHAOS model file availability | Confirm `models/CHAOS-8.5` exists / IGRF-only | IGRF fallback | LEO index accuracy |
| LEO index validation threshold | >10% variance explained / other threshold | >10% | DMSP defer/escalate |
| Station sampling if memory issues | All stations / Subsample by data quality / Regional subsets | All stations | Model generalization |
| Missing station loss weighting | Equal weight per station / Weight by latitude / Weight by data completeness | Equal weight | Spatial bias in training |

***

## File Inventory for Prompt 1B

### Scripts to Create (New)

| File | Purpose | Task ID |
|---|---|:---|
| `src/swmi/api/goes.py` | Unified GOES retrieval (mag + X-ray) | P0-G1, P0-G2, P0-G3 |
| `scripts/validate_sources.py` | Per-month source validation | P0-A1 |
| `scripts/validate_feature_matrix.py` | Feature matrix validation | P0-A1 |
| `src/swmi/preprocessing/cleaners.py` | Gap-aware dB/dt, X-ray normalization | P0-A3, P0-G6 |
| `src/swmi/features/goes_xray.py` | X-ray precursor feature engineering | P0-G4 |
| `scripts/04_train_baseline.py` | Multi-output production baseline model | P0-E2, P0-S7 |
| `notebooks/03_leo_index_validation.ipynb` | LEO index physics validation | P0-C1 |

### Scripts to Modify

| File | Changes | Task ID |
|---|---|:---|
| `scripts/retrieve_supermag.py` | Fix schema call, implement inventory-based all-station retrieval, NEZ-only dB/dt | P0-A5, P0-S1, P0-S3, P0-S4 |
| `scripts/build_feature_matrix.py` | Add GEO features, X-ray precursor features, multi-station target columns | P0-D2, P0-G4, P0-S5 |
| `scripts/build_sequences.py` | Vectorize validity mask, remove leakage fallback, multi-station `y` | P0-E1 |
| `src/logger.py` | Dask-aware logging | P0-F1 |
| `src/schema.py` | Add physical range validation | P0-F2 |
| `src/station_context.py` | Use SuperMAG-provided MLT/mag, remove `apexpy` fallback | P0-B4 |
| `run_pipeline.py` | Integrate validation scripts, completeness report, all-station steps | P0-A1, P0-A4, P0-S1 |

### Scripts to Deprecate/Remove

| File | Replacement | Task ID |
|---|---|:---|
| `scripts/retrieve_goes.py` | `src/swmi/api/goes.py` | P0-G2 |
| `scripts/retrieve_goes_legacy.py` | `src/swmi/api/goes.py` | P0-G2 |
| `scripts/retrieve_goes_modern.py` | `src/swmi/api/goes.py` | P0-G2 |
| `scripts/baseline_model.py` | `scripts/04_train_baseline.py` | P0-E2, P0-S7 |

***

## Data Source URLs (Confirmed)

| Source | Base URL | Pattern | Coverage |
|---|---|---|---|
| OMNI 1-min | CDAWeb via `cdasws` | `OMNI_HRO_1MIN` | 1995–present |
| GOES mag legacy | `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/magnetometer/` | `g{sat}_magnetometer_1m_YYYYMM_v01_01.nc` | 2006–2020 |
| GOES mag modern | `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{sat}/l2/data/magn-l2-avg1m/` | `sci_magn-l2-avg1m_g{sat}_d{YYYYMMDD}_v1-0-1.nc` | 2016–present |
| GOES X-ray legacy | `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/` | `sci_xrsf-l2-avg1m_g{sat}_y{year}_v{version}.nc` | 2001–2020 |
| GOES X-ray modern | `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{sat}/l2/data/xrsf-l2-avg1m_science/` | `sci_xrsf-l2-avg1m_g{sat}_d{YYYYMMDD}_v{version}.nc` | 2016–present |
| Swarm MAGx_LR_1B | VirES via `viresclient` | `SW_MAGx_LR_1B` | 2013–present |
| SuperMAG inventory | `https://supermag.jhuapl.edu/mag/lib/content/api/supermag-api.py` | `SuperMAGGetInventory` | Per-month dynamic |
| SuperMAG data | `https://supermag.jhuapl.edu/mag/lib/content/api/supermag-api.py` | `SuperMAGGetData` | Per-station, per-month |
| SuperMAG indices | `https://supermag.jhuapl.edu/mag/lib/content/api/supermag-api.py` | `SuperMAGGetIndices` | Global, per-month |

***

## SuperMAG API Integration Summary

| Function | Use In Pipeline | Returns |
|---|---|---|
| `SuperMAGGetInventory` | P0-S1: Dynamic station discovery per month | List of available IAGA codes |
| `SuperMAGGetData(flags="mlt,mag,geo")` | P0-S3: Per-station data retrieval | NEZ components + MLT + mag coords + geo coords |
| `SuperMAGGetIndices(flags="sme,sml,smu,smr")` | Optional validation / fallback | Global SME, SML, SMU, SMR indices |
| `sm_grabme(data, 'N', 'nez')` | Data extraction | North component in NEZ |
| `sm_grabme(data, 'E', 'nez')` | Data extraction | East component in NEZ |
| `sm_grabme(data, 'Z', 'nez')` | Data extraction | Vertical component in NEZ |

### Target Computation Formula

```python
# From NEZ components (default SuperMAG output)
dbdt_n = gap_aware_diff(N_nez, timestamps) / actual_dt
dbdt_e = gap_aware_diff(E_nez, timestamps) / actual_dt

# Primary GIC proxy target
dbdt_horizontal_magnitude = sqrt(dbdt_n**2 + dbdt_e**2)
```

***

This task list is complete, ordered, and ready for Prompt 1B execution.
