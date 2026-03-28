"""
config.py
Single source of truth for all pipeline constants.

Scientific invariants that must NOT be altered without updating the data manifest:
- REFERENCE_FIELD: must be consistent across the ENTIRE training period.
- DECAY_HALFLIFE_MIN: controls LEO persistence physics; treat as hyperparameter.
- QDLAT_HIGH_LAT_MIN: determines auroral zone membership; changing it re-defines
  the spatial index meaning and requires re-processing all LEO parquet files.
"""

import os

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
RAW_DATA_DIR      = "data/raw"
PROCESSED_DIR     = "data/processed"
FEATURES_DIR      = "data/processed/features"
SEQUENCES_DIR     = "data/sequences"
MODELS_DIR        = "models"
LOGS_DIR          = "logs"
ARTIFACTS_DIR     = "models/artifacts"   # scalers, encoders
DASK_TEMP_DIR     = "data/tmp"           # Dask spill directory

# ---------------------------------------------------------------------------
# Training period boundaries
# Buffer prevents storm-recovery autocorrelation from crossing split seams.
# ---------------------------------------------------------------------------
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2023-12-31"

# Days excluded on each side of a split boundary to prevent storm-recovery
# phase of one period from contaminating the adjacent split.
SPLIT_BUFFER_DAYS = 7

# ---------------------------------------------------------------------------
# Forecast horizon and master cadence
# ---------------------------------------------------------------------------
FORECAST_HORIZON_MIN = 60   # minutes; target is dB/dt at T + this value
MASTER_CADENCE = "1min"     # all sources resampled/aligned to this

# ---------------------------------------------------------------------------
# GOES satellites
# Satellites <=15 are routed to the NCEI legacy NetCDF parser (GOES-13/15).
# Satellites >=16 are routed to the NGDC modern 1-minute product (GOES-R series).
# ---------------------------------------------------------------------------
GOES_LEGACY_SATS  = [13, 15]
GOES_MODERN_SATS  = [16, 17, 18]
GOES_LEGACY_CUTOFF_YEAR = 2016   # year >= this uses the modern parser

# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------
SWARM_SATELLITES  = ["A", "B", "C"]
SWARM_COLLECTION_TEMPLATE = "SW_OPER_MAG{sat}_LR_1B"
SWARM_VARIABLES = [
    "B_NEC", "F", "QDLat", "QDLon", "MLT",
    "Latitude", "Longitude", "Radius", "Flags_B",
]

# ---------------------------------------------------------------------------
# LEO index physics thresholds
# Physical rationale: Swarm orbital period ~94 min; most 1-min windows are
# empty. The 10-min decay half-life means a stale value retains ~50% of its
# original magnitude after one orbital gap. Treat as a hyperparameter for
# sensitivity analysis.
# ---------------------------------------------------------------------------
QDLAT_HIGH_LAT_MIN  = 55.0   # degrees; primary auroral zone filter
QDLAT_MID_LAT_MIN   = 20.0   # degrees; lower bound for mid-lat sub-index
QDLAT_MID_LAT_MAX   = 55.0   # degrees; upper bound for mid-lat sub-index
MLT_DAYSIDE_START   = 6.0    # hours; dayside MLT window start
MLT_DAYSIDE_END     = 18.0   # hours; dayside MLT window end
# Nightside is the complement: MLT < 6 or MLT >= 18

DECAY_HALFLIFE_MIN  = 10.0   # minutes; exponential decay half-life for LEO persistence

# ---------------------------------------------------------------------------
# SuperMAG stations
# Selected for: QD latitude 60-72 deg, >85% data completeness,
# geographic diversity (Scandinavia, North America, East Asia).
# Expand this list to scale the geographic generalization study.
# ---------------------------------------------------------------------------
SUPERMAG_STATIONS = ["ABK", "TRO", "BJN"]

# ---------------------------------------------------------------------------
# dB/dt computation
# "backward" is operationally correct (requires only past data).
# "centered" reduces edge noise and may be used for offline training.
# Document which is used in published results.
# ---------------------------------------------------------------------------
DBDT_METHOD = "backward"

# ---------------------------------------------------------------------------
# OMNI fill-value sentinel
# Per CDF metadata, NASA OMNI encodes missing data with large sentinel values
# (e.g., 9999.99 for field components, 99999.9 for speed). Any value above
# this threshold is treated as missing and replaced with NaN.
# ---------------------------------------------------------------------------
OMNI_FILL_THRESHOLD = 9000.0

# ---------------------------------------------------------------------------
# Gap interpolation policy
# Short gaps in smooth L1/GEO variables may be linearly interpolated.
# Longer gaps remain NaN (exposed via missingness flags).
# NEVER interpolate: targets, LEO indices, dB/dt values.
# ---------------------------------------------------------------------------
MAX_INTERP_GAP_MIN = 5   # minutes

# ---------------------------------------------------------------------------
# Rolling Statistic Threshold
# Minimum fraction of valid points required to emit a rolling statistic
# Example: 0.50 means at least half the window must be present.
# ---------------------------------------------------------------------------
ROLLING_MIN_VALID_FRAC = 0.50

# ---------------------------------------------------------------------------
# Reference field model
# CHAOS is preferred for scientific accuracy but requires the coefficient
# file from https://www.spacecenter.dk/files/magnetic-models/CHAOS-8/
# IGRF via ppigrf is the fallback; choose once and use consistently.
# ---------------------------------------------------------------------------
REFERENCE_FIELD = "IGRF"   # options: "IGRF", "CHAOS"

# ---------------------------------------------------------------------------
# Dask resource limits
# 10 years of Swarm at 1 Hz across 3 satellites is ~240M rows.
# Without explicit limits Dask fills /tmp on the OS partition.
# Route spillover to DASK_TEMP_DIR which should be on a data drive.
# ---------------------------------------------------------------------------
DASK_N_WORKERS          = None      # None → os.cpu_count() - 1 at runtime
DASK_WORKER_MEMORY_LIMIT = "4GB"   # per-worker; prevents OOM kills
# DASK_TEMP_DIR defined above with directory layout constants.

# ---------------------------------------------------------------------------
# Scaler versioning
# Increment when re-fitting on a different training period or feature set.
# ---------------------------------------------------------------------------
SCALER_VERSION = "1"
