"""
config.py
Single source of truth for all pipeline constants.

Scientific invariants that must NOT be altered without updating the data manifest:
- REFERENCE_FIELD: must be consistent across the ENTIRE training period.
- DECAY_HALFLIFE_MIN: controls LEO persistence physics; treat as hyperparameter.
- QDLAT_HIGH_LAT_MIN: determines auroral zone membership; changing it re-defines
  the spatial index meaning and requires re-processing all LEO parquet files.

YAML config loading:
    Use ``load_config(path)`` to load any YAML config file.
    Use ``validate_scientific_invariants(cfg)`` to cross-check YAML against
    the Python constants defined here.
"""

import os
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
RAW_DATA_DIR      = "data/raw"
INTERIM_DIR       = "data/interim"
PROCESSED_DIR     = "data/processed"
FEATURES_DIR      = "data/processed/features"
SEQUENCES_DIR     = "data/sequences"
STATION_METADATA_DIR = "data/external/station_metadata"
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
GOES_MODERN_SATS  = [16, 17, 18, 19]
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

# ---------------------------------------------------------------------------
# Minimum completeness threshold for data quality checks
# ---------------------------------------------------------------------------
MIN_COMPLETENESS_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# GOES primary satellite (default; overridden by YAML per-year priority)
# ---------------------------------------------------------------------------
GOES_PRIMARY_SATELLITE = "GOES-16"


# ===================================================================
# YAML Configuration Loader
# ===================================================================

# Map of scientific invariants: (yaml_path, python_constant_value)
# Used by validate_scientific_invariants() to detect drift between
# YAML configs and hardcoded Python constants.
_SCIENTIFIC_INVARIANTS = {
    "REFERENCE_FIELD":       ("scientific_invariants.reference_field", REFERENCE_FIELD),
    "QDLAT_HIGH_LAT_MIN":   ("scientific_invariants.qdlat_high_lat_min", QDLAT_HIGH_LAT_MIN),
    "DECAY_HALFLIFE_MIN":   ("scientific_invariants.decay_halflife_min", DECAY_HALFLIFE_MIN),
    "DBDT_METHOD":          ("scientific_invariants.dbdt_method", DBDT_METHOD),
    "FORECAST_HORIZON_MIN": ("forecast_horizon_min", FORECAST_HORIZON_MIN),
    "SPLIT_BUFFER_DAYS":    ("split_buffer_days", SPLIT_BUFFER_DAYS),
}


def _resolve_dotpath(cfg: dict, dotpath: str) -> Any:
    """Resolve a dot-separated path in a nested dict.

    Parameters
    ----------
    cfg : dict
        Nested configuration dictionary.
    dotpath : str
        Dot-separated key path (e.g., ``"scientific_invariants.reference_field"``).

    Returns
    -------
    Any
        The value at the specified path.

    Raises
    ------
    KeyError
        If any key in the path is missing.
    """
    keys = dotpath.split(".")
    current = cfg
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(f"Expected dict at '{key}' in path '{dotpath}', got {type(current)}")
        if key not in current:
            raise KeyError(f"Key '{key}' not found in config at path '{dotpath}'")
        current = current[key]
    return current


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file and return a nested dict.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to a YAML file (e.g.,
        ``"configs/data_retrieval.yaml"``).

    Returns
    -------
    dict
        Parsed YAML content as a nested dictionary.

    Raises
    ------
    ImportError
        If ``pyyaml`` is not installed.
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If the YAML file is empty or contains invalid YAML.

    Examples
    --------
    >>> cfg = load_config("configs/data_retrieval.yaml")
    >>> cfg["master_cadence"]
    '1min'
    >>> cfg["goes"]["satellite_priority"][2017]
    ['GOES-16', 'GOES-15']
    """
    if yaml is None:
        raise ImportError(
            "pyyaml is required for YAML config loading. "
            "Install with: uv add pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}. "
            f"Check that the path is correct relative to the project root."
        )

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(
            f"Config file is empty or contains only comments: {path}. "
            "Populate the file with valid YAML content."
        )

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping (dict), "
            f"got {type(cfg).__name__}: {path}"
        )

    log.debug("Loaded config from %s (%d top-level keys)", path, len(cfg))
    return cfg


def validate_scientific_invariants(cfg: dict, config_name: str = "unknown") -> None:
    """Cross-validate YAML config values against hardcoded Python invariants.

    This function ensures that scientific invariants defined in YAML config
    files have not drifted from the authoritative values in this Python
    module.  Any mismatch indicates a configuration error that would
    invalidate prior outputs.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration dictionary.
    config_name : str
        Human-readable name for the config (used in error messages).

    Raises
    ------
    ValueError
        If any scientific invariant in the YAML differs from its
        approved Python constant value.

    Notes
    -----
    Only checks invariants whose YAML paths exist in ``cfg``.  Missing
    paths are logged as warnings but do not raise — this allows partial
    configs (e.g., ``model_baseline.yaml`` does not contain LEO physics
    thresholds).
    """
    violations = []

    for name, (dotpath, expected) in _SCIENTIFIC_INVARIANTS.items():
        try:
            actual = _resolve_dotpath(cfg, dotpath)
        except KeyError:
            # Path doesn't exist in this config file — that's OK for
            # partial configs. Only log at DEBUG.
            log.debug(
                "[%s] Invariant %s (path '%s') not present in config.",
                config_name, name, dotpath,
            )
            continue

        # Type-flexible comparison: YAML may load 55.0 as int 55
        if type(expected) is float:
            if abs(float(actual) - expected) > 1e-9:
                violations.append(
                    f"  {name}: YAML={actual!r} (path={dotpath}), "
                    f"Python={expected!r}"
                )
        elif str(actual) != str(expected):
            violations.append(
                f"  {name}: YAML={actual!r} (path={dotpath}), "
                f"Python={expected!r}"
            )

    if violations:
        msg = (
            f"[{config_name}] Scientific invariant mismatch detected!\n"
            f"The following YAML values differ from approved Python constants:\n"
            + "\n".join(violations)
            + "\n\nThis invalidates all prior outputs. Either:\n"
            "  1. Restore the approved values in the YAML, or\n"
            "  2. Update the Python constants in config.py and reprocess all data."
        )
        raise ValueError(msg)

    log.debug("[%s] All scientific invariants validated OK.", config_name)
