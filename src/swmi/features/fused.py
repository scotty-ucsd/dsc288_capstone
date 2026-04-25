"""
fused.py
Build monthly analysis-ready fused Parquet files for EDA and feature engineering.

This module post-processes the feature matrix produced by ``builder.py`` to
create output datasets in three formats:

    minimal — Model input only (features + targets, no metadata)
    eda     — All columns + quality flags + station context (default)
    debug   — All columns including raw values and intermediate computations

Output path convention:
    {output_dir}/YYYY/MM/fused_YYYYMM.parquet

Scientific constraints:
    - No physics modifications: reuses validated feature matrix as-is.
    - Source completeness flags are observational, not imputed.
    - validate_output_schema() called before every write.
    - Outer join semantics preserved from feature builder: NaN for missing sources.
    - No train/test split: no anti-leakage concerns in fused mode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swmi.preprocessing.validation import validate_output_schema
from swmi.utils import config
from swmi.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Columns excluded from the "minimal" format (metadata, raw intermediates,
# quality flags, internal partitioning columns).
_METADATA_COLUMNS = frozenset({
    "year",
    "month",
    "feature_schema_version",
})

_QUALITY_FLAG_COLUMNS_PATTERNS = (
    "_missing",
    "_missing_flag",
    "_ffill_applied",
    "_valid_points_",
    "_decay_age",
    "_is_fresh",
    "_count",
    "source_completeness_",
    "any_source_missing",
    "xray_missing_flag",
    "xray_normalized_missing_flag",
    "goes_mag_missing_flag",
    "l1_any_missing",
    "geo_any_missing",
)

_STATION_CONTEXT_PATTERNS = (
    "mlt_",
    "mlat_",
    "mlon_",
    "glat_",
    "glon_",
    "qdlat_",
)

_SOURCE_SATELLITE_COLUMNS = frozenset({
    "goes_source_satellite",
    "xray_source_satellite",
})

# Columns that are always included in every format.
_ALWAYS_INCLUDE = frozenset({
    "timestamp",
})


def _is_quality_flag(col: str) -> bool:
    """Return True if col is a quality/diagnostic column."""
    return any(pattern in col for pattern in _QUALITY_FLAG_COLUMNS_PATTERNS)


def _is_station_context(col: str) -> bool:
    """Return True if col is a per-station context column (not a target)."""
    # Station context columns are mlt_STATION, mlat_STATION, etc.
    # but NOT dbdt_horizontal_magnitude_STATION (those are targets).
    return any(col.startswith(pattern) for pattern in _STATION_CONTEXT_PATTERNS)


# ---------------------------------------------------------------------------
# Column selection per format
# ---------------------------------------------------------------------------

def select_columns_for_format(
    df: pd.DataFrame,
    format_mode: str,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame columns based on the output format mode.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature matrix with all columns.
    format_mode : str
        One of ``"minimal"``, ``"eda"``, ``"debug"``.
    sources : list[str] or None
        If not None, only include columns from these sources.
        Valid values: ``"omni"``, ``"goes"``, ``"swarm"``, ``"supermag"``, ``"leo"``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with selected columns.

    Raises
    ------
    ValueError
        If ``format_mode`` is not one of the three valid values.
    """
    if format_mode not in ("minimal", "eda", "debug"):
        raise ValueError(
            f"format_mode must be 'minimal', 'eda', or 'debug', got {format_mode!r}"
        )

    all_cols = list(df.columns)

    if format_mode == "debug":
        # Debug: include everything.
        selected = all_cols
    elif format_mode == "eda":
        # EDA: everything except internal partitioning / schema version columns.
        selected = [c for c in all_cols if c not in _METADATA_COLUMNS]
    elif format_mode == "minimal":
        # Minimal: features + targets only. No metadata, quality flags,
        # station context, or source satellite identifiers.
        selected = [
            c for c in all_cols
            if c in _ALWAYS_INCLUDE
            or (
                c not in _METADATA_COLUMNS
                and c not in _SOURCE_SATELLITE_COLUMNS
                and not _is_quality_flag(c)
                and not _is_station_context(c)
            )
        ]
    else:
        selected = all_cols  # defensive fallback

    # Source filtering: restrict to columns from specified sources.
    if sources is not None and sources != ["all"]:
        selected = _filter_columns_by_source(selected, sources)

    return df[selected].copy()


def _filter_columns_by_source(
    columns: list[str],
    sources: list[str],
) -> list[str]:
    """Keep only columns belonging to the specified source list.

    Source prefixes:
        omni    → omni_*, newell_*, l1_*
        goes    → goes_*, xray_*, xrsb_*, xrsa_*, geo_*
        swarm   → leo_*
        supermag → dbdt_*, mlt_*, mlat_*, mlon_*, glat_*, glon_*, qdlat_*
        leo     → leo_* (alias for swarm)

    Columns without a clear source prefix (timestamp, cyclical, etc.)
    are always included.
    """
    source_set = set(s.lower().strip() for s in sources)

    # Merge "leo" and "swarm" — they reference the same data.
    if "leo" in source_set:
        source_set.add("swarm")

    _PREFIX_MAP: dict[str, tuple[str, ...]] = {
        "omni": ("omni_", "newell_", "l1_any_"),
        "goes": ("goes_", "xray_", "xrsb_", "xrsa_", "geo_any_"),
        "swarm": ("leo_",),
        "supermag": ("dbdt_", "mlt_", "mlat_", "mlon_", "glat_", "glon_", "qdlat_"),
    }

    # Build the set of allowed prefixes.
    allowed_prefixes: list[str] = []
    for source in source_set:
        if source in _PREFIX_MAP:
            allowed_prefixes.extend(_PREFIX_MAP[source])

    def _col_belongs(col: str) -> bool:
        """Return True if the column belongs to an allowed source or is source-agnostic."""
        # Source-agnostic columns are always included.
        if col in _ALWAYS_INCLUDE:
            return True
        # Cyclical time encodings are source-agnostic.
        if col in ("ut_sin", "ut_cos", "doy_sin", "doy_cos", "sin_hour",
                    "cos_hour", "sin_doy", "cos_doy", "sin_month", "cos_month"):
            return True
        # Metadata columns — include if present (debug mode).
        if col in _METADATA_COLUMNS:
            return True
        # Source completeness flags: include only for requested sources.
        if col.startswith("source_completeness_"):
            source_suffix = col.replace("source_completeness_", "")
            return source_suffix in source_set
        if col == "any_source_missing":
            return True
        # Check if the column matches any allowed prefix.
        for prefix in allowed_prefixes:
            if col.startswith(prefix):
                return True
        # Unknown columns: exclude (conservative).
        return False

    return [c for c in columns if _col_belongs(c)]


# ---------------------------------------------------------------------------
# Source completeness flags
# ---------------------------------------------------------------------------

def add_source_completeness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-source completeness fractions and a global missing flag.

    Completeness is computed as the fraction of non-NaN values across
    primary indicator columns for each source. These are rolling-window-free,
    per-row boolean indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with standard column names.

    Returns
    -------
    pd.DataFrame
        Copy with added columns:
        - ``source_completeness_omni`` (float, 0–1)
        - ``source_completeness_goes`` (float, 0–1)
        - ``source_completeness_swarm`` (float, 0–1)
        - ``source_completeness_supermag`` (float, 0–1)
        - ``any_source_missing`` (int8, 1 if any source is fully missing)
    """
    out = df.copy()

    # OMNI: based on primary driver columns.
    omni_indicators = [c for c in ("omni_bz_gsm", "omni_vx") if c in out.columns]
    if omni_indicators:
        out["source_completeness_omni"] = out[omni_indicators].notna().mean(axis=1).astype("float32")
    else:
        out["source_completeness_omni"] = np.float32(0.0)

    # GOES: magnetometer + X-ray.
    goes_indicators = [c for c in ("goes_bz_gsm", "goes_xray_long_log") if c in out.columns]
    if goes_indicators:
        out["source_completeness_goes"] = out[goes_indicators].notna().mean(axis=1).astype("float32")
    else:
        out["source_completeness_goes"] = np.float32(0.0)

    # Swarm/LEO: based on high-latitude sub-index.
    leo_indicators = [c for c in ("leo_high_lat",) if c in out.columns]
    if leo_indicators:
        out["source_completeness_swarm"] = out[leo_indicators].notna().mean(axis=1).astype("float32")
    else:
        out["source_completeness_swarm"] = np.float32(0.0)

    # SuperMAG: based on target columns.
    smag_target_cols = [c for c in out.columns if c.startswith("dbdt_horizontal_magnitude_")]
    if smag_target_cols:
        out["source_completeness_supermag"] = out[smag_target_cols].notna().mean(axis=1).astype("float32")
    else:
        out["source_completeness_supermag"] = np.float32(0.0)

    # Global flag: 1 if ANY source has zero completeness for that row.
    completeness_cols = [
        "source_completeness_omni",
        "source_completeness_goes",
        "source_completeness_swarm",
        "source_completeness_supermag",
    ]
    # A source is "missing" for a row if its completeness is 0 (all NaN).
    out["any_source_missing"] = (
        (out[completeness_cols] == 0).any(axis=1)
    ).astype("int8")

    return out


# ---------------------------------------------------------------------------
# Cyclical month encoding
# ---------------------------------------------------------------------------

def _add_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin_month and cos_month if not already present."""
    if "sin_month" in df.columns and "cos_month" in df.columns:
        return df
    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    month_val = out["timestamp"].dt.month.astype(float)
    out["sin_month"] = np.sin(month_val * 2 * np.pi / 12.0).astype("float32")
    out["cos_month"] = np.cos(month_val * 2 * np.pi / 12.0).astype("float32")
    return out


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_fused_summary_report(
    output_dir: str,
    months_processed: list[tuple[int, int, bool]],
) -> dict[str, Any]:
    """Generate a summary report for the fused dataset run.

    Parameters
    ----------
    output_dir : str
        Base directory containing fused Parquet files.
    months_processed : list of (year, month, success)
        Processing status for each month.

    Returns
    -------
    dict
        Summary report with station counts, completeness, and data volumes.
    """
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_months": len(months_processed),
        "succeeded": sum(1 for _, _, ok in months_processed if ok),
        "failed": sum(1 for _, _, ok in months_processed if not ok),
        "months": [],
    }

    for year, month, ok in months_processed:
        month_str = f"{year:04d}{month:02d}"
        fused_path = Path(output_dir) / f"{year:04d}" / f"{month:02d}" / f"fused_{month_str}.parquet"
        entry: dict[str, Any] = {
            "month": f"{year:04d}-{month:02d}",
            "success": ok,
            "path": str(fused_path),
            "exists": fused_path.exists(),
        }

        if fused_path.exists():
            try:
                df = pd.read_parquet(fused_path)
                entry["rows"] = len(df)
                entry["columns"] = len(df.columns)
                entry["size_bytes"] = int(fused_path.stat().st_size)

                # Station counts.
                target_cols = [c for c in df.columns if c.startswith("dbdt_horizontal_magnitude_")]
                entry["station_count"] = len(target_cols)
                entry["station_names"] = [
                    c.replace("dbdt_horizontal_magnitude_", "") for c in target_cols
                ]

                # Per-source completeness summary.
                completeness_cols = [c for c in df.columns if c.startswith("source_completeness_")]
                entry["mean_completeness"] = {
                    c: float(df[c].mean()) for c in completeness_cols
                } if completeness_cols else {}

            except Exception as exc:
                entry["read_error"] = str(exc)

        report["months"].append(entry)

    # Write report JSON.
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "fused_summary_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=False)
        log.info("Fused summary report saved to %s", report_path)
    except Exception as exc:
        log.warning("Could not write fused summary report: %s", exc)

    return report


# ---------------------------------------------------------------------------
# Main entry point: build one fused month
# ---------------------------------------------------------------------------

def _feature_matrix_path(year: int, month: int) -> Path:
    """Return the canonical path for a monthly feature matrix."""
    month_str = f"{year:04d}{month:02d}"
    return Path(config.FEATURES_DIR) / f"{year:04d}" / f"{month:02d}" / f"features_{month_str}.parquet"


def _fused_output_path(output_dir: str, year: int, month: int) -> Path:
    """Return the canonical output path for a fused monthly file."""
    month_str = f"{year:04d}{month:02d}"
    return Path(output_dir) / f"{year:04d}" / f"{month:02d}" / f"fused_{month_str}.parquet"


def build_fused_month(
    year: int,
    month: int,
    *,
    output_dir: str = "data/fused/",
    format_mode: str = "eda",
    sources: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Build a single fused monthly Parquet file for EDA.

    Steps:
        1. Check if fused output already exists (skip unless ``force``).
        2. Locate the feature matrix at the canonical path.
        3. Load, add completeness flags and month cyclical encoding.
        4. Apply column filtering per ``format_mode``.
        5. Validate schema and write to ``output_dir/YYYY/MM/fused_YYYYMM.parquet``.

    Parameters
    ----------
    year, month : int
        Calendar year and month to process.
    output_dir : str
        Base directory for fused output files.
    format_mode : str
        Output column format: ``"minimal"``, ``"eda"``, or ``"debug"``.
    sources : list[str] or None
        If not None, restrict to these sources (e.g., ``["omni", "goes"]``).
    force : bool
        If True, overwrite existing fused output.
    dry_run : bool
        If True, print the plan without writing.

    Returns
    -------
    Path or None
        The path to the written fused Parquet file, or None if dry-run
        or skipped.

    Raises
    ------
    FileNotFoundError
        If the feature matrix for this month does not exist.
    """
    month_str = f"{year:04d}{month:02d}"
    out_path = _fused_output_path(output_dir, year, month)
    feat_path = _feature_matrix_path(year, month)

    if dry_run:
        print(f"  [DRY-RUN] fused {month_str}:")
        print(f"    feature_matrix: {feat_path} (exists={feat_path.exists()})")
        print(f"    output:         {out_path}")
        print(f"    format:         {format_mode}")
        print(f"    sources:        {sources or 'all'}")
        print(f"    force:          {force}")
        return None

    # Idempotency check.
    if out_path.exists() and not force:
        log.info("Fused output %s already exists, skipping (use --force to overwrite).", out_path)
        return out_path

    # Locate the feature matrix.
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}. "
            f"Run the feature pipeline first: "
            f"uv run python scripts/02_build_features.py --year {year} --month {month}"
        )

    # Load.
    log.info("Loading feature matrix for %s from %s...", month_str, feat_path)
    df = pd.read_parquet(feat_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    log.info(
        "Loaded feature matrix: %d rows × %d columns.",
        len(df), len(df.columns),
    )

    # Add month cyclical encoding if missing.
    df = _add_month_cyclical(df)

    # Add source completeness flags.
    df = add_source_completeness_flags(df)

    # Apply format-specific column filtering.
    df = select_columns_for_format(df, format_mode, sources)

    # Validate schema before writing.
    validate_output_schema(df, f"fused-{month_str}")

    # Write.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    log.info(
        "Fused dataset %s → %s | rows=%d | cols=%d | format=%s | size=%.1f MB",
        month_str, out_path,
        len(df), len(df.columns),
        format_mode,
        out_path.stat().st_size / (1024 * 1024),
    )

    return out_path
