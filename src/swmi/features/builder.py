"""Build SWMI feature matrices on a shared 1-minute UTC grid.

This module intentionally implements only the current P0 feature surface. It
preserves source identity, quality, missingness, and station/context columns so
the larger feature catalog in ``docs/research/potential_features.md`` can be
added later without changing upstream datasets.
"""

from __future__ import annotations

import os
from typing import Iterable

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from swmi.features.newell_coupling import compute_newell_numpy
from swmi.preprocessing.validation import validate_output_schema
from swmi.utils import config
from swmi.utils.logger import get_logger, install_dask_worker_file_logging

log = get_logger(__name__)

FEATURE_SCHEMA_VERSION = "feature_schema_v1"
GOES_MAG_ROLLING_WINDOWS_MIN = (60, 120)
GOES_MAG_ROLLING_STATS = ("mean", "std", "min", "max")
XRS_C_THRESHOLD = 1e-6
XRS_M_THRESHOLD = 1e-5
XRS_X_THRESHOLD = 1e-4
SUPERMAG_TARGET_COLUMN = "dbdt_horizontal_magnitude"
SUPERMAG_TARGET_ALIASES = (SUPERMAG_TARGET_COLUMN, "dbdt_magnitude")
SUPERMAG_CONTEXT_COLUMNS = ("mlt", "mlat", "mlon", "glat", "glon")


# ---------------------------------------------------------------------------
# Dask cluster
# ---------------------------------------------------------------------------

def _get_cluster():
    n_workers = config.DASK_N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(config.DASK_TEMP_DIR, exist_ok=True)
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit=config.DASK_WORKER_MEMORY_LIMIT,
        local_directory=config.DASK_TEMP_DIR,
    )
    client = Client(cluster)
    log.info("Dask cluster started: %s", client.dashboard_link)
    return cluster, client


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode UT hour and day-of-year as sin/cos pairs (in-place on copy)."""
    df = df.copy()
    ts = df["timestamp"]

    ut_hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.day_of_year.astype(float)

    df["ut_sin"]  = np.sin(ut_hour * 2 * np.pi / 24.0)
    df["ut_cos"]  = np.cos(ut_hour * 2 * np.pi / 24.0)
    df["doy_sin"] = np.sin((doy - 1.0) * 2 * np.pi / 365.0)
    df["doy_cos"] = np.cos((doy - 1.0) * 2 * np.pi / 365.0)

    return df


def _add_newell_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Newell coupling parameter and append as 'newell_phi'."""
    df = df.copy()
    vsw_abs = df["omni_vx"].abs()

    df["newell_phi"] = compute_newell_numpy(
        vsw    = vsw_abs.values,
        by_gsm = df["omni_by_gsm"].values,
        bz_gsm = df["omni_bz_gsm"].values,
    )
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std plus valid-point counts for key L1 features.

    Rolling stats are emitted only when the number of valid observations in the
    window meets the global minimum support threshold from config.
    """
    df = df.copy()

    feature_cols = ["omni_bz_gsm", "omni_vx", "newell_phi"]

    for col in feature_cols:
        if col not in df.columns:
            continue

        for w in [10, 30]:
            min_valid = max(1, int(np.ceil(w * config.ROLLING_MIN_VALID_FRAC)))

            valid_count = df[col].rolling(window=w, min_periods=1).count()
            rolling_mean = df[col].rolling(window=w, min_periods=1).mean()
            rolling_std = df[col].rolling(window=w, min_periods=1).std()

            mean_col = f"{col}_mean_{w}m"
            std_col = f"{col}_std_{w}m"
            count_col = f"{col}_valid_points_{w}m"

            df[count_col] = valid_count.astype("float32")
            df[mean_col] = rolling_mean.where(valid_count >= min_valid)
            df[std_col] = rolling_std.where(valid_count >= min_valid)

    return df


def _require_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("feature input is missing required 'timestamp' column")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("feature input contains null or unparseable timestamps")
    return out.sort_values("timestamp").reset_index(drop=True)


def _rolling_stat(series: pd.Series, window_min: int, stat: str, min_valid: int) -> pd.Series:
    rolling = series.rolling(window=window_min, min_periods=1)
    valid_count = rolling.count()
    if stat == "mean":
        values = rolling.mean()
    elif stat == "std":
        values = rolling.std()
    elif stat == "min":
        values = rolling.min()
    elif stat == "max":
        values = rolling.max()
    else:
        raise ValueError(f"Unsupported rolling statistic: {stat}")
    return values.where(valid_count >= min_valid)


def add_goes_features(
    df: pd.DataFrame,
    *,
    windows_min: Iterable[int] = GOES_MAG_ROLLING_WINDOWS_MIN,
    statistics: Iterable[str] = GOES_MAG_ROLLING_STATS,
    min_valid_fraction: float | None = None,
) -> pd.DataFrame:
    """Add scoped GEO GOES magnetometer state features.

    This is intentionally narrow for P0-D2: it uses canonical ``goes_bz_gsm``,
    emits past-only rolling state summaries, and preserves source/quality fields
    such as ``goes_source_satellite`` for future multi-satellite and MLT features.
    """
    out = _require_timestamp(df)
    if "goes_bz_gsm" not in out.columns:
        raise KeyError("add_goes_features requires canonical column 'goes_bz_gsm'")

    min_frac = config.ROLLING_MIN_VALID_FRAC if min_valid_fraction is None else min_valid_fraction
    if not 0 < min_frac <= 1:
        raise ValueError(f"min_valid_fraction must be in (0, 1], got {min_frac}")

    if "goes_mag_missing_flag" not in out.columns:
        out["goes_mag_missing_flag"] = out["goes_bz_gsm"].isna().astype("int8")
    else:
        out["goes_mag_missing_flag"] = out["goes_mag_missing_flag"].fillna(
            out["goes_bz_gsm"].isna().astype("int8")
        ).astype("int8")

    out["goes_bz_gsm_missing"] = out["goes_bz_gsm"].isna().astype("int8")

    for window in windows_min:
        min_valid = max(1, int(np.ceil(window * min_frac)))
        valid_count = out["goes_bz_gsm"].rolling(window=window, min_periods=1).count()
        out[f"goes_bz_gsm_valid_points_{window}m"] = valid_count.astype("float32")
        for stat in statistics:
            out[f"goes_bz_gsm_{stat}_{window}m"] = _rolling_stat(
                out["goes_bz_gsm"],
                int(window),
                stat,
                min_valid,
            )

    return out


def _select_xray_long_log(out: pd.DataFrame) -> pd.Series:
    if "goes_xray_long_normalized" in out.columns:
        return pd.to_numeric(out["goes_xray_long_normalized"], errors="coerce")
    if "goes_xray_long_log" in out.columns:
        return pd.to_numeric(out["goes_xray_long_log"], errors="coerce")
    if "xrsb_flux" in out.columns:
        flux = pd.to_numeric(out["xrsb_flux"], errors="coerce")
        return np.log10(flux.where(flux > 0))
    raise KeyError(
        "add_xray_features requires one of 'goes_xray_long_normalized', "
        "'goes_xray_long_log', or 'xrsb_flux'"
    )


def _time_since_last_event_minutes(timestamps: pd.Series, event_mask: pd.Series) -> pd.Series:
    last_event_ts = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns, UTC]")
    last_event_ts.loc[event_mask.fillna(False)] = timestamps.loc[event_mask.fillna(False)]
    last_event_ts = last_event_ts.ffill()
    return (timestamps - last_event_ts).dt.total_seconds() / 60.0


def add_xray_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add event-driven GOES X-ray precursor features for P0-G4.

    X-ray is treated as a solar precursor. This function deliberately avoids
    generic rolling windows and preserves raw/normalized XRS columns plus
    ``xray_quality_flags`` and ``xray_source_satellite`` for future flare,
    active-region, and cross-dataset interaction features.
    """
    out = _require_timestamp(df)
    long_log = _select_xray_long_log(out)
    out["goes_xray_long_log"] = long_log

    dt_min = out["timestamp"].diff().dt.total_seconds() / 60.0
    out["goes_xray_long_dlog_dt"] = long_log.diff() / dt_min
    out.loc[dt_min <= 0, "goes_xray_long_dlog_dt"] = np.nan

    if "xray_missing_flag" not in out.columns:
        source_flux_missing = long_log.isna()
        out["xray_missing_flag"] = source_flux_missing.astype("int8")
    else:
        out["xray_missing_flag"] = out["xray_missing_flag"].fillna(long_log.isna()).astype("int8")

    c_event = long_log >= np.log10(XRS_C_THRESHOLD)
    m_event = long_log >= np.log10(XRS_M_THRESHOLD)
    x_event = long_log >= np.log10(XRS_X_THRESHOLD)

    out["goes_xray_time_since_last_c_flare"] = _time_since_last_event_minutes(out["timestamp"], c_event)
    out["goes_xray_time_since_last_m_flare"] = _time_since_last_event_minutes(out["timestamp"], m_event)
    out["goes_xray_time_since_last_x_flare"] = _time_since_last_event_minutes(out["timestamp"], x_event)

    # Event-driven 24h accumulators, not arbitrary rolling statistics.
    out["goes_xray_cumulative_m_class_24h"] = (
        m_event.astype("int8").rolling(window=24 * 60, min_periods=1).sum()
    )
    out["goes_xray_max_flux_24h"] = (
        pd.to_numeric(out.get("xrsb_flux", 10**long_log), errors="coerce")
        .rolling(window=24 * 60, min_periods=1)
        .max()
    )

    return out

def _add_gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for missing L1/GEO values."""
    df = df.copy()
    l1_cols = ["omni_bz_gsm", "omni_by_gsm", "omni_vx", "omni_proton_density", "omni_pressure"]
    geo_cols = ["goes_bz_gsm", "goes_xray_long_log"]
    for c in l1_cols + geo_cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype("int8")

    if all(c in df.columns for c in l1_cols):
        df["l1_any_missing"] = df[l1_cols].isna().any(axis=1).astype("int8")
    if all(c in df.columns for c in geo_cols):
        df["geo_any_missing"] = df[geo_cols].isna().any(axis=1).astype("int8")
    return df


def _short_gap_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Apply forward-fill for short L1 smooth-variable gaps."""
    df = df.copy()
    smooth_cols = ["omni_vx", "omni_proton_density", "omni_pressure"]
    for c in smooth_cols:
        if c in df.columns:
            orig_missing = df[c].isna()
            df[c] = df[c].ffill(limit=config.MAX_INTERP_GAP_MIN)
            df[f"{c}_ffill_applied"] = (orig_missing & df[c].notna()).astype("int8")
    return df


def _finalize_station_target_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure station target flags are complete after joining to the master grid."""
    df = df.copy()
    for col in [c for c in df.columns if c.startswith("dbdt_missing_flag_")]:
        df[col] = df[col].fillna(1).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Per-partition transformation (for dask map_partitions)
# ---------------------------------------------------------------------------

def _transform_partition(partition: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transforms to a single Dask partition."""
    partition = _add_cyclical_features(partition)
    partition = _add_newell_phi(partition)
    partition = _add_rolling_features(partition)
    if "goes_bz_gsm" in partition.columns:
        partition = add_goes_features(partition)
    if (
        "goes_xray_long_normalized" in partition.columns
        or "goes_xray_long_log" in partition.columns
        or "xrsb_flux" in partition.columns
    ):
        partition = add_xray_features(partition)
    partition = _add_gap_flags(partition)
    partition = _short_gap_fill(partition)
    partition = _finalize_station_target_flags(partition)

    # Partition year/month columns for Parquet partitioning
    partition["year"]  = partition["timestamp"].dt.year.astype(int)
    partition["month"] = partition["timestamp"].dt.month.astype(int)
    partition["feature_schema_version"] = FEATURE_SCHEMA_VERSION

    return partition


# ---------------------------------------------------------------------------
# Column renaming helpers (align to canonical names)
# ---------------------------------------------------------------------------

def _rename_omni(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "BX_GSE":         "omni_bx_gse",
        "BY_GSM":         "omni_by_gsm",
        "BZ_GSM":         "omni_bz_gsm",
        "F":              "omni_f",
        "Vx":             "omni_vx",
        "proton_density": "omni_proton_density",
        "Pressure":       "omni_pressure",
        "SYM_H":          "omni_sym_h",
        "AL_INDEX":       "omni_al",
        "AU_INDEX":       "omni_au",
    })


def _rename_goes(df: pd.DataFrame) -> pd.DataFrame:
    # Already canonical from retrieve_goes.py; no rename needed.
    return df


def _rename_leo(df: pd.DataFrame) -> pd.DataFrame:
    return df  # sub-index columns already canonical from compute_leo_index.py


def _station_suffix(station: object) -> str:
    """Return a stable IAGA station suffix for wide target columns."""
    return str(station).strip().upper()


def _prepare_supermag_targets(smag_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format SuperMAG targets into canonical multi-output columns.

    Only the forecast target, its missingness flag, and SuperMAG-provided station
    context are propagated. Raw NEZ components and instantaneous derivative
    components stay out of the feature matrix to avoid target leakage.
    """
    target_source_col = next((col for col in SUPERMAG_TARGET_ALIASES if col in smag_df.columns), None)
    required = {"timestamp", "station"}
    missing = required - set(smag_df.columns)
    if missing or target_source_col is None:
        missing_list = sorted(missing)
        if target_source_col is None:
            missing_list.append(f"one of {list(SUPERMAG_TARGET_ALIASES)}")
        raise KeyError(f"SuperMAG target input missing required columns: {missing_list}")

    smag = smag_df.copy()
    smag["timestamp"] = pd.to_datetime(smag["timestamp"], utc=True, errors="coerce")
    if smag["timestamp"].isna().any():
        raise ValueError("SuperMAG target input contains null or unparseable timestamps.")

    smag["station"] = smag["station"].map(_station_suffix)
    smag = smag.sort_values(["timestamp", "station"]).drop_duplicates(
        subset=["timestamp", "station"],
        keep="first",
    )

    target = pd.to_numeric(smag[target_source_col], errors="coerce")
    if "dbdt_missing_flag" in smag.columns:
        missing_flag = pd.to_numeric(smag["dbdt_missing_flag"], errors="coerce").fillna(0).astype("int8")
    elif "dbdt_gap_flag" in smag.columns:
        missing_flag = pd.to_numeric(smag["dbdt_gap_flag"], errors="coerce").fillna(0).astype("int8")
    else:
        missing_flag = pd.Series(0, index=smag.index, dtype="int8")

    smag["dbdt_target_missing_flag"] = (target.isna() | (missing_flag.astype(bool))).astype("int8")
    smag[SUPERMAG_TARGET_COLUMN] = target.where(smag["dbdt_target_missing_flag"] == 0)

    pivot_values = [SUPERMAG_TARGET_COLUMN, "dbdt_target_missing_flag"]
    pivot_values.extend(col for col in SUPERMAG_CONTEXT_COLUMNS if col in smag.columns)

    wide = smag.pivot(index="timestamp", columns="station", values=pivot_values)
    wide.columns = [
        (
            f"dbdt_missing_flag_{station}"
            if value == "dbdt_target_missing_flag"
            else f"{value}_{station}"
        )
        for value, station in wide.columns
    ]
    wide = wide.reset_index()

    target_cols = [col for col in wide.columns if col.startswith(f"{SUPERMAG_TARGET_COLUMN}_")]
    if not target_cols:
        raise ValueError("SuperMAG target pivot produced no station target columns.")

    for col in [c for c in wide.columns if c.startswith("dbdt_missing_flag_")]:
        wide[col] = wide[col].fillna(1).astype("int8")

    return wide.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(year: int, month: int) -> None:
    """Build the feature matrix for one calendar month using Dask.

    Joins OMNI, GOES, LEO index, and SuperMAG onto a 1-minute UTC grid.
    Writes partitioned Parquet to data/processed/features/YYYY/MM/.

    Parameters
    ----------
    year, month:
        Calendar year and month to process.
    """
    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.FEATURES_DIR, f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"features_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("Feature matrix %s already exists, skipping.", month_str)
        return

    # ------------------------------------------------------------------
    # Locate input files
    # ------------------------------------------------------------------
    omni_path = os.path.join(
        config.PROCESSED_DIR, "omni", f"{year:04d}", f"{month:02d}", f"omni_{month_str}.parquet"
    )
    goes_mag_path = os.path.join(config.RAW_DATA_DIR, "goes", f"goes_mag_{month_str}.parquet")
    goes_xray_norm_path = os.path.join(
        config.PROCESSED_DIR,
        "goes",
        f"{year:04d}",
        f"{month:02d}",
        f"goes_xray_normalized_{month_str}.parquet",
    )
    goes_xray_raw_path = os.path.join(config.RAW_DATA_DIR, "goes", f"goes_xray_{month_str}.parquet")
    leo_path = os.path.join(
        config.PROCESSED_DIR, "swarm", f"{year:04d}", f"{month:02d}",
        f"swarm_leo_index_{month_str}.parquet",
    )
    smag_path = os.path.join(
        config.PROCESSED_DIR, "supermag", f"{year:04d}", f"{month:02d}",
        f"supermag_{month_str}.parquet",
    )

    missing = [p for p in [omni_path, leo_path, smag_path] if not os.path.exists(p)]
    if missing:
        log.error("Missing input files for feature matrix %s: %s", month_str, missing)
        return

    # ------------------------------------------------------------------
    # Start Dask cluster
    # ------------------------------------------------------------------
    cluster, client = _get_cluster()
    install_dask_worker_file_logging(client)

    try:
        # --------------------------------------------------------------
        # Load sources as Dask DataFrames (lazy)
        # --------------------------------------------------------------
        log.info("Loading source tables lazily for %s...", month_str)

        omni_dd = dd.read_parquet(omni_path)
        leo_dd  = dd.read_parquet(leo_path)
        smag_dd = dd.read_parquet(smag_path)

        # Build a master 1-minute grid for this month
        start_ts = pd.Timestamp(year, month, 1, tz="UTC")
        if month == 12:
            end_ts = pd.Timestamp(year + 1, 1, 1, tz="UTC")
        else:
            end_ts = pd.Timestamp(year, month + 1, 1, tz="UTC")
        master_ts = pd.date_range(start=start_ts, end=end_ts, freq="1min", inclusive="left", tz="UTC")
        grid_pd = pd.DataFrame({"timestamp": master_ts})
        grid_dd = dd.from_pandas(grid_pd, npartitions=4)

        # Rename OMNI columns to canonical names (computed step)
        omni_renamed_pd = _rename_omni(omni_dd.compute())
        omni_dd = dd.from_pandas(omni_renamed_pd, npartitions=4)

        # --------------------------------------------------------------
        # Joins (all on 'timestamp'; exact match after resampling)
        # --------------------------------------------------------------
        log.info("Joining OMNI onto master grid for %s...", month_str)
        joined_dd = grid_dd.merge(omni_dd, on="timestamp", how="left")

        if os.path.exists(goes_mag_path):
            log.info("Joining canonical GOES magnetometer for %s...", month_str)
            goes_mag_pd = _rename_goes(pd.read_parquet(goes_mag_path))
            goes_mag_pd["timestamp"] = pd.to_datetime(goes_mag_pd["timestamp"], utc=True)
            goes_mag_dd = dd.from_pandas(goes_mag_pd.sort_values("timestamp"), npartitions=4)
            joined_dd = joined_dd.merge(goes_mag_dd, on="timestamp", how="left")
        else:
            log.warning("No canonical GOES magnetometer file found for %s: %s", month_str, goes_mag_path)

        xray_path = goes_xray_norm_path if os.path.exists(goes_xray_norm_path) else goes_xray_raw_path
        if os.path.exists(xray_path):
            log.info("Joining GOES X-ray for %s: %s", month_str, xray_path)
            xray_pd = pd.read_parquet(xray_path)
            xray_pd["timestamp"] = pd.to_datetime(xray_pd["timestamp"], utc=True)
            xray_dd = dd.from_pandas(xray_pd.sort_values("timestamp"), npartitions=4)
            joined_dd = joined_dd.merge(xray_dd, on="timestamp", how="left")
        else:
            log.warning("No GOES X-ray file found for %s. X-ray columns will be absent.", month_str)

        log.info("Joining LEO index for %s...", month_str)
        joined_dd = joined_dd.merge(leo_dd, on="timestamp", how="left")

        # SuperMAG targets are long-format (station x minute). Pivot into
        # canonical multi-output columns while preserving station context.
        log.info("Pivoting SuperMAG and joining for %s...", month_str)
        smag_pd = smag_dd.compute()
        if not smag_pd.empty and "station" in smag_pd.columns:
            smag_pivot = _prepare_supermag_targets(smag_pd)
            target_cols = [
                col for col in smag_pivot.columns
                if col.startswith(f"{SUPERMAG_TARGET_COLUMN}_")
            ]
            log.info(
                "Prepared %d station dB/dt target columns for %s.",
                len(target_cols),
                month_str,
            )
            smag_dd_pivoted = dd.from_pandas(smag_pivot, npartitions=4)
            joined_dd = joined_dd.merge(smag_dd_pivoted, on="timestamp", how="left")
        else:
            raise ValueError(f"SuperMAG target file for {month_str} is empty or missing 'station'.")

        # --------------------------------------------------------------
        # Feature engineering via map_partitions (keeps computation lazy)
        # --------------------------------------------------------------
        log.info("Applying feature transforms for %s...", month_str)
        meta_partition = _transform_partition(
            joined_dd.get_partition(0).compute().head(0)
        )
        features_dd = joined_dd.map_partitions(_transform_partition, meta=meta_partition)

        # --------------------------------------------------------------
        # Compute and write
        # --------------------------------------------------------------
        log.info("Computing and writing feature matrix for %s...", month_str)
        features_pd = features_dd.compute()
        features_pd = features_pd.sort_values("timestamp").reset_index(drop=True)

        validate_output_schema(features_pd, f"features-{month_str}")
        features_pd.to_parquet(output_path, index=False)

        log.info(
            "Feature matrix %s → %s | rows=%d | cols=%d | ts=[%s, %s]",
            month_str, output_path,
            len(features_pd), len(features_pd.columns),
            features_pd["timestamp"].min(), features_pd["timestamp"].max(),
        )

    finally:
        client.close()
        cluster.close()
        log.info("Dask cluster closed.")


if __name__ == "__main__":
    build_feature_matrix(year=2015, month=3)
