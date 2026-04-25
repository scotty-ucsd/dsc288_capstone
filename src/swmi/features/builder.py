"""
build_feature_matrix.py
Assembles the full feature matrix by lazily joining OMNI, GOES, Swarm LEO
index, and SuperMAG onto a shared 1-minute UTC grid using Dask DataFrames.

Computes:
  - Newell coupling parameter (phi) via src/newell_coupling.py
  - Rolling 10/30-minute mean and std for key L1 features
  - Cyclical encodings for UT hour and day-of-year
  - Gap missingness flags

Output path convention:
    data/processed/features/YYYY/MM/features_YYYYMM.parquet

Scientific invariants
---------------------
- No future data in features. All rolling windows use past-only history.
- Cyclical features (ut_sin, ut_cos, doy_sin, doy_cos) are excluded from
  Z-score scaling (they are already in [-1, 1]).
- GOES data is already GSM; no coordinate rotation applied.
- OMNI data is already bow-shock-propagated; no time-shifting applied.

TODO: Add GOES X-ray precursor features; add multi-station target columns; integrate into scripts/02_build_features.py
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema
from newell_coupling import compute_newell_numpy

log = get_logger(__name__)


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

def _add_gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for missing L1/GEO values."""
    df = df.copy()
    l1_cols = ["omni_bz_gsm", "omni_by_gsm", "omni_vx", "omni_proton_density", "omni_pressure"]
    geo_cols = ["goes_bz_gsm"]
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


# ---------------------------------------------------------------------------
# Per-partition transformation (for dask map_partitions)
# ---------------------------------------------------------------------------

def _transform_partition(partition: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transforms to a single Dask partition."""
    partition = _add_cyclical_features(partition)
    partition = _add_newell_phi(partition)
    partition = _add_rolling_features(partition)
    partition = _add_gap_flags(partition)
    partition = _short_gap_fill(partition)

    # Partition year/month columns for Parquet partitioning
    partition["year"]  = partition["timestamp"].dt.year.astype(int)
    partition["month"] = partition["timestamp"].dt.month.astype(int)

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
    # GOES: take the first available satellite for this month
    goes_dir = os.path.join(config.PROCESSED_DIR, "goes", f"{year:04d}", f"{month:02d}")
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

    try:
        # --------------------------------------------------------------
        # Load sources as Dask DataFrames (lazy)
        # --------------------------------------------------------------
        log.info("Loading source tables lazily for %s...", month_str)

        omni_dd = dd.read_parquet(omni_path)
        leo_dd  = dd.read_parquet(leo_path)
        smag_dd = dd.read_parquet(smag_path)

        # GOES: merge across all available satellite files for month
        goes_files = [
            os.path.join(goes_dir, f)
            for f in os.listdir(goes_dir)
            if f.startswith("goes") and f.endswith(".parquet")
        ] if os.path.isdir(goes_dir) else []

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

        if goes_files:
            log.info("Joining GOES (%d files) for %s...", len(goes_files), month_str)
            goes_pds = []
            for gf in sorted(goes_files):
                try:
                    goes_pds.append(pd.read_parquet(gf))
                except Exception as exc:
                    log.warning("Could not read GOES file %s: %s", gf, exc)
            if goes_pds:
                goes_pd = pd.concat(goes_pds, ignore_index=True)
                goes_pd = goes_pd.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                goes_dd_inner = dd.from_pandas(goes_pd, npartitions=4)
                joined_dd = joined_dd.merge(goes_dd_inner, on="timestamp", how="left")
        else:
            log.warning("No GOES files found for %s. GOES columns will be absent.", month_str)

        log.info("Joining LEO index for %s...", month_str)
        joined_dd = joined_dd.merge(leo_dd, on="timestamp", how="left")

        # SuperMAG is long-format (station × minute). Pivot before joining
        # so each station's dbdt_magnitude becomes a separate column.
        log.info("Pivoting SuperMAG and joining for %s...", month_str)
        smag_pd = smag_dd.compute()
        if not smag_pd.empty and "station" in smag_pd.columns:
            smag_pivot = smag_pd.pivot_table(
                index="timestamp",
                columns="station",
                values=["b_n", "b_e", "b_z", "dbn_dt", "dbe_dt", "dbdt_magnitude", "dbdt_missing_flag"],
                aggfunc="first",
            )
            smag_pivot.columns = [f"{stat}_{col.lower()}" for col, stat in smag_pivot.columns]
            smag_pivot = smag_pivot.reset_index()
            smag_dd_pivoted = dd.from_pandas(smag_pivot, npartitions=4)
            joined_dd = joined_dd.merge(smag_dd_pivoted, on="timestamp", how="left")

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
