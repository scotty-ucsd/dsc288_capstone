"""
compute_leo_index.py

Fuses raw Swarm A/B/C 1-Hz observations into a structured set of 1-minute
spatial sub-indices capturing ionospheric perturbation across different
magnetic latitude bands and magnetic local time sectors.

Processing pipeline per month
------------------------------
1. Build lightweight file/task descriptors on the client.
2. Dispatch one task per day to Dask workers.
3. Each worker task:
   a. Reads only the needed Swarm monthly files locally.
   b. Filters to one UTC day and valid quality rows.
   c. Evaluates the reference field (CHAOS or IGRF).
   d. Subtracts reference and computes residual horizontal magnitude.
4. Gather daily residual outputs.
5. Aggregate 1-Hz residuals to 1-minute sub-indices (median, robust to spikes).
6. Apply exponential decay persistence per sub-index (no zero-fill).
7. Validate and write to Parquet.

Output path convention:
    data/processed/swarm/YYYY/MM/swarm_leo_index_YYYYMM.parquet

Scientific constraints
----------------------
- No zero-fill: zero residual is physically meaningful.
- Decay from last valid value only; decay_age tracks staleness.
- Median, not mean, is used for aggregation.
- Reference field version is logged and must be consistent across the run.

TODO: Refactor Dask parallelization; add worker error handling; integrate into scripts/02_build_features.py
"""

import os
import math
import functools
import importlib.metadata
from typing import List, Optional

import numpy as np
import pandas as pd

from swmi.preprocessing.validation import validate_output_schema
from swmi.utils import config
from swmi.utils.logger import get_logger, install_dask_worker_file_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SWARM_COLUMNS = [
    "timestamp",
    "B_NEC",
    "Latitude",
    "Longitude",
    "Radius",
    "QDLat",
    "MLT",
]

OPTIONAL_SWARM_COLUMNS = [
    "Flags_B",
    "satellite",
]

READ_COLUMNS = REQUIRED_SWARM_COLUMNS + OPTIONAL_SWARM_COLUMNS


# ---------------------------------------------------------------------------
# Worker-local model cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_chaos_model(model_path: str):
    import chaosmagpy as cp
    import logging

    _log = logging.getLogger(__name__)
    _log.info("[worker] Loading CHAOS model from %s ...", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CHAOS model not found at {model_path}. "
            "Download the .mat coefficients and point config.MODELS_DIR to them."
        )

    return cp.load_CHAOS_matfile(model_path)


# ---------------------------------------------------------------------------
# Reference field evaluation
# ---------------------------------------------------------------------------

def _empty_residual_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
            "QDLat": pd.Series([], dtype=float),
            "MLT": pd.Series([], dtype=float),
            "res_N": pd.Series([], dtype=float),
            "res_E": pd.Series([], dtype=float),
            "res_H": pd.Series([], dtype=float),
        }
    )


def _stack_b_nec(series: pd.Series) -> np.ndarray:
    vals = series.values
    if len(vals) == 0:
        return np.empty((0, 3), dtype=float)

    try:
        arr = np.stack(vals)
    except Exception as exc:
        raise ValueError(f"Could not stack B_NEC vectors: {exc}") from exc

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"B_NEC has unexpected shape: {arr.shape}")

    return np.asarray(arr, dtype=float)


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _prepare_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    if df_chunk is None or df_chunk.empty:
        return _empty_residual_df()

    df = _coerce_timestamp(df_chunk)

    needed = ["timestamp", "B_NEC", "Latitude", "Longitude", "Radius", "QDLat", "MLT"]
    df = df.dropna(subset=needed).copy()
    if df.empty:
        return _empty_residual_df()

    b_nec = _stack_b_nec(df["B_NEC"])
    if len(b_nec) != len(df):
        raise ValueError("B_NEC stack length mismatch with DataFrame rows.")

    df["_b_n"] = b_nec[:, 0]
    df["_b_e"] = b_nec[:, 1]
    return df


def _eval_reference_field_chaos(df_chunk: pd.DataFrame, model_path: str) -> pd.DataFrame:
    import chaosmagpy as cp

    df = _prepare_chunk(df_chunk)
    if df.empty:
        return df

    model = _get_chaos_model(model_path)

    radius_km = df["Radius"].to_numpy(dtype=float) / 1000.0
    theta_deg = 90.0 - df["Latitude"].to_numpy(dtype=float)
    phi_deg = df["Longitude"].to_numpy(dtype=float)

    ts = df["timestamp"].dt
    time_mjd = cp.data_utils.mjd2000(
        ts.year.to_numpy(),
        ts.month.to_numpy(),
        ts.day.to_numpy(),
        ts.hour.to_numpy(),
        ts.minute.to_numpy(),
        ts.second.to_numpy(),
    )

    _, b_theta, b_phi = model.synth_values_tdep(
        time_mjd, radius_km, theta_deg, phi_deg, nmax=15
    )

    b_n_model = -np.asarray(b_theta, dtype=float)
    b_e_model = np.asarray(b_phi, dtype=float)

    df["res_N"] = df["_b_n"].to_numpy(dtype=float) - b_n_model
    df["res_E"] = df["_b_e"].to_numpy(dtype=float) - b_e_model
    df["res_H"] = np.sqrt(df["res_N"] ** 2 + df["res_E"] ** 2)

    return df[["timestamp", "QDLat", "MLT", "res_N", "res_E", "res_H"]].copy()


def _eval_reference_field_igrf(df_chunk: pd.DataFrame) -> pd.DataFrame:
    import ppigrf

    df = _prepare_chunk(df_chunk)
    if df.empty:
        return df

    alt_km = (df["Radius"].to_numpy(dtype=float) / 1000.0) - 6371.2
    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)

    dates = df["timestamp"].dt.date
    unique_dates = pd.unique(dates)

    res_n_arr = np.full(len(df), np.nan, dtype=float)
    res_e_arr = np.full(len(df), np.nan, dtype=float)

    for d in unique_dates:
        mask = (dates == d).to_numpy()
        if not mask.any():
            continue

        try:
            be, bn, bu = ppigrf.igrf(
                lon[mask],
                lat[mask],
                alt_km[mask],
                pd.Timestamp(str(d)).to_pydatetime(),
            )
            res_n_arr[mask] = df["_b_n"].to_numpy(dtype=float)[mask] - np.asarray(bn[0], dtype=float)
            res_e_arr[mask] = df["_b_e"].to_numpy(dtype=float)[mask] - np.asarray(be[0], dtype=float)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "ppigrf failed for date %s: %s -- leaving NaN residuals", d, exc
            )

    df["res_N"] = res_n_arr
    df["res_E"] = res_e_arr
    df["res_H"] = np.sqrt(df["res_N"] ** 2 + df["res_E"] ** 2)

    return df[["timestamp", "QDLat", "MLT", "res_N", "res_E", "res_H"]].copy()


def _process_daily_chunk(df_day: pd.DataFrame, reference_field: str, chaos_path: str) -> pd.DataFrame:
    if df_day is None or df_day.empty:
        return _empty_residual_df()

    if reference_field == "CHAOS":
        return _eval_reference_field_chaos(df_day, chaos_path)

    return _eval_reference_field_igrf(df_day)


# ---------------------------------------------------------------------------
# Worker-side file loading
# ---------------------------------------------------------------------------

def _safe_read_parquet_columns(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        if columns is None:
            return pd.read_parquet(path)
        return pd.read_parquet(path, columns=columns)
    except TypeError:
        return pd.read_parquet(path)


def _load_and_filter_satellite_file(
    fpath: str,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
) -> pd.DataFrame:
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = _safe_read_parquet_columns(fpath, columns=READ_COLUMNS)
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        raise KeyError(f"Missing required column 'timestamp' in {fpath}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    mask = (df["timestamp"] >= day_start) & (df["timestamp"] < day_end)
    df = df.loc[mask].copy()
    if df.empty:
        return df

    if "Flags_B" in df.columns:
        df = df[df["Flags_B"] == 0].copy()

    missing_required = [c for c in REQUIRED_SWARM_COLUMNS if c not in df.columns]
    if missing_required:
        raise KeyError(f"{fpath} missing required columns: {missing_required}")

    keep_cols = [c for c in READ_COLUMNS if c in df.columns]
    return df[keep_cols].copy()


def _load_and_process_daily_chunk(
    swarm_paths: List[str],
    day_start: pd.Timestamp,
    reference_field: str,
    chaos_path: str,
) -> pd.DataFrame:
    day_end = day_start + pd.Timedelta(days=1)
    dfs = []

    for fpath in swarm_paths:
        try:
            df = _load_and_filter_satellite_file(fpath, day_start, day_end)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed reading/filtering %s for %s: %s", fpath, day_start.date(), exc
            )

    if not dfs:
        return _empty_residual_df()

    df_day = pd.concat(dfs, ignore_index=True)
    df_day = df_day.sort_values("timestamp").reset_index(drop=True)

    return _process_daily_chunk(df_day, reference_field, chaos_path)


# ---------------------------------------------------------------------------
# Persistence decay
# ---------------------------------------------------------------------------

def _apply_decay(
    raw_df: pd.DataFrame,
    master_ts: pd.DatetimeIndex,
    col: str,
    halflife_min: float,
):
    decay_lambda = math.log(2) / halflife_min
    master_df = pd.DataFrame({"timestamp": master_ts})
    merged = master_df.merge(raw_df[["timestamp", col]], on="timestamp", how="left")

    vals = merged[col].to_numpy(dtype=float).copy()
    decay_age = np.full(len(merged), np.nan, dtype=float)
    is_fresh = np.zeros(len(merged), dtype=np.int8)

    last_val = np.nan
    last_idx = -1

    for i in range(len(merged)):
        if np.isfinite(vals[i]):
            last_val = vals[i]
            last_idx = i
            is_fresh[i] = 1
            decay_age[i] = 0.0
        elif np.isfinite(last_val):
            dt = i - last_idx
            vals[i] = last_val * math.exp(-decay_lambda * dt)
            decay_age[i] = float(dt)
            is_fresh[i] = 0

    return vals, decay_age, is_fresh


# ---------------------------------------------------------------------------
# Sub-index aggregation
# ---------------------------------------------------------------------------

def _build_subindices(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")})

    df = df_all.copy()
    df["ts_min"] = df["timestamp"].dt.floor("1min")

    abs_qdlat = df["QDLat"].abs()

    m_high_lat = abs_qdlat >= config.QDLAT_HIGH_LAT_MIN
    m_mid_lat = (abs_qdlat >= config.QDLAT_MID_LAT_MIN) & (abs_qdlat < config.QDLAT_MID_LAT_MAX)
    m_dayside = (
        m_high_lat
        & (df["MLT"] >= config.MLT_DAYSIDE_START)
        & (df["MLT"] < config.MLT_DAYSIDE_END)
    )
    m_nightside = (
        m_high_lat
        & ((df["MLT"] < config.MLT_DAYSIDE_START) | (df["MLT"] >= config.MLT_DAYSIDE_END))
    )

    def agg_group(mask: pd.Series, name: str) -> pd.DataFrame:
        grp = df.loc[mask].groupby("ts_min")["res_H"]
        med = grp.median().rename(name)
        cnt = grp.count().rename(f"{name}_count")
        return pd.concat([med, cnt], axis=1)

    hl = agg_group(m_high_lat, "leo_high_lat")
    ml = agg_group(m_mid_lat, "leo_mid_lat")
    day = agg_group(m_dayside, "leo_dayside")
    ngt = agg_group(m_nightside, "leo_nightside")

    idx = hl.join(ml, how="outer").join(day, how="outer").join(ngt, how="outer")
    idx = idx.reset_index().rename(columns={"ts_min": "timestamp"})

    for cnt_col in [
        "leo_high_lat_count",
        "leo_mid_lat_count",
        "leo_dayside_count",
        "leo_nightside_count",
    ]:
        if cnt_col in idx.columns:
            idx[cnt_col] = idx[cnt_col].fillna(0).astype(int)

    return idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_start_end(year: int, month: int):
    start = pd.Timestamp(year, month, 1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    else:
        end = pd.Timestamp(year, month + 1, 1, tz="UTC")
    return start, end


def _collect_swarm_paths(year: int, month: int) -> List[str]:
    month_str = f"{year:04d}{month:02d}"
    paths = []

    for sat in config.SWARM_SATELLITES:
        fpath = os.path.join(
            config.RAW_DATA_DIR,
            "swarm",
            f"{year:04d}",
            f"{month:02d}",
            f"swarm{sat}_LR1B_{month_str}.parquet",
        )
        if os.path.exists(fpath):
            paths.append(fpath)
        else:
            log.warning("Swarm %s file missing for %s: %s", sat, month_str, fpath)

    return paths


def _estimate_input_rows(swampaths: List[str]) -> int:
    total = 0
    for p in swampaths:
        try:
            total += len(pd.read_parquet(p, columns=["timestamp"]))
        except Exception:
            try:
                total += len(pd.read_parquet(p))
            except Exception:
                pass
    return int(total)


# ---------------------------------------------------------------------------
# Main monthly function
# ---------------------------------------------------------------------------

def build_leo_index_month(year: int, month: int) -> None:
    from dask.distributed import Client, LocalCluster

    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.PROCESSED_DIR, "swarm", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"swarm_leo_index_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("LEO index %s already exists, skipping.", month_str)
        return

    reference_field = config.REFERENCE_FIELD
    chaos_path = os.path.join(config.MODELS_DIR, "CHAOS-8.5")

    if reference_field == "CHAOS":
        try:
            rf_version = importlib.metadata.version("chaosmagpy")
        except Exception:
            rf_version = "unknown"
        log.info("Reference field: CHAOS (chaosmagpy v%s) at %s", rf_version, chaos_path)
    else:
        try:
            rf_version = importlib.metadata.version("ppigrf")
        except Exception:
            rf_version = "unknown"
        log.info("Reference field: IGRF (ppigrf v%s)", rf_version)

    swarm_paths = _collect_swarm_paths(year, month)
    if not swarm_paths:
        log.warning("No Swarm data available for %s. Skipping LEO index.", month_str)
        return

    n_input = _estimate_input_rows(swarm_paths)
    log.info(
        "Found %d Swarm source files for %s with about %d raw rows.",
        len(swarm_paths), month_str, n_input
    )

    start_date, end_date = _month_start_end(year, month)
    days = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")[:-1]

    n_workers = config.DASK_N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(config.DASK_TEMP_DIR, exist_ok=True)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=config.DASK_WORKER_MEMORY_LIMIT,
        local_directory=config.DASK_TEMP_DIR,
    )
    client = Client(cluster)
    log.info("Dask cluster started: %s", client.dashboard_link)

    install_dask_worker_file_logging(client)

    try:
        futures = []
        for day in days:
            fut = client.submit(
                _load_and_process_daily_chunk,
                swarm_paths,
                day,
                reference_field,
                chaos_path,
                pure=False,
            )
            futures.append(fut)

        log.info("Dispatching %d daily chunks to %d workers...", len(futures), n_workers)
        results = client.gather(futures, errors="return")

    finally:
        client.close()
        cluster.close()
        log.info("Dask cluster closed.")

    valid_results: list[pd.DataFrame] = []
    for day, res in zip(days, results):
        if isinstance(res, BaseException):
            log.error("LEO index daily chunk raised for %s: %s", day, res, exc_info=res)
            raise RuntimeError(
                f"LEO index Dask worker failure for {month_str} (day={day!r})."
            ) from res
        if res is None or res.empty or "res_H" not in res.columns:
            log.error(
                "LEO index daily chunk returned invalid data for %s (empty=%s, res_H=%s).",
                day,
                res is None or getattr(res, "empty", True),
                "res_H" in res.columns if res is not None and not res.empty else False,
            )
            continue
        valid_results.append(res)

    if not valid_results:
        raise RuntimeError(
            f"No valid LEO index daily chunks for {month_str} after Dask run (all days empty/invalid)."
        )

    df_res = pd.concat(valid_results, ignore_index=True)
    df_res = df_res.sort_values("timestamp").reset_index(drop=True)

    n_output = len(df_res)
    if n_input > 0 and n_output < n_input * 0.5:
        log.warning(
            "Significant row loss during reference field evaluation for %s: input=%d, output=%d.",
            month_str, n_input, n_output,
        )

    log.info("Reference field evaluated: %d residual rows retained.", n_output)

    log.info("Aggregating to 1-minute sub-indices for %s...", month_str)
    idx_df = _build_subindices(df_res)

    master_ts = pd.date_range(
        start=start_date,
        end=end_date,
        freq="1min",
        inclusive="left",
        tz="UTC",
    )

    sub_indices = ["leo_high_lat", "leo_mid_lat", "leo_dayside", "leo_nightside"]
    result_df = pd.DataFrame({"timestamp": master_ts})

    for si in sub_indices:
        if si not in idx_df.columns:
            log.warning("Sub-index '%s' missing from aggregation output for %s.", si, month_str)
            result_df[si] = np.nan
            result_df[f"{si}_decay_age"] = np.nan
            result_df[f"{si}_is_fresh"] = np.int8(0)
            result_df[f"{si}_count"] = 0
            continue

        vals, decay_age, is_fresh = _apply_decay(
            idx_df, master_ts, si, config.DECAY_HALFLIFE_MIN
        )
        result_df[si] = vals
        result_df[f"{si}_decay_age"] = decay_age
        result_df[f"{si}_is_fresh"] = is_fresh

        cnt_col = f"{si}_count"
        if cnt_col in idx_df.columns:
            cnt_merged = result_df[["timestamp"]].merge(
                idx_df[["timestamp", cnt_col]],
                on="timestamp",
                how="left",
            )
            result_df[cnt_col] = cnt_merged[cnt_col].fillna(0).astype(int).to_numpy()
        else:
            result_df[cnt_col] = 0

    validate_output_schema(result_df, f"LEO-index-{month_str}")
    result_df.to_parquet(output_path, index=False)

    fresh_cols = [f"{si}_is_fresh" for si in sub_indices]
    fresh_fraction = float(np.mean(np.column_stack([result_df[c].to_numpy() for c in fresh_cols])))

    log.info(
        "Saved LEO index for %s -> %s (%d rows, %d sub-indices, mean fresh fraction=%.4f)",
        month_str,
        output_path,
        len(result_df),
        len(sub_indices),
        fresh_fraction,
    )


if __name__ == "__main__":
    build_leo_index_month(year=2015, month=3)
