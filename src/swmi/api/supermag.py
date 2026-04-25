#!/usr/bin/env python3
"""
retrieve_supermag.py
Downloads SuperMAG ground magnetometer data from the native JSON API
for all stations listed in config.SUPERMAG_STATIONS.

Produces one long-format Parquet per month containing all stations.

Output path convention:
    data/processed/supermag/YYYY/MM/supermag_YYYYMM.parquet

Columns: timestamp, station, b_n, b_e, b_z, dbn_dt, dbe_dt,
        dbdt_magnitude, dbdt_missing_flag

TODO: Refactor for all-station inventory retrieval; add SuperMAGGetInventory; integrate into scripts/01_download_data.py
"""

import os
import sys
import json
import math
import urllib.request
import urllib.error
import datetime
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)

# ---------------------------------------------------------------------------
SUPERMAG_API_BASE = "https://supermag.jhuapl.edu/services/data-api.php"
SUPERMAG_USERNAME = os.environ.get("SUPERMAG_USERNAME", "rsrogers")

_EXPECTED_MINUTES_PER_MONTH = 31 * 24 * 60  # upper bound; used for completeness check
_COMPLETENESS_WARN_THRESHOLD = 0.80          # warn if station has <80% data


def _compute_dbdt(b_n: pd.Series, b_e: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute dB/dt components using backward difference (config.DBDT_METHOD).

    Returns
    -------
    dbn_dt, dbe_dt, dbdt_magnitude, dbdt_missing_flag
    """
    if config.DBDT_METHOD != "backward":
        log.warning(
            "DBDT_METHOD=%r not implemented; falling back to 'backward'.", config.DBDT_METHOD
        )

    dbn_dt = b_n.diff() / 60.0           # nT/min
    dbe_dt = b_e.diff() / 60.0           # nT/min

    # dbdt_magnitude is NaN if EITHER component is missing — never zero-fill.
    # Zero is a physically real quiet-time value.
    dbdt_sq = np.where(
        dbn_dt.isna() | dbe_dt.isna(),
        float("nan"),
        dbn_dt.fillna(0) ** 2 + dbe_dt.fillna(0) ** 2,
    )
    dbdt_magnitude = pd.Series(np.sqrt(dbdt_sq), index=b_n.index)
    dbdt_magnitude[dbn_dt.isna() | dbe_dt.isna()] = float("nan")

    missing_flag = (dbn_dt.isna() | dbe_dt.isna()).astype("int8")

    return dbn_dt, dbe_dt, dbdt_magnitude, missing_flag


def _fetch_station(
    station: str,
    start_dt: datetime.datetime,
    extent_sec: int,
) -> Optional[pd.DataFrame]:
    """Fetch one station's data from SuperMAG API.

    Returns a DataFrame with columns [timestamp, b_n, b_e, b_z] or None
    if the request failed or returned no data.
    """
    url = (
        f"{SUPERMAG_API_BASE}?fmt=json&python&nohead&"
        f"start={start_dt.strftime('%Y-%m-%dT%H:%M')}&extent={extent_sec:012d}&"
        f"logon={SUPERMAG_USERNAME}&all&station={station.upper()}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw_data = response.read().decode("utf-8")
            rawlines = json.loads(raw_data)
    except urllib.error.URLError as exc:
        log.error("SuperMAG API connection failed for %s: %s", station, exc)
        return None
    except json.JSONDecodeError:
        log.error("SuperMAG returned non-JSON for %s. API may be down.", station)
        return None

    if not rawlines:
        log.warning("No data returned for SuperMAG station %s.", station)
        return None

    records = []
    for line in rawlines:
        try:
            records.append({
                "tval": line.get("tval"),
                "b_n": line.get("N", {}).get("nez", float("nan")) if isinstance(line.get("N"), dict) else float("nan"),
                "b_e": line.get("E", {}).get("nez", float("nan")) if isinstance(line.get("E"), dict) else float("nan"),
                "b_z": line.get("Z", {}).get("nez", float("nan")) if isinstance(line.get("Z"), dict) else float("nan"),
            })
        except AttributeError:
            continue

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["tval"], unit="s", utc=True)
    df = df.drop(columns=["tval"])

    # Mask SuperMAG physical fill values
    for col in ["b_n", "b_e", "b_z"]:
        df[col] = df[col].where(df[col] < 90000.0, float("nan"))

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _empty_station_df() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical SuperMAG schema."""
    cols = {
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
        "station": pd.Series([], dtype=str),
        "b_n": pd.Series([], dtype=float),
        "b_e": pd.Series([], dtype=float),
        "b_z": pd.Series([], dtype=float),
        "dbn_dt": pd.Series([], dtype=float),
        "dbe_dt": pd.Series([], dtype=float),
        "dbdt_magnitude": pd.Series([], dtype=float),
        "dbdt_missing_flag": pd.Series([], dtype="int8"),
    }
    return pd.DataFrame(cols)


def retrieve_supermag_month(year: int, month: int, stations: Optional[list] = None) -> None:
    """Retrieve SuperMAG data for all stations in one calendar month.

    Produces a long-format Parquet with one row per (station, minute).

    Parameters
    ----------
    year, month:
        Calendar year and month.
    stations:
        IAGA station codes to retrieve. Defaults to ``config.SUPERMAG_STATIONS``.
    """
    if not SUPERMAG_USERNAME:
        log.error("SUPERMAG_USERNAME env variable not set. Export it before running.")
        sys.exit(1)

    if stations is None:
        stations = config.SUPERMAG_STATIONS

    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.PROCESSED_DIR, "supermag", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"supermag_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("SuperMAG %s already exists, skipping.", month_str)
        return

    start_dt = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end_dt = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end_dt = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)

    minutes_in_month = int((end_dt - start_dt).total_seconds() / 60)
    extent_sec = int((end_dt - start_dt).total_seconds())

    frames = []
    for station in stations:
        station = station.upper()
        log.info("Retrieving SuperMAG station %s for %s...", station, month_str)

        raw_df = _fetch_station(station, start_dt, extent_sec)
        if raw_df is None or raw_df.empty:
            log.warning(
                "Station %s returned empty data for %s. Writing empty rows.", station, month_str
            )
            # Still append an empty frame so the schema is consistent
            frames.append(_empty_station_df())
            continue

        # Completeness check
        completeness = len(raw_df) / max(minutes_in_month, 1)
        if completeness < _COMPLETENESS_WARN_THRESHOLD:
            log.warning(
                "Station %s completeness for %s: %.1f%% (%d/%d rows).",
                station, month_str, completeness * 100, len(raw_df), minutes_in_month,
            )

        # Compute dB/dt
        dbn_dt, dbe_dt, dbdt_magnitude, missing_flag = _compute_dbdt(raw_df["b_n"], raw_df["b_e"])

        station_df = pd.DataFrame({
            "timestamp":        raw_df["timestamp"].values,
            "station":          station,
            "b_n":              raw_df["b_n"].values,
            "b_e":              raw_df["b_e"].values,
            "b_z":              raw_df["b_z"].values,
            "dbn_dt":           dbn_dt.values,
            "dbe_dt":           dbe_dt.values,
            "dbdt_magnitude":   dbdt_magnitude.values,
            "dbdt_missing_flag": missing_flag.values,
        })

        frames.append(station_df)
        log.info("  %s %s: %d rows, completeness=%.1f%%",
                 station, month_str, len(station_df), completeness * 100)

    if not frames or all(len(f) == 0 for f in frames):
        log.warning("All stations empty for SuperMAG %s. Writing empty Parquet.", month_str)
        _empty_station_df().to_parquet(output_path, index=False)
        return

    df_all = pd.concat(frames, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
    df_all = df_all.sort_values(["timestamp", "station"]).reset_index(drop=True)

    validate_output_schema(df_all, f"SuperMAG-{month_str}", allow_duplicates=True)
    df_all.to_parquet(output_path, index=False)
    log.info(
        "Saved SuperMAG %s → %s (%d rows, %d stations)",
        month_str, output_path, len(df_all), len(stations),
    )


if __name__ == "__main__":
    retrieve_supermag_month(year=2015, month=3, stations=config.SUPERMAG_STATIONS)
