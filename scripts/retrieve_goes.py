#!/usr/bin/env python3
"""
retrieve_goes.py
Unified GOES magnetometer retrieval routing legacy (GOES-13/15) and
R-series (GOES-16/17/18) satellites to their respective NOAA archives.

Routing:
  satellite <= 15  →  NCEI legacy NetCDF parser (GOES-13, GOES-15)
  satellite >= 16  →  NGDC modern 1-minute NetCDF parser (GOES-R series)

Output path convention:
    data/processed/goes/YYYY/MM/goes_YYYYMM.parquet

Canonical output schema
-----------------------
timestamp          | UTC, 1-minute aligned
goes_bz_gsm        | GSM Bz (nT); primary compression indicator
goes_bt            | Total field magnitude (nT)
goes_bx_gsm        | GSM Bx (nT)
goes_by_gsm        | GSM By (nT)
goes_satellite     | Integer satellite number (e.g., 16)
goes_missing_flag  | Binary; 1 if this row was gap-filled by interpolation

IMPORTANT: GOES data is already in GSM coordinates. No coordinate
rotation is applied. Treat goes_bz_gsm as a direct compression indicator.
"""

import os
import re
import sys
import time
import datetime
import numpy as np
import pandas as pd
import requests
import xarray as xr
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
_NCEI_BASE_URL = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/mag/goes{sat}/magn-l2-hires"
)
_NGDC_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes{sat}/l2/data/magn-l2-avg1m/{year:04d}/{month:02d}/"
)

# J2000 epoch used in GOES NetCDF time encoding ("seconds since 2000-01-01 12:00:00")
_J2K_EPOCH = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_RETRIES = 3
_RETRY_SLEEP = 2


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _canonical_schema_empty(satellite: int) -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical GOES schema."""
    return pd.DataFrame({
        "timestamp":        pd.Series([], dtype="datetime64[ns, UTC]"),
        "goes_bz_gsm":      pd.Series([], dtype=float),
        "goes_bt":          pd.Series([], dtype=float),
        "goes_bx_gsm":      pd.Series([], dtype=float),
        "goes_by_gsm":      pd.Series([], dtype=float),
        "goes_satellite":   pd.Series([], dtype=int),
        "goes_missing_flag": pd.Series([], dtype="int8"),
    })


def _to_canonical(df_raw: pd.DataFrame, satellite: int) -> pd.DataFrame:
    """Rename raw [B_X_GSM, B_Y_GSM, B_Z_GSM] to canonical schema.

    Adds goes_bt, goes_satellite, and goes_missing_flag = 0.
    Applies short-gap linear interpolation per config.MAX_INTERP_GAP_MIN.
    """
    df = df_raw.rename(columns={
        "B_X_GSM": "goes_bx_gsm",
        "B_Y_GSM": "goes_by_gsm",
        "B_Z_GSM": "goes_bz_gsm",
    }).copy()

    df["goes_bt"] = np.sqrt(
        df["goes_bx_gsm"] ** 2 + df["goes_by_gsm"] ** 2 + df["goes_bz_gsm"] ** 2
    )
    df["goes_satellite"] = satellite
    df["goes_missing_flag"] = 0

    # Short-gap linear interpolation for smooth field components.
    # Only fills gaps <= MAX_INTERP_GAP_MIN; longer gaps remain NaN.
    limit = config.MAX_INTERP_GAP_MIN
    for col in ["goes_bx_gsm", "goes_by_gsm", "goes_bz_gsm", "goes_bt"]:
        was_missing = df[col].isna()
        df[col] = df[col].interpolate(method="linear", limit=limit, limit_direction="forward")
        newly_filled = was_missing & df[col].notna()
        df.loc[newly_filled, "goes_missing_flag"] = 1

    # Reorder
    cols = ["timestamp", "goes_bz_gsm", "goes_bt", "goes_bx_gsm", "goes_by_gsm",
            "goes_satellite", "goes_missing_flag"]
    return df[cols]


# ---------------------------------------------------------------------------
# Legacy parser (GOES <=15, NCEI)
# ---------------------------------------------------------------------------

def _parse_nc_file(local_path: str) -> Optional[pd.DataFrame]:
    """Parse a single GOES NetCDF file → DataFrame with raw B_X/Y/Z_GSM + timestamp."""
    try:
        ds = xr.open_dataset(local_path, engine="netcdf4", decode_times=False)
        times = ds["time"].values
        b_gsm = ds["b_gsm"].values
        ds.close()
    except Exception as exc:
        log.debug("Error opening %s: %s", local_path, exc)
        return None

    timestamps = _J2K_EPOCH + pd.to_timedelta(times, unit="s")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "B_X_GSM": b_gsm[:, 0],
        "B_Y_GSM": b_gsm[:, 1],
        "B_Z_GSM": b_gsm[:, 2],
    })
    for col in ["B_X_GSM", "B_Y_GSM", "B_Z_GSM"]:
        df[col] = df[col].where(df[col] > -9000.0, float("nan"))
    df = df.dropna(subset=["timestamp"])
    return df


def _download_file(url: str, local_path: str, max_retries: int = _MAX_RETRIES) -> bool:
    """Download a single file with retry. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        if os.path.exists(local_path):
            return True  # already downloaded
        log.debug("Downloading %s (attempt %d/%d)...", os.path.basename(url), attempt, max_retries)
        try:
            with requests.get(url, stream=True, timeout=60, headers=_HEADERS) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as exc:
            log.debug("Download attempt %d failed for %s: %s", attempt, url, exc)
            if os.path.exists(local_path):
                os.remove(local_path)
            time.sleep(_RETRY_SLEEP)
    return False


def _retrieve_legacy(year: int, month: int, satellite: int) -> Optional[pd.DataFrame]:
    """Retrieve and resample GOES-13/15 legacy data for one month."""
    sat_str = f"{satellite:02d}"
    month_url = _NCEI_BASE_URL.format(sat=sat_str) + f"/{year:04d}/{month:02d}/"
    raw_dir = os.path.join(config.RAW_DATA_DIR, "goes", "legacy", f"{year:04d}", f"{month:02d}")
    os.makedirs(raw_dir, exist_ok=True)

    log.info("Scanning NCEI for GOES-%d %04d-%02d: %s", satellite, year, month, month_url)
    try:
        resp = requests.get(month_url, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        log.error("Failed to access NCEI for GOES-%d %04d-%02d: %s", satellite, year, month, exc)
        return None

    nc_files = sorted(set(re.findall(
        r'href="(dn_magn-l2-hires_g\d{2}_d\d{8}_v[\d_]+\.nc)"', resp.text
    )))
    if not nc_files:
        log.warning("No NetCDF files found at %s", month_url)
        return None

    monthly_data = []
    for filename in nc_files:
        file_url = f"{month_url}{filename}"
        local_path = os.path.join(raw_dir, filename)

        for attempt in range(1, _MAX_RETRIES + 1):
            if not _download_file(file_url, local_path):
                log.warning("Could not download %s after %d attempts.", filename, _MAX_RETRIES)
                break

            day_df = _parse_nc_file(local_path)
            if day_df is not None:
                # Resample to 1-minute means (legacy cadence is 512 ms)
                day_df = (
                    day_df.set_index("timestamp")
                    .resample("1min")
                    .mean()
                    .reset_index()
                )
                monthly_data.append(day_df)
                break
            else:
                log.debug("Corrupted file %s; removing and retrying.", filename)
                if os.path.exists(local_path):
                    os.remove(local_path)
                time.sleep(_RETRY_SLEEP)

    if not monthly_data:
        return None
    return pd.concat(monthly_data, ignore_index=True)


# ---------------------------------------------------------------------------
# Modern parser (GOES >=16, NGDC)
# ---------------------------------------------------------------------------

def _retrieve_modern(year: int, month: int, satellite: int) -> Optional[pd.DataFrame]:
    """Retrieve GOES-R series 1-minute data for one month."""
    month_url = _NGDC_BASE_URL.format(sat=satellite, year=year, month=month)
    raw_dir = os.path.join(config.RAW_DATA_DIR, "goes", "modern", f"{year:04d}", f"{month:02d}")
    os.makedirs(raw_dir, exist_ok=True)

    log.info("Scanning NGDC for GOES-%d %04d-%02d: %s", satellite, year, month, month_url)
    try:
        resp = requests.get(month_url, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        log.error("Failed to access NGDC for GOES-%d %04d-%02d: %s", satellite, year, month, exc)
        return None

    pattern = rf'href="(dn_magn-l2-avg1m_g{satellite}_d\d{{8}}_v[\d_]+\.nc)"'
    nc_files = sorted(set(re.findall(pattern, resp.text)))
    if not nc_files:
        log.warning("No NetCDF files found for GOES-%d %04d-%02d", satellite, year, month)
        return None

    monthly_data = []
    for filename in nc_files:
        file_url = f"{month_url}{filename}"
        local_path = os.path.join(raw_dir, filename)

        for attempt in range(1, _MAX_RETRIES + 1):
            if not _download_file(file_url, local_path):
                log.warning("Could not download %s after %d attempts.", filename, _MAX_RETRIES)
                break

            day_df = _parse_nc_file(local_path)
            if day_df is not None:
                # Modern product is already 1-minute; resample to ensure alignment
                day_df = (
                    day_df.set_index("timestamp")
                    .resample("1min")
                    .mean()
                    .reset_index()
                )
                monthly_data.append(day_df)
                break
            else:
                log.debug("Corrupted file %s; removing and retrying.", filename)
                if os.path.exists(local_path):
                    os.remove(local_path)
                time.sleep(_RETRY_SLEEP)

    if not monthly_data:
        return None
    return pd.concat(monthly_data, ignore_index=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def retrieve_goes_month(year: int, month: int, satellite: int) -> None:
    """Retrieve one month of GOES magnetometer data.

    Routing is automatic: satellites <=15 use the NCEI legacy parser;
    satellites >=16 use the NGDC modern parser.

    Parameters
    ----------
    year, month:
        Calendar year and month.
    satellite:
        GOES satellite number (int), e.g. 13, 15, 16, 17, 18.

    Output
    ------
    Writes ``data/processed/goes/YYYY/MM/goes_YYYYMM.parquet``.
    Skips silently if already exists.
    On any fetch failure: writes an empty-schema Parquet and logs ERROR.
    """
    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.PROCESSED_DIR, "goes", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"goes{satellite}_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("GOES-%d %s already exists, skipping.", satellite, month_str)
        return

    log.info("Retrieving GOES-%d for %s...", satellite, month_str)

    if satellite <= 15:
        raw_df = _retrieve_legacy(year, month, satellite)
    elif satellite >= 16:
        raw_df = _retrieve_modern(year, month, satellite)
    else:
        log.error("Invalid GOES satellite number: %d", satellite)
        _canonical_schema_empty(satellite).to_parquet(output_path, index=False)
        return

    if raw_df is None or raw_df.empty:
        log.error("No data for GOES-%d %s. Writing empty Parquet.", satellite, month_str)
        _canonical_schema_empty(satellite).to_parquet(output_path, index=False)
        return

    raw_df = raw_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Ensure UTC timezone
    if raw_df["timestamp"].dt.tz is None:
        raw_df["timestamp"] = raw_df["timestamp"].dt.tz_localize("UTC")
    else:
        raw_df["timestamp"] = raw_df["timestamp"].dt.tz_convert("UTC")

    df = _to_canonical(raw_df, satellite)

    validate_output_schema(df, f"GOES-{satellite}-{month_str}")
    df.to_parquet(output_path, index=False)
    log.info("Saved %d rows for GOES-%d %s → %s", len(df), satellite, month_str, output_path)


def retrieve_goes(year: int, month: int, satellites: Optional[list] = None) -> None:
    """Retrieve all configured GOES satellites for a given month.

    Parameters
    ----------
    year, month:
        Calendar year and month.
    satellites:
        List of satellite integers. Defaults to
        ``config.GOES_LEGACY_SATS + config.GOES_MODERN_SATS``.
    """
    if satellites is None:
        satellites = config.GOES_LEGACY_SATS + config.GOES_MODERN_SATS
    for sat in satellites:
        try:
            retrieve_goes_month(year, month, sat)
        except Exception as exc:
            log.error("Unexpected error retrieving GOES-%d %04d-%02d: %s", sat, year, month, exc)


if __name__ == "__main__":
    retrieve_goes(year=2015, month=3)
