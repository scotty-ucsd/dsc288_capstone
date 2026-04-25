#!/usr/bin/env python3
"""
retrieve_omni.py
Downloads OMNI 1-minute data via CDAWeb REST API using cdasws.
Uses a chunking mechanism to bypass payload limits, cleans fill values,
and saves to a monthly Parquet file.

Output path convention:
    data/processed/omni/YYYY/MM/omni_YYYYMM.parquet

TODO: Refactor to class-based retriever inheriting from BaseRetriever; integrate into scripts/01_download_data.py
"""

import os
import sys
import datetime
import pandas as pd
from cdasws import CdasWs

# --- project imports --------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)

# ---------------------------------------------------------------------------
OMNI_DATASET_ID = "OMNI_HRO2_1MIN"

OMNI_VARIABLES = [
    "BX_GSE",          # IMF Bx in GSE
    "BY_GSM",          # IMF By in GSM
    "BZ_GSM",          # IMF Bz in GSM
    "F",               # IMF magnitude (Bt)
    "Vx",              # Solar wind speed
    "proton_density",  # Np
    "Pressure",        # Dynamic pressure
    "SYM_H",           # SYM-H index
    "AL_INDEX",        # AL index
    "AU_INDEX",        # AU index
]

# Per-variable fill thresholds from NASA CDF metadata.
# Values above threshold are OMNI missing-data sentinels and are mapped to NaN.
_FILL_THRESHOLDS = {
    "BX_GSE":         config.OMNI_FILL_THRESHOLD,
    "BY_GSM":         config.OMNI_FILL_THRESHOLD,
    "BZ_GSM":         config.OMNI_FILL_THRESHOLD,
    "F":              config.OMNI_FILL_THRESHOLD,
    "Vx":             config.OMNI_FILL_THRESHOLD,
    "proton_density": config.OMNI_FILL_THRESHOLD,
    "Pressure":       config.OMNI_FILL_THRESHOLD,
    "SYM_H":          config.OMNI_FILL_THRESHOLD,
    "AL_INDEX":       config.OMNI_FILL_THRESHOLD,
    "AU_INDEX":       config.OMNI_FILL_THRESHOLD,
}


def retrieve_omni(year: int, month: int, chunk_days: int = 5) -> None:
    """Download one month of OMNI 1-minute data and write to Parquet.

    Parameters
    ----------
    year, month:
        Calendar year and month to retrieve.
    chunk_days:
        Number of days per CDAWeb request; limits payload size to stay
        within the 5-day API result window.

    Output
    ------
    Writes to ``data/processed/omni/YYYY/MM/omni_YYYYMM.parquet``.
    Skips silently if the output already exists (idempotent).
    """
    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.PROCESSED_DIR, "omni", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"omni_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("OMNI %s already exists, skipping: %s", month_str, output_path)
        return

    start_dt = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    # One month forward; handles December correctly
    if month == 12:
        final_end_dt = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        final_end_dt = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)

    log.info("Retrieving OMNI %s in %d-day chunks...", month_str, chunk_days)
    cdas = CdasWs()
    current_start = start_dt
    df_list = []

    while current_start < final_end_dt:
        current_end = min(current_start + datetime.timedelta(days=chunk_days), final_end_dt)
        log.debug("  Fetching %s to %s...", current_start.date(), current_end.date())
        try:
            status, data = cdas.get_data(
                OMNI_DATASET_ID,
                OMNI_VARIABLES,
                current_start,
                current_end,
            )
        except Exception as exc:
            log.error("OMNI chunk %s failed with exception: %s", current_start.date(), exc)
            current_start = current_end
            continue

        if status["http"]["status_code"] == 200 and data is not None:
            chunk_df = data.to_dataframe().reset_index()
            df_list.append(chunk_df)
        else:
            log.warning(
                "OMNI chunk %s returned HTTP %s.",
                current_start.date(),
                status["http"]["status_code"],
            )
        current_start = current_end

    if not df_list:
        log.warning("No OMNI data returned for %s. Writing empty Parquet.", month_str)
        empty = pd.DataFrame(columns=["timestamp"] + OMNI_VARIABLES)
        empty["timestamp"] = pd.Series([], dtype="datetime64[ns, UTC]")
        empty.to_parquet(output_path, index=False)
        return

    # Combine chunks
    df = pd.concat(df_list, ignore_index=True)

    if "Epoch" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Epoch"], utc=True)
        df = df.drop(columns=["Epoch"])
    else:
        log.error("OMNI: Expected 'Epoch' column not found in returned data for %s.", month_str)
        return

    # Remove overlap from chunk boundaries
    df = df.drop_duplicates(subset=["timestamp"])
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols].sort_values("timestamp").reset_index(drop=True)

    # Replace fill sentinels with NaN
    for col, threshold in _FILL_THRESHOLDS.items():
        if col in df.columns:
            df[col] = df[col].where(df[col] < threshold, float("nan"))

    validate_output_schema(df, f"OMNI-{month_str}")
    df.to_parquet(output_path, index=False)
    log.info("Saved %d rows of OMNI %s → %s", len(df), month_str, output_path)


if __name__ == "__main__":
    retrieve_omni(year=2015, month=3)
