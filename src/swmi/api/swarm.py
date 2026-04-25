#!/usr/bin/env python3
"""
retrieve_swarm.py
Downloads Swarm MAGx_LR_1B data for satellites A, B, C via viresclient.
Requests B_NEC, QDLat, MLT, and ancillary orbit parameters.

Output path convention:
    data/raw/swarm/YYYY/MM/swarm{sat}_LR1B_YYYYMM.parquet

TODO: Refactor to class-based retriever; integrate into scripts/01_download_data.py
"""

import os
import datetime
import pandas as pd
from viresclient import SwarmRequest

from swmi.preprocessing.validation import validate_output_schema
from swmi.utils import config
from swmi.utils.logger import get_logger

log = get_logger(__name__)


def _collection_for(satellite: str) -> str:
    """Return the VirES collection string for a Swarm satellite letter."""
    sat = satellite.upper()
    return config.SWARM_COLLECTION_TEMPLATE.format(sat=sat)


def retrieve_swarm_month(year: int, month: int, satellite: str) -> None:
    """Retrieve one month of Swarm MAGx_LR_1B data and write to Parquet.

    Parameters
    ----------
    year, month:
        Calendar year/month to retrieve.
    satellite:
        One of ``config.SWARM_SATELLITES`` (``"A"``, ``"B"``, or ``"C"``).

    Output
    ------
    ``data/raw/swarm/YYYY/MM/swarm{sat}_LR1B_YYYYMM.parquet``
    Skips silently if already exists.
    """
    satellite = satellite.upper()
    if satellite not in config.SWARM_SATELLITES:
        log.error("Invalid Swarm satellite '%s'. Must be one of %s.", satellite, config.SWARM_SATELLITES)
        raise ValueError(f"Satellite must be one of {config.SWARM_SATELLITES}. Got: {satellite!r}")

    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.RAW_DATA_DIR, "swarm", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"swarm{satellite}_LR1B_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("Swarm %s %s already exists, skipping.", satellite, month_str)
        return

    start = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)

    log.info("Retrieving Swarm %s for %s...", satellite, month_str)

    try:
        request = SwarmRequest()
        request.set_collection(_collection_for(satellite))
        request.set_products(
            measurements=["B_NEC", "F", "Flags_B"],
            auxiliaries=["QDLat", "QDLon", "MLT"],
        )
        data = request.get_between(start, end, asynchronous=True)
        df = data.as_dataframe()
    except Exception as exc:
        log.error("Swarm %s %s retrieval failed: %s", satellite, month_str, exc)
        _write_empty(output_path, satellite)
        return

    if df is None or df.empty:
        log.warning("No data returned for Swarm %s %s. Writing empty marker.", satellite, month_str)
        _write_empty(output_path, satellite)
        return

    # Standardise timestamp column
    if "Timestamp" in df.index.names:
        df = df.reset_index()
    if "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.drop(columns=["Timestamp"])
    else:
        log.error("Swarm %s: expected 'Timestamp' column not found for %s.", satellite, month_str)
        return

    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols]
    df["satellite"] = satellite

    validate_output_schema(df, f"Swarm-{satellite}-{month_str}")
    df.to_parquet(output_path, index=False)
    log.info("Saved %d rows for Swarm %s %s → %s", len(df), satellite, month_str, output_path)


def _write_empty(output_path: str, satellite: str) -> None:
    """Write a zero-row placeholder Parquet to retain the path contract."""
    cols = ["timestamp", "B_NEC", "F", "Flags_B", "QDLat", "QDLon", "MLT",
            "Latitude", "Longitude", "Radius", "satellite"]
    empty = pd.DataFrame(columns=cols)
    empty["timestamp"] = pd.Series([], dtype="datetime64[ns, UTC]")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    empty.to_parquet(output_path, index=False)
    log.debug("Wrote empty Swarm placeholder to %s", output_path)


def retrieve_swarm_range(start_date: str, end_date: str, satellites: list | None = None) -> None:
    """Retrieve multiple months of Swarm data for a list of satellites.

    Parameters
    ----------
    start_date, end_date:
        ISO date strings (``"YYYY-MM-DD"``).
    satellites:
        Satellite letters to retrieve.  Defaults to ``config.SWARM_SATELLITES``.
    """
    if satellites is None:
        satellites = config.SWARM_SATELLITES

    start = datetime.datetime.fromisoformat(start_date).replace(tzinfo=datetime.timezone.utc)
    end   = datetime.datetime.fromisoformat(end_date).replace(tzinfo=datetime.timezone.utc)
    current = datetime.datetime(start.year, start.month, 1, tzinfo=datetime.timezone.utc)

    while current <= end:
        for sat in satellites:
            retrieve_swarm_month(current.year, current.month, sat)
        if current.month == 12:
            current = datetime.datetime(current.year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        else:
            current = datetime.datetime(current.year, current.month + 1, 1, tzinfo=datetime.timezone.utc)


if __name__ == "__main__":
    retrieve_swarm_range(
        start_date="2015-03-01",
        end_date="2015-03-31",
        satellites=config.SWARM_SATELLITES,
    )
