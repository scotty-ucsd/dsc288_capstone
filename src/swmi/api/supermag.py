#!/usr/bin/env python3
"""
supermag.py
SuperMAG ground magnetometer data retrieval module.

Provides:
    get_station_inventory  — Dynamic per-month station inventory via SuperMAG API
    retrieve_supermag_month — Downloads NEZ magnetometer data for all/specified stations

Output path conventions:
    Inventory: data/external/station_metadata/supermag_inventory_YYYYMM.json
    Data:      data/raw/supermag/YYYY/MM/supermag_YYYYMM.parquet

Data columns: timestamp, station, n_nez, e_nez, z_nez, mlt, mlat, mlon, glat, glon

Design decisions:
    - Station list is DYNAMIC — retrieved via SuperMAGGetInventory per month (Warning #19)
    - No hard-coded station lists for production use
    - SuperMAG-provided MLT/mag coords preferred over apexpy (Warning #20)
    - API rate limits handled with exponential backoff (Warning #21)
"""

import os
import sys
import json
import urllib.request
import urllib.error
import datetime
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence

# Resolve project-internal imports: config.py, logger.py live in utils/;
# validation.py lives in preprocessing/.
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_pkg_root, "utils"))
sys.path.insert(0, os.path.join(_pkg_root, "preprocessing"))

import config
from logger import get_logger
from validation import validate_output_schema

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# SuperMAG API endpoints
# ---------------------------------------------------------------------------
SUPERMAG_API_BASE = "https://supermag.jhuapl.edu/services/data-api.php"
SUPERMAG_INVENTORY_URL = "https://supermag.jhuapl.edu/services/inventory.php"
SUPERMAG_USERNAME = os.environ.get("SUPERMAG_USERNAME", "rsrogers")

_COMPLETENESS_WARN_THRESHOLD = 0.80          # warn if station has <80% data
_SUPERMAG_FILL_VALUE_THRESHOLD = 90000.0

# Exponential backoff for API rate limits (Warning #21)
_API_INITIAL_BACKOFF_SEC = 2.0
_API_MAX_BACKOFF_SEC = 120.0
_API_MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Station metadata output directory
# ---------------------------------------------------------------------------
_STATION_METADATA_DIR = os.path.join("data", "external", "station_metadata")
_STATION_METADATA_COLUMNS = [
    "station",
    "glat",
    "glon",
    "mlat",
    "mlon",
    "mlt_offset",
    "operational_start",
    "operational_end",
    "sample_months",
    "metadata_available",
]


# ===================================================================
# P0-S1: Dynamic Station Inventory
# ===================================================================

def get_station_inventory(
    year: int,
    month: int,
    force_refresh: bool = False,
) -> list[str]:
    """Retrieve the list of SuperMAG stations with data for a given month.

    Wraps the SuperMAG Inventory Service (``inventory.php``) which returns
    IAGA station codes for all stations that have data in the specified
    time window.  Results are cached as JSON files to avoid redundant
    API calls.

    Parameters
    ----------
    year, month : int
        Calendar year and month to query.
    force_refresh : bool, default False
        If True, ignore cached inventory and re-query the API.

    Returns
    -------
    list[str]
        Sorted list of uppercase IAGA station codes (e.g., ["ABK", "AND", ...]).
        Empty list if the API returns no stations or fails after retries.

    Raises
    ------
    RuntimeError
        If ``SUPERMAG_USERNAME`` is not set.

    Notes
    -----
    - The inventory is queried for the full calendar month (day 1 00:00
      to day 1 of the next month 00:00).
    - Station availability varies by month due to station outages,
      decommissioning, and new deployments.  This is why a dynamic
      inventory is essential (Warning #19).
    - The inventory JSON includes metadata: query timestamp, station
      count, and geographic coverage summary.
    - API failures use exponential backoff with up to
      ``_API_MAX_RETRIES`` retries (Warning #21).

    Complexity
    ----------
    Time:  $O(1)$ network call + $O(S \\log S)$ sort where S = station count
    Space: $O(S)$ for the station list

    Examples
    --------
    >>> stations = get_station_inventory(2015, 3)
    >>> len(stations) > 100  # typically 200-400 stations
    True
    >>> "ABK" in stations
    True
    """
    if not SUPERMAG_USERNAME:
        raise RuntimeError(
            "SUPERMAG_USERNAME environment variable not set. "
            "Export it before calling get_station_inventory()."
        )

    month_str = f"{year:04d}{month:02d}"
    os.makedirs(_STATION_METADATA_DIR, exist_ok=True)
    cache_path = os.path.join(
        _STATION_METADATA_DIR, f"supermag_inventory_{month_str}.json"
    )

    # ------------------------------------------------------------------
    # Check cache
    # ------------------------------------------------------------------
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            stations = cached.get("stations", [])
            log.info(
                "Loaded cached SuperMAG inventory for %s: %d stations.",
                month_str, len(stations),
            )
            return stations
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning(
                "Cached inventory for %s is corrupt (%s); re-querying API.",
                month_str, exc,
            )

    # ------------------------------------------------------------------
    # Compute time window: full calendar month
    # ------------------------------------------------------------------
    start_dt = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end_dt = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end_dt = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)
    extent_sec = int((end_dt - start_dt).total_seconds())

    # ------------------------------------------------------------------
    # Query the SuperMAG Inventory Service with retry logic
    # ------------------------------------------------------------------
    url = (
        f"{SUPERMAG_INVENTORY_URL}"
        f"?logon={SUPERMAG_USERNAME}"
        f"&start={start_dt.strftime('%Y-%m-%dT%H:%M')}"
        f"&extent={extent_sec:012d}"
    )
    log.info(
        "Querying SuperMAG inventory for %s (extent=%d sec)...",
        month_str, extent_sec,
    )

    import time as _time  # local import to avoid top-level conflict

    stations: list[str] = []
    backoff = _API_INITIAL_BACKOFF_SEC

    for attempt in range(1, _API_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                raw_data = response.read().decode("utf-8")

            # The inventory endpoint returns a JSON array of station codes
            # or a JSON object with an error message.
            parsed = json.loads(raw_data)

            if isinstance(parsed, list):
                # Direct list of station code strings
                stations = sorted([str(s).upper().strip() for s in parsed if s])
            elif isinstance(parsed, dict):
                # Some API versions wrap in {"stations": [...]}
                if "stations" in parsed:
                    stations = sorted(
                        [str(s).upper().strip() for s in parsed["stations"] if s]
                    )
                elif "error" in parsed:
                    log.error(
                        "SuperMAG inventory API returned error for %s: %s",
                        month_str, parsed["error"],
                    )
                    return []
                else:
                    # Unknown dict structure — try to extract station-like keys
                    log.warning(
                        "SuperMAG inventory returned unexpected dict structure for %s. "
                        "Keys: %s", month_str, list(parsed.keys()),
                    )
                    return []
            else:
                log.error(
                    "SuperMAG inventory returned unexpected type %s for %s.",
                    type(parsed).__name__, month_str,
                )
                return []

            # Success — break out of retry loop
            break

        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code == 503:
                # Rate limited or service unavailable — back off
                log.warning(
                    "SuperMAG inventory HTTP %d for %s (attempt %d/%d). "
                    "Backing off %.1f sec.",
                    exc.code, month_str, attempt, _API_MAX_RETRIES, backoff,
                )
                _time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            else:
                log.error(
                    "SuperMAG inventory HTTP %d for %s: %s",
                    exc.code, month_str, exc,
                )
                return []

        except urllib.error.URLError as exc:
            log.error(
                "SuperMAG inventory connection failed for %s (attempt %d/%d): %s",
                month_str, attempt, _API_MAX_RETRIES, exc,
            )
            if attempt < _API_MAX_RETRIES:
                _time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            return []

        except json.JSONDecodeError:
            log.error(
                "SuperMAG inventory returned non-JSON for %s (attempt %d/%d). "
                "API may be down.",
                month_str, attempt, _API_MAX_RETRIES,
            )
            if attempt < _API_MAX_RETRIES:
                _time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            return []

    if not stations:
        log.warning("No stations returned by SuperMAG inventory for %s.", month_str)
        return []

    # ------------------------------------------------------------------
    # Geographic coverage summary
    # ------------------------------------------------------------------
    n_stations = len(stations)
    log.info(
        "SuperMAG inventory for %s: %d stations available.",
        month_str, n_stations,
    )

    # ------------------------------------------------------------------
    # Save to cache
    # ------------------------------------------------------------------
    inventory_record = {
        "year": year,
        "month": month,
        "query_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "station_count": n_stations,
        "stations": stations,
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(inventory_record, fh, indent=2)
        log.info(
            "Saved SuperMAG inventory → %s (%d stations)",
            cache_path, n_stations,
        )
    except OSError as exc:
        log.warning(
            "Failed to cache inventory for %s: %s. Continuing without cache.",
            month_str, exc,
        )

    return stations


def load_cached_inventory(year: int, month: int) -> Optional[list[str]]:
    """Load a previously cached station inventory without making API calls.

    Parameters
    ----------
    year, month : int
        Calendar year and month.

    Returns
    -------
    list[str] or None
        Station list if cache exists, None otherwise.
    """
    month_str = f"{year:04d}{month:02d}"
    cache_path = os.path.join(
        _STATION_METADATA_DIR, f"supermag_inventory_{month_str}.json"
    )
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        return cached.get("stations", [])
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _iter_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[tuple[int, int]]:
    start = pd.Period(f"{start_year:04d}-{start_month:02d}", freq="M")
    end = pd.Period(f"{end_year:04d}-{end_month:02d}", freq="M")
    if end < start:
        raise ValueError("End month must be on or after start month.")
    months = pd.period_range(start=start, end=end, freq="M")
    return [(int(period.year), int(period.month)) for period in months]


def _station_metadata_output_path(output_path: str | os.PathLike | None = None) -> Path:
    return Path(output_path or Path(_STATION_METADATA_DIR) / "supermag_station_coords.parquet")


def _empty_station_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": pd.Series([], dtype="string"),
            "glat": pd.Series([], dtype=float),
            "glon": pd.Series([], dtype=float),
            "mlat": pd.Series([], dtype=float),
            "mlon": pd.Series([], dtype=float),
            "mlt_offset": pd.Series([], dtype=float),
            "operational_start": pd.Series([], dtype="string"),
            "operational_end": pd.Series([], dtype="string"),
            "sample_months": pd.Series([], dtype="int64"),
            "metadata_available": pd.Series([], dtype=bool),
        }
    )


def _validate_station_metadata_schema(df: pd.DataFrame) -> None:
    missing = [col for col in _STATION_METADATA_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Station metadata missing required columns: {missing}")
    if df["station"].isna().any() or df["station"].astype(str).str.strip().eq("").any():
        raise ValueError("Station metadata contains null or empty station codes.")
    if df["station"].duplicated().any():
        duplicates = df.loc[df["station"].duplicated(keep=False), "station"].tolist()
        raise ValueError(f"Station metadata contains duplicate station rows: {duplicates[:5]}")


def _cached_raw_supermag_path(year: int, month: int) -> Path:
    month_str = f"{year:04d}{month:02d}"
    return Path(config.RAW_DATA_DIR) / "supermag" / f"{year:04d}" / f"{month:02d}" / f"supermag_{month_str}.parquet"


def _load_cached_station_month(station: str, year: int, month: int) -> pd.DataFrame | None:
    path = _cached_raw_supermag_path(year, month)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.exception("Could not read cached SuperMAG metadata source %s: %s", path, exc)
        return None
    if "station" not in df.columns:
        return None
    station_mask = df["station"].astype(str).str.upper() == station.upper()
    if not station_mask.any():
        return None
    return df.loc[station_mask].copy()


def _month_token(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _mlt_offset_hours(frame: pd.DataFrame) -> float:
    if "mlt" not in frame.columns or "timestamp" not in frame.columns:
        return float("nan")
    work = frame[["timestamp", "mlt"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["mlt"] = pd.to_numeric(work["mlt"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return float("nan")
    utc_hour = (
        work["timestamp"].dt.hour
        + work["timestamp"].dt.minute / 60.0
        + work["timestamp"].dt.second / 3600.0
    )
    offset = (work["mlt"] - utc_hour) % 24.0
    return float(offset.median())


def _station_month_metadata(station: str, frame: pd.DataFrame | None, year: int, month: int) -> dict:
    row = {
        "station": station.upper(),
        "month": _month_token(year, month),
        "glat": float("nan"),
        "glon": float("nan"),
        "mlat": float("nan"),
        "mlon": float("nan"),
        "mlt_offset": float("nan"),
        "metadata_available": False,
    }
    if frame is None or frame.empty:
        return row

    for col in ("glat", "glon", "mlat", "mlon"):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not values.empty:
                row[col] = float(values.median())

    row["mlt_offset"] = _mlt_offset_hours(frame)
    row["metadata_available"] = any(pd.notna(row[col]) for col in ("glat", "glon", "mlat", "mlon"))
    return row


def _collapse_station_metadata(monthly_rows: list[dict]) -> pd.DataFrame:
    if not monthly_rows:
        return _empty_station_metadata()

    monthly = pd.DataFrame(monthly_rows)
    records = []
    for station, group in monthly.groupby("station", sort=True):
        months = sorted(group["month"].astype(str).unique())
        record = {
            "station": station,
            "operational_start": months[0],
            "operational_end": months[-1],
            "sample_months": int(len(months)),
            "metadata_available": bool(group["metadata_available"].any()),
        }
        for col in ("glat", "glon", "mlat", "mlon", "mlt_offset"):
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            record[col] = float(values.median()) if not values.empty else float("nan")
        records.append(record)

    result = pd.DataFrame(records)
    result = result[_STATION_METADATA_COLUMNS].sort_values("station").reset_index(drop=True)
    return result


def precompute_station_metadata(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    *,
    stations: Sequence[str] | None = None,
    output_path: str | os.PathLike | None = None,
    force_inventory_refresh: bool = False,
    use_cached_raw: bool = True,
    fetch_missing_metadata: bool = True,
    request_delay_sec: float = 0.0,
) -> pd.DataFrame:
    """Precompute station coordinates and operational coverage across months.

    Coordinates come from the same SuperMAG ``mag`` and ``geo`` metadata fields
    retrieved with station data. Cached raw SuperMAG files are preferred when
    available; missing station-month metadata can be fetched from the API.
    """
    months = _iter_months(start_year, start_month, end_year, end_month)
    requested_stations = (
        sorted({str(station).upper().strip() for station in stations if str(station).strip()})
        if stations is not None
        else None
    )
    monthly_rows: list[dict] = []

    for year, month in months:
        month_stations = requested_stations
        if month_stations is None:
            month_stations = get_station_inventory(year, month, force_refresh=force_inventory_refresh)
        if not month_stations:
            log.warning("No stations available for metadata precompute in %04d-%02d.", year, month)
            continue

        start_dt, end_dt = _month_window(year, month)
        sample_extent_sec = min(int((end_dt - start_dt).total_seconds()), 24 * 3600)
        for station in month_stations:
            station = station.upper()
            frame = _load_cached_station_month(station, year, month) if use_cached_raw else None
            if (frame is None or frame.empty) and fetch_missing_metadata:
                frame = _fetch_station(station, start_dt, sample_extent_sec)
                if request_delay_sec > 0:
                    time.sleep(request_delay_sec)
            monthly_rows.append(_station_month_metadata(station, frame, year, month))

    metadata = _collapse_station_metadata(monthly_rows)
    _validate_station_metadata_schema(metadata)

    final_output_path = _station_metadata_output_path(output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(final_output_path, index=False)
    log.info("Saved SuperMAG station metadata -> %s (%d stations)", final_output_path, len(metadata))
    return metadata


def _month_window(year: int, month: int) -> tuple[datetime.datetime, datetime.datetime]:
    start_dt = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end_dt = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end_dt = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)
    return start_dt, end_dt


def _month_minutes(year: int, month: int) -> int:
    start_dt, end_dt = _month_window(year, month)
    return int((end_dt - start_dt).total_seconds() // 60)


def _request_json_with_backoff(url: str, station: str, timeout: int = 120) -> object | None:
    """Fetch JSON from SuperMAG with explicit retry/backoff for rate limits."""
    backoff = _API_INITIAL_BACKOFF_SEC
    for attempt in range(1, _API_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 503} and attempt < _API_MAX_RETRIES:
                log.warning(
                    "SuperMAG station %s HTTP %d (attempt %d/%d). Backing off %.1f sec.",
                    station, exc.code, attempt, _API_MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            log.error("SuperMAG station %s HTTP %d: %s", station, exc.code, exc)
            return None
        except urllib.error.URLError as exc:
            log.error(
                "SuperMAG station %s connection failed (attempt %d/%d): %s",
                station, attempt, _API_MAX_RETRIES, exc,
            )
            if attempt < _API_MAX_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            return None
        except json.JSONDecodeError:
            log.error(
                "SuperMAG station %s returned non-JSON (attempt %d/%d).",
                station, attempt, _API_MAX_RETRIES,
            )
            if attempt < _API_MAX_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * 2, _API_MAX_BACKOFF_SEC)
                continue
            return None
    return None


def _as_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if abs(numeric) >= _SUPERMAG_FILL_VALUE_THRESHOLD:
        return float("nan")
    return numeric


def _nested_get(record: dict, *path):
    current = record
    for key in path:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]
    return current


def _first_numeric(record: dict, paths: list[tuple]) -> float:
    for path in paths:
        value = _nested_get(record, *path)
        numeric = _as_float(value)
        if pd.notna(numeric):
            return numeric
    return float("nan")


def _extract_nez_component(record: dict, component: str) -> float:
    return _first_numeric(
        record,
        [
            (component, "nez"),
            (component.upper(), "nez"),
            (component.lower(), "nez"),
            (f"{component.lower()}_nez",),
            (f"{component.upper()}_NEZ",),
        ],
    )


def _parse_station_records(rawlines: object, station: str) -> pd.DataFrame:
    """Parse SuperMAG JSON records into the canonical raw schema."""
    if not isinstance(rawlines, list):
        log.error("SuperMAG station %s returned unexpected payload type %s.", station, type(rawlines).__name__)
        return _empty_station_df()

    records = []
    for line in rawlines:
        if not isinstance(line, dict):
            continue

        timestamp_value = line.get("tval") or line.get("time") or line.get("timestamp")
        timestamp = pd.to_datetime(timestamp_value, unit="s", utc=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")

        records.append({
            "timestamp": timestamp,
            "station": station.upper(),
            "n_nez": _extract_nez_component(line, "N"),
            "e_nez": _extract_nez_component(line, "E"),
            "z_nez": _extract_nez_component(line, "Z"),
            "mlt": _first_numeric(line, [
                ("mlt",), ("MLT",), ("m_lt",), ("N", "mlt"), ("N", "MLT"),
            ]),
            "mlat": _first_numeric(line, [
                ("mlat",), ("MLAT",), ("maglat",), ("MAGLAT",),
                ("mag", "lat"), ("mag", "mlat"), ("mag", "maglat"),
                ("MAG", "lat"), ("MAG", "mlat"), ("MAG", "MAGLAT"),
                ("N", "mag", "lat"), ("N", "mag", "mlat"),
            ]),
            "mlon": _first_numeric(line, [
                ("mlon",), ("MLON",), ("maglon",), ("MAGLON",),
                ("mag", "lon"), ("mag", "mlon"), ("mag", "maglon"),
                ("MAG", "lon"), ("MAG", "mlon"), ("MAG", "MAGLON"),
                ("N", "mag", "lon"), ("N", "mag", "mlon"),
            ]),
            "glat": _first_numeric(line, [
                ("glat",), ("GLAT",), ("geolat",), ("GEOLAT",),
                ("geo", "lat"), ("geo", "glat"), ("geo", "geolat"),
                ("GEO", "lat"), ("GEO", "glat"), ("GEO", "GEOLAT"),
                ("N", "geo", "lat"), ("N", "geo", "glat"),
            ]),
            "glon": _first_numeric(line, [
                ("glon",), ("GLON",), ("geolon",), ("GEOLON",),
                ("geo", "lon"), ("geo", "glon"), ("geo", "geolon"),
                ("GEO", "lon"), ("GEO", "glon"), ("GEO", "GEOLON"),
                ("N", "geo", "lon"), ("N", "geo", "glon"),
            ]),
        })

    if not records:
        return _empty_station_df()

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp", "station"], keep="first")
    return df[_raw_supermag_columns()].sort_values("timestamp").reset_index(drop=True)


def _fetch_station(
    station: str,
    start_dt: datetime.datetime,
    extent_sec: int,
) -> Optional[pd.DataFrame]:
    """Fetch one station's data from SuperMAG API.

    Returns canonical raw columns or None
    if the request failed or returned no data.
    """
    api_flags = getattr(config, "SUPERMAG_API_FLAGS", None) or "mlt,mag,geo"
    flag_query = "".join(f"&{flag.strip()}" for flag in api_flags.split(",") if flag.strip())
    url = (
        f"{SUPERMAG_API_BASE}?fmt=json&python&nohead&"
        f"start={start_dt.strftime('%Y-%m-%dT%H:%M')}&extent={extent_sec:012d}&"
        f"logon={SUPERMAG_USERNAME}&all{flag_query}&station={station.upper()}"
    )
    rawlines = _request_json_with_backoff(url, station.upper())
    if rawlines is None:
        return None

    if not rawlines:
        log.warning("No data returned for SuperMAG station %s.", station)
        return None

    return _parse_station_records(rawlines, station)


def _raw_supermag_columns() -> list[str]:
    return ["timestamp", "station", "n_nez", "e_nez", "z_nez", "mlt", "mlat", "mlon", "glat", "glon"]


def _empty_station_df() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical SuperMAG schema."""
    cols = {
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
        "station": pd.Series([], dtype=str),
        "n_nez": pd.Series([], dtype=float),
        "e_nez": pd.Series([], dtype=float),
        "z_nez": pd.Series([], dtype=float),
        "mlt": pd.Series([], dtype=float),
        "mlat": pd.Series([], dtype=float),
        "mlon": pd.Series([], dtype=float),
        "glat": pd.Series([], dtype=float),
        "glon": pd.Series([], dtype=float),
    }
    return pd.DataFrame(cols)


def retrieve_supermag_month(
    year: int,
    month: int,
    stations: Optional[list] = None,
    use_inventory: bool = True,
    force_inventory_refresh: bool = False,
    request_delay_sec: float = 0.0,
) -> Optional[pd.DataFrame]:
    """Retrieve SuperMAG data for all stations in one calendar month.

    Produces raw long-format Parquet with one row per (station, minute).

    Parameters
    ----------
    year, month:
        Calendar year and month.
    stations:
        IAGA station codes to retrieve. If None and ``use_inventory`` is
        True, queries the dynamic inventory via ``get_station_inventory()``.
        Passing a station list is intended for tests or targeted debugging.
    use_inventory : bool, default True
        If True, use dynamic inventory to discover available stations.
        This is the recommended production mode (Warning #19).
    force_inventory_refresh:
        If True, bypass the cached monthly inventory and re-query SuperMAG.
    request_delay_sec:
        Optional pause between station requests. Useful for conservative
        long-running all-station pulls.
    """
    if not SUPERMAG_USERNAME:
        raise RuntimeError("SUPERMAG_USERNAME env variable not set. Export it before running.")

    if stations is None:
        if use_inventory:
            stations = get_station_inventory(year, month, force_refresh=force_inventory_refresh)
            if not stations:
                raise RuntimeError(
                    "No stations available from dynamic inventory for %04d-%02d. "
                    "Check API connectivity and credentials." % (year, month)
                )
            log.info(
                "Using dynamic inventory: %d stations for %04d-%02d.",
                len(stations), year, month,
            )
        else:
            stations = config.SUPERMAG_STATIONS
            log.warning(
                "Using static station list (config.SUPERMAG_STATIONS = %s). "
                "Set use_inventory=True for production runs to use dynamic "
                "per-month station discovery.",
                stations,
            )
    stations = sorted({str(station).upper().strip() for station in stations if str(station).strip()})
    if not stations:
        raise ValueError("No SuperMAG stations requested.")

    month_str = f"{year:04d}{month:02d}"
    out_dir = os.path.join(config.RAW_DATA_DIR, "supermag", f"{year:04d}", f"{month:02d}")
    output_path = os.path.join(out_dir, f"supermag_{month_str}.parquet")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        log.debug("SuperMAG raw %s already exists, loading: %s", month_str, output_path)
        return pd.read_parquet(output_path)

    start_dt, end_dt = _month_window(year, month)
    minutes_in_month = _month_minutes(year, month)
    extent_sec = int((end_dt - start_dt).total_seconds())

    frames = []
    for idx, station in enumerate(stations, start=1):
        station = station.upper()
        log.info(
            "Retrieving SuperMAG station %s for %s (%d/%d)...",
            station, month_str, idx, len(stations),
        )

        raw_df = _fetch_station(station, start_dt, extent_sec)
        if raw_df is None or raw_df.empty:
            log.warning(
                "Station %s returned empty data for %s.", station, month_str
            )
            if request_delay_sec > 0:
                time.sleep(request_delay_sec)
            continue

        # Completeness check
        completeness = raw_df["timestamp"].nunique() / max(minutes_in_month, 1)
        if completeness < _COMPLETENESS_WARN_THRESHOLD:
            log.warning(
                "Station %s completeness for %s: %.1f%% (%d/%d rows).",
                station, month_str, completeness * 100, raw_df["timestamp"].nunique(), minutes_in_month,
            )

        frames.append(raw_df[_raw_supermag_columns()])
        log.info("  %s %s: %d rows, completeness=%.1f%%",
                 station, month_str, len(raw_df), completeness * 100)

        if request_delay_sec > 0:
            time.sleep(request_delay_sec)

    if not frames or all(len(f) == 0 for f in frames):
        log.warning("All stations empty for SuperMAG %s. Writing empty Parquet.", month_str)
        empty = _empty_station_df()
        validate_output_schema(empty, f"SuperMAG-{month_str}", unique_subset=["timestamp", "station"])
        empty.to_parquet(output_path, index=False)
        return empty

    df_all = pd.concat(frames, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
    df_all = df_all.sort_values(["timestamp", "station"]).reset_index(drop=True)

    # Long-format: duplicate timestamps are expected (one row per station per minute).
    # Uniqueness constraint is the composite key (timestamp, station).
    validate_output_schema(df_all, f"SuperMAG-{month_str}", unique_subset=["timestamp", "station"])
    df_all.to_parquet(output_path, index=False)
    log.info(
        "Saved raw SuperMAG %s → %s (%d rows, %d/%d stations with data)",
        month_str, output_path, len(df_all), df_all["station"].nunique(), len(stations),
    )
    return df_all


if __name__ == "__main__":
    retrieve_supermag_month(year=2015, month=3)
