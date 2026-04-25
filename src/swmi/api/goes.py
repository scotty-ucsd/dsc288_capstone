#!/usr/bin/env python3
"""Unified GOES retrievers.

This module owns the active GOES pipeline implementation. Archived scripts under
``archive/scripts/`` remain reference material only.

P0-G2 implements the magnetometer path:

- satellite-number era detection
- legacy and modern NOAA directory routing
- NetCDF parsing into canonical GSM Bz
- deterministic primary/backup merge from ``configs/data_retrieval.yaml``
- canonical monthly Parquet output at ``data/raw/goes/goes_mag_YYYYMM.parquet``
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr

from swmi.preprocessing.validation import validate_output_schema
from swmi.utils.config import load_config
from swmi.utils.logger import get_logger

log = get_logger(__name__)

Era = Literal["legacy", "modern"]

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_J2K_EPOCH = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")
_MAX_RETRIES = 3
_RETRY_SLEEP_SEC = 2.0

_NCEI_MAG_BASE_URL = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/mag/goes{sat:02d}/magn-l2-hires/{year:04d}/{month:02d}/"
)
_NGDC_MAG_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes{sat}/l2/data/magn-l2-avg1m/{year:04d}/{month:02d}/"
)
_NCEI_XRS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/xrs/goes{sat}/xrsf-l2-avg1m_science/"
)
_NGDC_XRS_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes{sat}/l2/data/xrsf-l2-avg1m_science/"
)

_XRS_CONTAMINATION_BITS = (8, 16, 32, 64, 128, 256)
_XRSA_VALID_RANGE = (1e-9, 3e-3)
_XRSB_VALID_RANGE = (1e-9, 2e-1)


def parse_goes_satellite_number(satellite: str | int) -> int:
    """Parse GOES satellite identifiers like ``GOES-16``, ``g16``, or ``16``."""
    if isinstance(satellite, int):
        number = satellite
    else:
        match = re.search(r"(\d{2})", satellite)
        if match is None:
            raise ValueError(f"Could not parse GOES satellite number from {satellite!r}")
        number = int(match.group(1))

    if number < 1:
        raise ValueError(f"Invalid GOES satellite number: {number}")
    return number


def format_goes_satellite(satellite: str | int) -> str:
    """Return the canonical satellite label, e.g. ``GOES-16``."""
    return f"GOES-{parse_goes_satellite_number(satellite)}"


def detect_goes_era(satellite: str | int) -> Era:
    """Route GOES satellites by spacecraft generation."""
    number = parse_goes_satellite_number(satellite)
    return "legacy" if number <= 15 else "modern"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC")
    return start, end


def _expected_minute_index(year: int, month: int) -> pd.DatetimeIndex:
    start, end = _month_bounds(year, month)
    return pd.date_range(start, end, freq="1min", inclusive="left")


def _decode_time_values(values: np.ndarray, units: str | None = None) -> pd.Series:
    """Decode GOES time arrays to UTC timestamps.

    GOES NetCDF files commonly use seconds since J2000, but some XRS science
    files expose milliseconds or ISO-like decoded values. This keeps retrieval
    tolerant of both legacy and modern products.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Series(pd.to_datetime(values, utc=True))

    unit = "s"
    units_lower = (units or "").lower()
    if "millisecond" in units_lower or "milliseconds" in units_lower:
        unit = "ms"
    elif "microsecond" in units_lower or "microseconds" in units_lower:
        unit = "us"
    elif "nanosecond" in units_lower or "nanoseconds" in units_lower:
        unit = "ns"
    elif "minute" in units_lower or "minutes" in units_lower:
        unit = "m"
    elif "hour" in units_lower or "hours" in units_lower:
        unit = "h"

    epoch = _J2K_EPOCH
    match = re.search(r"since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units or "")
    if match:
        time_part = match.group(2) or "00:00:00"
        epoch = pd.Timestamp(f"{match.group(1)} {time_part}", tz="UTC")

    return pd.Series(epoch + pd.to_timedelta(values, unit=unit))


def _load_goes_priority(year: int, config_path: Path | str | None = None) -> list[str]:
    path = Path(config_path or _repo_root() / "configs" / "data_retrieval.yaml")
    cfg = load_config(path)
    priorities = cfg["goes"]["satellite_priority"]
    if year in priorities:
        configured = priorities[year]
    elif str(year) in priorities:
        configured = priorities[str(year)]
    else:
        raise KeyError(f"No GOES satellite priority configured for {year}")
    return [format_goes_satellite(sat) for sat in configured]


def _empty_mag_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": index,
            "goes_bz_gsm": np.nan,
            "goes_source_satellite": pd.Series([pd.NA] * len(index), dtype="string"),
            "goes_mag_missing_flag": np.ones(len(index), dtype="int8"),
        }
    )


def _coerce_utc_minute_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("GOES data is missing required 'timestamp' column")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    return out.reset_index(drop=True)


def _extract_bz_column(df: pd.DataFrame) -> pd.Series:
    for col in ("goes_bz_gsm", "B_Z_GSM", "BZ_GSM", "Bz_GSM"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    raise KeyError("GOES data is missing a GSM Bz column")


def _canonicalize_satellite_mag(df: pd.DataFrame, satellite: str | int) -> pd.DataFrame:
    """Normalize one satellite's magnetometer data to the merge input schema."""
    sat_label = format_goes_satellite(satellite)
    out = _coerce_utc_minute_frame(df)
    out["goes_bz_gsm"] = _extract_bz_column(out).where(lambda s: s > -9000.0, np.nan)
    out["goes_source_satellite"] = sat_label
    return out[["timestamp", "goes_bz_gsm", "goes_source_satellite"]]


def _empty_xray_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": index,
            "xrsa_flux": np.nan,
            "xrsb_flux": np.nan,
            "xray_quality_flags": pd.Series([pd.NA] * len(index), dtype="string"),
            "xray_source_satellite": pd.Series([pd.NA] * len(index), dtype="string"),
            "xray_missing_flag": np.ones(len(index), dtype="int8"),
        }
    )


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower_to_actual = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _extract_flux_column(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    col = _first_existing_column(df, candidates)
    if col is None:
        raise KeyError(f"GOES XRS data is missing one of {list(candidates)}")
    return pd.to_numeric(df[col], errors="coerce")


def _quality_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    col = _first_existing_column(df, candidates)
    if col is None:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[col], errors="coerce").fillna(1).astype("int64")


def _modern_electron_flag(df: pd.DataFrame) -> pd.Series:
    col = _first_existing_column(
        df,
        (
            "electron_correction_flag",
            "electron_correction",
            "electron_contamination_flag",
            "electron_contamination",
        ),
    )
    if col is None:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")


def _modern_contamination_mask(flags: pd.Series) -> pd.Series:
    contaminated = flags.isin((1, 4))
    for bit in _XRS_CONTAMINATION_BITS:
        contaminated |= (flags.astype("int64") & bit) != 0
    return contaminated


def _xrs_flux_in_range(flux: pd.Series, valid_range: tuple[float, float]) -> pd.Series:
    valid_min, valid_max = valid_range
    return flux.ge(valid_min) & flux.le(valid_max)


def _canonicalize_satellite_xray(df: pd.DataFrame, satellite: str | int, era: Era) -> pd.DataFrame:
    """Normalize and quality-filter one satellite's XRS data."""
    sat_label = format_goes_satellite(satellite)
    out = _coerce_utc_minute_frame(df)

    xrsa = _extract_flux_column(out, ("xrsa_flux", "xrsa", "a_flux", "xs", "short_flux"))
    xrsb = _extract_flux_column(out, ("xrsb_flux", "xrsb", "b_flux", "xl", "long_flux"))
    a_flag = _quality_series(out, ("xrsa_flag", "xrsa_quality_flag", "a_quality_flag", "quality_flag"))
    b_flag = _quality_series(out, ("xrsb_flag", "xrsb_quality_flag", "b_quality_flag", "quality_flag"))

    if era == "modern":
        electron_flag = _modern_electron_flag(out)
        contamination = _modern_contamination_mask(electron_flag)
        a_valid = (a_flag == 0) & ~contamination & _xrs_flux_in_range(xrsa, _XRSA_VALID_RANGE)
        b_valid = (b_flag == 0) & ~contamination & _xrs_flux_in_range(xrsb, _XRSB_VALID_RANGE)
        quality_summary = (
            "a="
            + a_flag.astype(str)
            + ";b="
            + b_flag.astype(str)
            + ";electron="
            + electron_flag.astype(str)
        )
    else:
        a_valid = (a_flag == 0) & _xrs_flux_in_range(xrsa, _XRSA_VALID_RANGE)
        b_valid = (b_flag == 0) & _xrs_flux_in_range(xrsb, _XRSB_VALID_RANGE)
        quality_summary = "a=" + a_flag.astype(str) + ";b=" + b_flag.astype(str)

    result = pd.DataFrame(
        {
            "timestamp": out["timestamp"],
            "xrsa_flux": xrsa.where(a_valid, np.nan),
            "xrsb_flux": xrsb.where(b_valid, np.nan),
            "xray_quality_flags": quality_summary.astype("string"),
            "xray_source_satellite": sat_label,
        }
    )
    return result


def merge_goes_xray_satellites(
    satellite_frames: Mapping[str, pd.DataFrame],
    priority: Sequence[str | int],
    *,
    expected_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Merge XRS frames using deterministic satellite priority."""
    frames_by_sat = {
        format_goes_satellite(sat): _coerce_utc_minute_frame(df)
        for sat, df in satellite_frames.items()
        if df is not None and not df.empty
    }
    ordered_sats = [format_goes_satellite(sat) for sat in priority]
    merged = _empty_xray_frame(expected_index).set_index("timestamp")

    for sat_label in ordered_sats:
        frame = frames_by_sat.get(sat_label)
        if frame is None or frame.empty:
            continue
        candidate = frame.set_index("timestamp").reindex(expected_index)
        has_candidate = candidate["xrsa_flux"].notna() | candidate["xrsb_flux"].notna()
        empty_slot = merged["xrsa_flux"].isna() & merged["xrsb_flux"].isna()
        take = has_candidate & empty_slot
        for col in ("xrsa_flux", "xrsb_flux", "xray_quality_flags"):
            merged.loc[take, col] = candidate.loc[take, col]
        merged.loc[take, "xray_source_satellite"] = sat_label

    merged["xray_missing_flag"] = (
        merged["xrsa_flux"].isna() & merged["xrsb_flux"].isna()
    ).astype("int8")
    result = merged.reset_index().rename(columns={"index": "timestamp"})
    result = result.sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("GOES X-ray merge produced duplicate timestamps")
    return result[
        [
            "timestamp",
            "xrsa_flux",
            "xrsb_flux",
            "xray_quality_flags",
            "xray_source_satellite",
            "xray_missing_flag",
        ]
    ]


def merge_goes_satellites(
    satellite_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    priority: Sequence[str | int] | None = None,
    *,
    year: int | None = None,
    config_path: Path | str | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Merge GOES satellite frames with deterministic primary/backup priority.

    Parameters
    ----------
    satellite_frames:
        Mapping from satellite label to DataFrame, or a sequence of DataFrames
        that already contain ``goes_source_satellite``.
    priority:
        Ordered satellite priority. The first valid value at each timestamp wins.
        If omitted, pass ``year`` to load priority from ``configs/data_retrieval.yaml``.
    year:
        Year used to load configured priority when ``priority`` is omitted.
    config_path:
        Optional YAML path for tests or alternate deployments.
    expected_index:
        Optional complete minute index. Pass this during production retrieval to
        retain explicit missing rows for the whole month.
    """
    if priority is None and year is not None:
        priority = _load_goes_priority(year, config_path)

    if isinstance(satellite_frames, Mapping):
        frames_by_sat = {
            format_goes_satellite(sat): _canonicalize_satellite_mag(df, sat)
            for sat, df in satellite_frames.items()
            if df is not None and not df.empty
        }
        ordered_sats = [format_goes_satellite(sat) for sat in (priority or satellite_frames.keys())]
    else:
        frames_by_sat = {}
        for i, frame in enumerate(satellite_frames):
            if frame is None or frame.empty:
                continue
            canonical = _coerce_utc_minute_frame(frame)
            if "goes_source_satellite" in canonical.columns:
                sat_label = str(canonical["goes_source_satellite"].dropna().iloc[0])
            elif priority and i < len(priority):
                sat_label = format_goes_satellite(priority[i])
            else:
                sat_label = f"GOES-{i}"
            frames_by_sat[format_goes_satellite(sat_label)] = _canonicalize_satellite_mag(
                canonical, sat_label
            )
        ordered_sats = [format_goes_satellite(sat) for sat in (priority or frames_by_sat.keys())]

    if expected_index is None:
        timestamps = [
            frame["timestamp"]
            for frame in frames_by_sat.values()
            if frame is not None and not frame.empty
        ]
        if not timestamps:
            return _empty_mag_frame(pd.DatetimeIndex([], tz="UTC"))
        expected_index = pd.DatetimeIndex(pd.concat(timestamps).drop_duplicates().sort_values())

    merged = _empty_mag_frame(expected_index)
    merged = merged.set_index("timestamp")

    for sat_label in ordered_sats:
        frame = frames_by_sat.get(sat_label)
        if frame is None or frame.empty:
            continue

        candidate = frame.set_index("timestamp").reindex(expected_index)
        valid = candidate["goes_bz_gsm"].notna() & merged["goes_bz_gsm"].isna()
        merged.loc[valid, "goes_bz_gsm"] = candidate.loc[valid, "goes_bz_gsm"]
        merged.loc[valid, "goes_source_satellite"] = sat_label

    merged["goes_mag_missing_flag"] = merged["goes_bz_gsm"].isna().astype("int8")
    merged = merged.reset_index().rename(columns={"index": "timestamp"})
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if merged["timestamp"].duplicated().any():
        raise ValueError("GOES merge produced duplicate timestamps")

    primary = ordered_sats[0] if ordered_sats else None
    selected_source = merged["goes_source_satellite"].astype("string")
    valid = merged["goes_bz_gsm"].notna()
    backup_filled = valid & selected_source.notna()
    if primary is not None:
        backup_filled &= selected_source != primary
    gap_summary = {
        "expected_minutes": int(len(merged)),
        "valid_minutes": int(valid.sum()),
        "missing_minutes": int(merged["goes_mag_missing_flag"].sum()),
        "backup_filled_minutes": int(backup_filled.sum()),
    }
    result = merged[["timestamp", "goes_bz_gsm", "goes_source_satellite", "goes_mag_missing_flag"]]
    result.attrs["gap_summary"] = gap_summary
    return result


@dataclass
class BaseRetriever:
    """Shared GOES retrieval helpers."""

    config_path: Path | str | None = None
    output_root: Path | str | None = None

    def __post_init__(self) -> None:
        root = _repo_root()
        self.config_path = Path(self.config_path or root / "configs" / "data_retrieval.yaml")
        self.config = load_config(self.config_path)
        self.output_root = Path(
            self.output_root or root / self.config["directories"]["raw"] / "goes"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)

    def detect_era(self, satellite: str | int) -> Era:
        return detect_goes_era(satellite)

    def month_bounds(self, year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        return _month_bounds(year, month)

    def expected_minute_index(self, year: int, month: int) -> pd.DatetimeIndex:
        return _expected_minute_index(year, month)

    def satellite_priority(self, year: int) -> list[str]:
        priorities = self.config["goes"]["satellite_priority"]
        if year in priorities:
            return [format_goes_satellite(sat) for sat in priorities[year]]
        if str(year) in priorities:
            return [format_goes_satellite(sat) for sat in priorities[str(year)]]
        raise KeyError(f"No GOES satellite priority configured for {year}")

    def is_operational(self, satellite: str | int, year: int, month: int) -> bool:
        periods = self.config.get("goes", {}).get("operational_periods", [])
        if not periods:
            return True

        sat_label = format_goes_satellite(satellite)
        month_start, month_end = self.month_bounds(year, month)
        for period in periods:
            if format_goes_satellite(period["satellite"]) != sat_label:
                continue
            start = pd.Timestamp(period["start_date"], tz="UTC")
            end = pd.Timestamp(period["end_date"], tz="UTC") + pd.Timedelta(days=1)
            return month_start < end and month_end > start
        return False

    def validate_and_write(
        self,
        df: pd.DataFrame,
        schema_name: str,
        output_path: Path,
        *,
        unique_subset: list[str] | None = None,
    ) -> Path:
        validate_output_schema(df, schema_name, unique_subset=unique_subset)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        return output_path


class GOESMagRetriever(BaseRetriever):
    """GOES magnetometer retriever with automatic legacy/modern routing."""

    product = "mag"

    def retrieve(
        self,
        year: int,
        month: int,
        *,
        satellites: Sequence[str | int] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Retrieve, merge, validate, and write one month of GOES magnetometer data."""
        month_str = f"{year:04d}{month:02d}"
        output_path = Path(self.output_root) / f"goes_mag_{month_str}.parquet"
        if output_path.exists() and not force:
            log.info("GOES mag %s already exists, loading %s", month_str, output_path)
            return pd.read_parquet(output_path)

        priority = [format_goes_satellite(sat) for sat in (satellites or self.satellite_priority(year))]
        expected_index = self.expected_minute_index(year, month)
        satellite_frames: dict[str, pd.DataFrame] = {}

        for sat_label in priority:
            if not self.is_operational(sat_label, year, month):
                log.warning("%s is outside configured operational period for %s", sat_label, month_str)
                continue

            raw = self._retrieve_satellite_mag(sat_label, year, month)
            if raw is None or raw.empty:
                log.warning("No GOES magnetometer rows for %s %s", sat_label, month_str)
                continue
            satellite_frames[sat_label] = _canonicalize_satellite_mag(raw, sat_label)

        merged = merge_goes_satellites(
            satellite_frames,
            priority,
            expected_index=expected_index,
        )
        self.validate_and_write(
            merged,
            f"GOES-mag-{month_str}",
            output_path,
            unique_subset=["timestamp"],
        )
        valid = int(merged["goes_bz_gsm"].notna().sum())
        log.info(
            "Saved GOES magnetometer %s -> %s (%d/%d valid minutes)",
            month_str,
            output_path,
            valid,
            len(merged),
        )
        return merged

    def _retrieve_satellite_mag(
        self,
        satellite: str | int,
        year: int,
        month: int,
    ) -> pd.DataFrame | None:
        sat_number = parse_goes_satellite_number(satellite)
        if self.detect_era(satellite) == "legacy":
            return self._retrieve_legacy_mag(sat_number, year, month)
        return self._retrieve_modern_mag(sat_number, year, month)

    def _retrieve_legacy_mag(self, satellite: int, year: int, month: int) -> pd.DataFrame | None:
        month_url = _NCEI_MAG_BASE_URL.format(sat=satellite, year=year, month=month)
        pattern = rf'href="(dn_magn-l2-hires_g{satellite:02d}_d\d{{8}}_v[\d_]+\.nc)"'
        return self._retrieve_nc_month(month_url, pattern, satellite, year, month, "legacy")

    def _retrieve_modern_mag(self, satellite: int, year: int, month: int) -> pd.DataFrame | None:
        month_url = _NGDC_MAG_BASE_URL.format(sat=satellite, year=year, month=month)
        pattern = rf'href="(dn_magn-l2-avg1m_g{satellite}_d\d{{8}}_v[\d_]+\.nc)"'
        return self._retrieve_nc_month(month_url, pattern, satellite, year, month, "modern")

    def _retrieve_nc_month(
        self,
        month_url: str,
        file_pattern: str,
        satellite: int,
        year: int,
        month: int,
        era: Era,
    ) -> pd.DataFrame | None:
        log.info("Scanning %s GOES-%d magnetometer directory: %s", era, satellite, month_url)
        try:
            response = requests.get(month_url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
        except Exception as exc:
            log.error("Failed to access GOES-%d %04d-%02d directory: %s", satellite, year, month, exc)
            return None

        nc_files = sorted(set(re.findall(file_pattern, response.text)))
        if not nc_files:
            log.warning("No GOES-%d magnetometer files found for %04d-%02d", satellite, year, month)
            return None

        raw_dir = Path(self.output_root) / era / f"{year:04d}" / f"{month:02d}" / f"goes{satellite}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        daily_frames: list[pd.DataFrame] = []

        for filename in nc_files:
            file_url = f"{month_url}{filename}"
            local_path = raw_dir / filename
            parsed = self._download_and_parse_nc(file_url, local_path)
            if parsed is None or parsed.empty:
                continue

            daily_1min = (
                parsed.set_index("timestamp")
                .resample("1min")
                .mean(numeric_only=True)
                .reset_index()
            )
            daily_frames.append(daily_1min)

        if not daily_frames:
            return None

        month_df = pd.concat(daily_frames, ignore_index=True)
        month_df = _coerce_utc_minute_frame(month_df)

        start, end = self.month_bounds(year, month)
        in_month = (month_df["timestamp"] >= start) & (month_df["timestamp"] < end)
        return month_df.loc[in_month].reset_index(drop=True)

    def _download_and_parse_nc(self, file_url: str, local_path: Path) -> pd.DataFrame | None:
        for attempt in range(1, _MAX_RETRIES + 1):
            if not local_path.exists():
                try:
                    with requests.get(file_url, stream=True, headers=_HEADERS, timeout=60) as response:
                        response.raise_for_status()
                        with open(local_path, "wb") as fh:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    fh.write(chunk)
                except Exception as exc:
                    log.warning(
                        "GOES download failed for %s (attempt %d/%d): %s",
                        file_url,
                        attempt,
                        _MAX_RETRIES,
                        exc,
                    )
                    local_path.unlink(missing_ok=True)
                    time.sleep(_RETRY_SLEEP_SEC)
                    continue

            parsed = self._parse_mag_netcdf(local_path)
            if parsed is not None:
                return parsed

            local_path.unlink(missing_ok=True)
            time.sleep(_RETRY_SLEEP_SEC)

        return None

    @staticmethod
    def _parse_mag_netcdf(local_path: Path) -> pd.DataFrame | None:
        try:
            with xr.open_dataset(local_path, engine="netcdf4", decode_times=False) as ds:
                times = ds["time"].values
                b_gsm = ds["b_gsm"].values
        except Exception as exc:
            log.warning("Could not parse GOES NetCDF %s: %s", local_path, exc)
            return None

        if b_gsm.ndim != 2 or b_gsm.shape[1] < 3:
            log.warning("GOES NetCDF %s has unexpected b_gsm shape %s", local_path, b_gsm.shape)
            return None

        timestamps = _J2K_EPOCH + pd.to_timedelta(times, unit="s")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "B_X_GSM": b_gsm[:, 0],
                "B_Y_GSM": b_gsm[:, 1],
                "B_Z_GSM": b_gsm[:, 2],
            }
        )
        for col in ("B_X_GSM", "B_Y_GSM", "B_Z_GSM"):
            df[col] = pd.to_numeric(df[col], errors="coerce").where(lambda s: s > -9000.0, np.nan)
        return _coerce_utc_minute_frame(df)


class GOESXrayRetriever(BaseRetriever):
    """GOES X-ray flux retriever with era-specific quality filtering."""

    product = "xray"

    def retrieve(
        self,
        year: int,
        month: int,
        *,
        satellites: Sequence[str | int] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Retrieve, merge, validate, and write one month of GOES X-ray flux."""
        month_str = f"{year:04d}{month:02d}"
        output_path = Path(self.output_root) / f"goes_xray_{month_str}.parquet"
        if output_path.exists() and not force:
            log.info("GOES X-ray %s already exists, loading %s", month_str, output_path)
            return pd.read_parquet(output_path)

        priority = [format_goes_satellite(sat) for sat in (satellites or self.satellite_priority(year))]
        expected_index = self.expected_minute_index(year, month)
        satellite_frames: dict[str, pd.DataFrame] = {}

        for sat_label in priority:
            if not self.is_operational(sat_label, year, month):
                log.warning("%s is outside configured operational period for %s", sat_label, month_str)
                continue

            raw = self._retrieve_satellite_xray(sat_label, year, month)
            if raw is None or raw.empty:
                log.warning("No GOES X-ray rows for %s %s", sat_label, month_str)
                continue
            era = self.detect_era(sat_label)
            satellite_frames[sat_label] = _canonicalize_satellite_xray(raw, sat_label, era)

        merged = merge_goes_xray_satellites(
            satellite_frames,
            priority,
            expected_index=expected_index,
        )
        self.validate_and_write(
            merged,
            f"GOES-xray-{month_str}",
            output_path,
            unique_subset=["timestamp"],
        )
        valid = int((merged["xrsa_flux"].notna() | merged["xrsb_flux"].notna()).sum())
        log.info(
            "Saved GOES X-ray %s -> %s (%d/%d valid minutes)",
            month_str,
            output_path,
            valid,
            len(merged),
        )
        return merged

    def _retrieve_satellite_xray(
        self,
        satellite: str | int,
        year: int,
        month: int,
    ) -> pd.DataFrame | None:
        sat_number = parse_goes_satellite_number(satellite)
        if self.detect_era(satellite) == "legacy":
            return self._retrieve_legacy_xray(sat_number, year, month)
        return self._retrieve_modern_xray(sat_number, year, month)

    def _retrieve_legacy_xray(self, satellite: int, year: int, month: int) -> pd.DataFrame | None:
        base_url = _NCEI_XRS_BASE_URL.format(sat=satellite)
        pattern = rf'href="(sci_xrsf-l2-avg1m_g{satellite}_y{year:04d}_v[\d\-_.]+\.nc)"'
        return self._retrieve_xray_year_file(base_url, pattern, satellite, year, month, "legacy")

    def _retrieve_modern_xray(self, satellite: int, year: int, month: int) -> pd.DataFrame | None:
        base_url = _NGDC_XRS_BASE_URL.format(sat=satellite)
        pattern = rf'href="(sci_xrsf-l2-avg1m_g{satellite}_y{year:04d}_v[\d\-_.]+\.nc)"'
        return self._retrieve_xray_year_file(base_url, pattern, satellite, year, month, "modern")

    def _retrieve_xray_year_file(
        self,
        base_url: str,
        file_pattern: str,
        satellite: int,
        year: int,
        month: int,
        era: Era,
    ) -> pd.DataFrame | None:
        log.info("Scanning %s GOES-%d XRS directory: %s", era, satellite, base_url)
        try:
            response = requests.get(base_url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
        except Exception as exc:
            log.error("Failed to access GOES-%d XRS directory for %04d: %s", satellite, year, exc)
            return None

        nc_files = sorted(set(re.findall(file_pattern, response.text)))
        if not nc_files:
            log.warning("No GOES-%d XRS science file found for %04d", satellite, year)
            return None

        filename = nc_files[-1]
        file_url = f"{base_url}{filename}"
        raw_dir = Path(self.output_root) / "xray" / era / f"{year:04d}" / f"goes{satellite}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        local_path = raw_dir / filename

        parsed = self._download_and_parse_xray(file_url, local_path)
        if parsed is None or parsed.empty:
            return None

        start, end = self.month_bounds(year, month)
        in_month = (parsed["timestamp"] >= start) & (parsed["timestamp"] < end)
        return parsed.loc[in_month].reset_index(drop=True)

    def _download_and_parse_xray(self, file_url: str, local_path: Path) -> pd.DataFrame | None:
        for attempt in range(1, _MAX_RETRIES + 1):
            if not local_path.exists():
                try:
                    with requests.get(file_url, stream=True, headers=_HEADERS, timeout=120) as response:
                        response.raise_for_status()
                        with open(local_path, "wb") as fh:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    fh.write(chunk)
                except Exception as exc:
                    log.warning(
                        "GOES XRS download failed for %s (attempt %d/%d): %s",
                        file_url,
                        attempt,
                        _MAX_RETRIES,
                        exc,
                    )
                    local_path.unlink(missing_ok=True)
                    time.sleep(_RETRY_SLEEP_SEC)
                    continue

            parsed = self._parse_xray_netcdf(local_path)
            if parsed is not None:
                return parsed

            local_path.unlink(missing_ok=True)
            time.sleep(_RETRY_SLEEP_SEC)

        return None

    @staticmethod
    def _parse_xray_netcdf(local_path: Path) -> pd.DataFrame | None:
        try:
            with xr.open_dataset(local_path, engine="netcdf4", decode_times=False) as ds:
                if "time" not in ds:
                    raise KeyError("time")
                times = _decode_time_values(ds["time"].values, ds["time"].attrs.get("units"))
                data: dict[str, pd.Series | np.ndarray] = {"timestamp": times}
                for name in ds.data_vars:
                    values = ds[name].values
                    if getattr(values, "ndim", 0) == 1 and len(values) == len(times):
                        data[name] = values
        except Exception as exc:
            log.warning("Could not parse GOES XRS NetCDF %s: %s", local_path, exc)
            return None

        return _coerce_utc_minute_frame(pd.DataFrame(data))


def retrieve_goes_mag(year: int, month: int, *, force: bool = False) -> pd.DataFrame:
    """Convenience entry point for scripts."""
    return GOESMagRetriever().retrieve(year, month, force=force)


def retrieve_goes_xray(year: int, month: int, *, force: bool = False) -> pd.DataFrame:
    """Convenience entry point for GOES X-ray retrieval."""
    return GOESXrayRetriever().retrieve(year, month, force=force)


def _main() -> None:
    GOESMagRetriever().retrieve(year=2015, month=3)


if __name__ == "__main__":
    _main()
