"""
schema.py
Schema validation for all Parquet outputs in the dB/dt forecasting pipeline.

Call ``validate_output_schema(df, source_name)`` immediately before writing
any Parquet file to catch structural violations before they corrupt downstream
temporal joins or training sequences.

Conventions guaranteed by this module:
- Every output DataFrame has a column named exactly ``timestamp``.
- ``df["timestamp"]`` is ``datetime64[ns, UTC]`` (timezone-aware UTC).
- No duplicate timestamps within a single-source monthly file.
- Columns that are entirely NaN are logged at WARNING level.

TODO: Add physical range checks; fix allow_duplicates bug
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_pkg_root, "utils"))

from logger import get_logger

log = get_logger(__name__)


def validate_output_schema(df: pd.DataFrame, source_name: str, unique_subset: list[str] | None = None) -> None:
    """Validate structural invariants for a pipeline output DataFrame.

    This function is a guard rail, not a transformer. It raises on hard
    violations and logs warnings on soft violations. The caller is
    responsible for fixing the data before calling this function.

    Parameters
    ----------
    df:
        The DataFrame to validate. Must satisfy all invariants listed below.
    source_name:
        Human-readable label for the source (e.g. ``"OMNI"``, ``"GOES-16"``).
        Used in log messages and exception text.

    Raises
    ------
    KeyError
        If the ``timestamp`` column is absent.
    TypeError
        If ``df["timestamp"]`` is not ``datetime64[ns, UTC]``.
    ValueError
        If duplicate timestamps are detected.

    Warnings
    --------
    Logs a WARNING for any column whose NaN fraction exceeds 50 %.
    Logs a WARNING for any column that is entirely NaN.

    Notes
    -----
    - The NaN fraction check is informational. It does NOT raise.
    - Duplicate detection uses ``pd.Series.duplicated(keep=False)``, which
      flags ALL rows involved in a duplicate pair (not just the second).

    Examples
    --------
    >>> import pandas as pd
    >>> from schema import validate_output_schema
    >>> df = pd.DataFrame({
    ...     "timestamp": pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC"),
    ...     "goes_bz_gsm": [-5.0, -7.0, -9.0],
    ... })
    >>> validate_output_schema(df, "GOES-15")   # passes silently
    """
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Invariant 1: 'timestamp' column must exist with this exact name.
    # ------------------------------------------------------------------
    if "timestamp" not in df.columns:
        present = list(df.columns)
        raise KeyError(
            f"[{source_name}] Missing required column 'timestamp'. "
            f"Present columns: {present}. "
            "Rename the time column to 'timestamp' before writing."
        )

    # ------------------------------------------------------------------
    # Invariant 2: timestamp must be timezone-aware UTC.
    # ------------------------------------------------------------------
    ts_dtype = df["timestamp"].dtype
    is_utc = (
        hasattr(ts_dtype, "tz")
        and ts_dtype.tz is not None
        and str(ts_dtype.tz) in ("UTC", "utc")
    )
    if not is_utc:
        raise TypeError(
            f"[{source_name}] df['timestamp'] must be datetime64[ns, UTC]. "
            f"Found dtype: {ts_dtype!r}. "
            "Use pd.to_datetime(col, utc=True) to convert."
        )

    # ------------------------------------------------------------------
    # Invariant 3: no duplicate rows across uniqueness subset
    # ------------------------------------------------------------------
    subset = unique_subset if unique_subset is not None else ["timestamp"]
    
    missing_cols = [c for c in subset if c not in df.columns]
    if missing_cols:
        raise KeyError(f"[{source_name}] Unique subset columns missing: {missing_cols}")

    dups = df.duplicated(subset=subset, keep=False)
    if dups.any():
        n_dup = int(dups.sum())
        dup_examples = df.loc[dups, subset].head(5).to_dict(orient="records")
        raise ValueError(
            f"[{source_name}] {n_dup} duplicated rows found for keys {subset}. "
            f"Examples: {dup_examples}. "
            "Remove duplicates before writing (e.g. drop_duplicates or resample)."
        )

    # ------------------------------------------------------------------
    # Soft check: per-column NaN fraction.
    # ------------------------------------------------------------------
    for col in df.columns:
        if col == "timestamp":
            continue
        null_count = int(df[col].isna().sum())
        if n_rows == 0:
            continue
        frac = null_count / n_rows
        if frac == 1.0:
            log.warning(
                "[%s] Column '%s' is entirely NaN (%d/%d rows). "
                "Possible failed join or empty retrieval.",
                source_name, col, null_count, n_rows,
            )
        elif frac > 0.5:
            log.warning(
                "[%s] Column '%s' has %.1f%% NaN (%d/%d rows).",
                source_name, col, frac * 100, null_count, n_rows,
            )

    # ------------------------------------------------------------------
    # Summary log.
    # ------------------------------------------------------------------
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    nan_fracs = {
        c: f"{df[c].isna().mean():.2%}"
        for c in df.columns
        if c != "timestamp" and df[c].isna().any()
    }
    log.info(
        "[%s] Schema OK | rows=%d | timestamp=[%s, %s] | nan_cols=%s",
        source_name,
        n_rows,
        ts_min,
        ts_max,
        nan_fracs if nan_fracs else "none",
    )


# ---------------------------------------------------------------------------
# P0-F2: Physical plausibility (per-source, finite values only)
# ---------------------------------------------------------------------------

_BZ_ABS_MAX_NT = 100.0
_VSW_MIN_KMS = 200.0
_VSW_MAX_KMS = 2000.0
_DBDT_HORIZONTAL_MAX_NT_PER_MIN = 10000.0


def _first_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _solar_wind_speed_kms(df: pd.DataFrame) -> pd.Series | None:
    """OMNI HRO2: use vector magnitude if available, else |Vx| (proxy for speed)."""
    vx = _first_col(df, ("Vx", "omni_vx"))
    vy = _first_col(df, ("Vy", "omni_vy"))
    vz = _first_col(df, ("Vz", "omni_vz"))
    if vx and vy and vz:
        a = pd.to_numeric(df[vx], errors="coerce")
        b = pd.to_numeric(df[vy], errors="coerce")
        c = pd.to_numeric(df[vz], errors="coerce")
        return np.sqrt(a * a + b * b + c * c)
    if vx:
        return pd.to_numeric(df[vx], errors="coerce").abs()
    return None


def validate_physical_ranges(df: pd.DataFrame, dataset: str) -> None:
    """Check physical bounds; raises ``ValueError`` on violation (informative text).

    OMNI/GOES: ``|Bz| < 100`` nT; solar wind speed ``200--2000`` km/s (magnitude).
    SuperMAG: horizontal dB/dt proxy columns ``< 10000`` nT/min.
    """
    if df is None or df.empty:
        return
    errors: list[str] = []

    if dataset == "omni":
        bz_col = _first_col(df, ("BZ_GSM", "omni_bz_gsm"))
        if bz_col:
            bz = pd.to_numeric(df[bz_col], errors="coerce")
            bad = bz.notna() & (bz.abs() >= _BZ_ABS_MAX_NT)
            n_bad = int(bad.sum())
            if n_bad:
                errors.append(
                    f"OMNI Bz: {n_bad} values have |Bz| >= {_BZ_ABS_MAX_NT} nT in '{bz_col}' (non-NaN)."
                )
        vsw = _solar_wind_speed_kms(df)
        if vsw is not None:
            bad = vsw.notna() & ((vsw < _VSW_MIN_KMS) | (vsw > _VSW_MAX_KMS))
            n_bad = int(bad.sum())
            if n_bad:
                errors.append(
                    f"OMNI Vsw: {n_bad} values outside ({_VSW_MIN_KMS}, {_VSW_MAX_KMS}) km/s."
                )

    elif dataset == "goes_mag":
        bz_col = _first_col(df, ("goes_bz_gsm", "B_Z_GSM", "BZ_GSM", "Bz_GSM"))
        if bz_col:
            bz = pd.to_numeric(df[bz_col], errors="coerce")
            bad = bz.notna() & (bz.abs() >= _BZ_ABS_MAX_NT)
            n_bad = int(bad.sum())
            if n_bad:
                errors.append(
                    f"GOES Bz: {n_bad} values have |Bz| >= {_BZ_ABS_MAX_NT} nT in '{bz_col}'."
                )

    elif dataset == "supermag":
        if "dbdt_n" in df.columns and "dbdt_e" in df.columns:
            dn = pd.to_numeric(df["dbdt_n"], errors="coerce")
            de = pd.to_numeric(df["dbdt_e"], errors="coerce")
            mag = (dn * dn + de * de) ** 0.5
            bad = mag.notna() & (mag >= _DBDT_HORIZONTAL_MAX_NT_PER_MIN)
            n_bad = int(bad.sum())
            if n_bad:
                errors.append(
                    f"SuperMAG: {n_bad} rows have horizontal sqrt(dbdt_n^2+dbdt_e^2) >= "
                    f"{_DBDT_HORIZONTAL_MAX_NT_PER_MIN} nT/min."
                )
        else:
            horiz_cols = [
                c for c in df.columns
                if c == "dbdt_horizontal_magnitude" or c.startswith("dbdt_horizontal_magnitude_")
            ]
            for col in horiz_cols:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = s.notna() & (s.abs() >= _DBDT_HORIZONTAL_MAX_NT_PER_MIN)
                n_bad = int(bad.sum())
                if n_bad:
                    errors.append(
                        f"SuperMAG: {n_bad} values in '{col}' have |dB/dt| >= "
                        f"{_DBDT_HORIZONTAL_MAX_NT_PER_MIN} nT/min."
                    )
            if "dbdt_magnitude" in df.columns and not horiz_cols:
                s = pd.to_numeric(df["dbdt_magnitude"], errors="coerce")
                bad = s.notna() & (s.abs() >= _DBDT_HORIZONTAL_MAX_NT_PER_MIN)
                n_bad = int(bad.sum())
                if n_bad:
                    errors.append(
                        f"SuperMAG: {n_bad} values in 'dbdt_magnitude' exceed "
                        f"{_DBDT_HORIZONTAL_MAX_NT_PER_MIN} nT/min."
                    )

    if errors:
        raise ValueError(
            f"[{dataset}] Physical range validation failed (finite samples):\n" + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# P0-A1: Monthly source and feature-matrix validation
# ---------------------------------------------------------------------------

_SOURCE_CANDIDATES: dict[str, list[tuple[str, list[str], list[str]]]] = {
    "omni": [
        (
            "data/processed/omni/{year:04d}/{month:02d}/omni_{yyyymm}.parquet",
            ["timestamp"],
            ["BZ_GSM", "BY_GSM", "Vx"],
        ),
        (
            "data/raw/omni/{year:04d}/{month:02d}/omni_{yyyymm}.parquet",
            ["timestamp"],
            ["BZ_GSM", "BY_GSM", "Vx"],
        ),
    ],
    "goes_mag": [
        (
            "data/raw/goes/goes_mag_{yyyymm}.parquet",
            ["timestamp", "goes_bz_gsm"],
            ["goes_source_satellite", "goes_mag_missing_flag"],
        ),
        (
            "data/processed/aligned_1min/{year:04d}/{month:02d}/goes15_1min_{yyyymm}.parquet",
            ["timestamp"],
            ["goes_bz_gsm", "B_Z_GSM", "BZ_GSM", "Bz_GSM"],
        ),
        (
            "data/processed/goes/{year:04d}/{month:02d}/goes_{yyyymm}.parquet",
            ["timestamp"],
            ["goes_bz_gsm", "B_Z_GSM", "BZ_GSM", "Bz_GSM"],
        ),
    ],
    "swarm": [
        (
            "data/raw/swarm/{year:04d}/{month:02d}/swarmA_LR1B_{yyyymm}.parquet",
            ["timestamp"],
            ["B_NEC", "F", "QDLat"],
        ),
        (
            "data/raw/swarm/{year:04d}/{month:02d}/swarmB_LR1B_{yyyymm}.parquet",
            ["timestamp"],
            ["B_NEC", "F", "QDLat"],
        ),
        (
            "data/raw/swarm/{year:04d}/{month:02d}/swarmC_LR1B_{yyyymm}.parquet",
            ["timestamp"],
            ["B_NEC", "F", "QDLat"],
        ),
    ],
    "leo_index": [
        (
            "data/processed/swarm/{year:04d}/{month:02d}/swarm_leo_index_{yyyymm}.parquet",
            ["timestamp"],
            ["leo_index_global", "leo_high_lat"],
        ),
        (
            "data/processed/aligned_1min/{year:04d}/{month:02d}/leo_index_global_{yyyymm}.parquet",
            ["timestamp"],
            ["leo_index_global", "leo_high_lat"],
        ),
    ],
    "supermag": [
        (
            "data/raw/supermag/{year:04d}/{month:02d}/supermag_{yyyymm}.parquet",
            ["timestamp", "station"],
            ["n_nez", "e_nez", "z_nez"],
        ),
        (
            "data/processed/supermag/{year:04d}/{month:02d}/supermag_{yyyymm}.parquet",
            ["timestamp", "station"],
            ["dbdt_horizontal_magnitude", "dbdt_magnitude"],
        ),
    ],
}

_FEATURE_REQUIRED_COLUMNS = ["timestamp", "ut_sin", "ut_cos", "doy_sin", "doy_cos"]
_FEATURE_CORE_ANY_COLUMNS = {
    "omni_bz_gsm": ["omni_bz_gsm", "BZ_GSM"],
    "omni_by_gsm": ["omni_by_gsm", "BY_GSM"],
    "omni_vx": ["omni_vx", "Vx"],
    "newell_phi": ["newell_phi"],
}

_COMPLETENESS_PRIMARY_COLUMNS: dict[str, list[str]] = {
    "omni": ["BZ_GSM", "BY_GSM", "Vx", "omni_bz_gsm", "omni_by_gsm", "omni_vx"],
    "goes_mag": ["goes_bz_gsm", "B_Z_GSM", "BZ_GSM", "Bz_GSM"],
    "goes_xray": ["xrsa_flux", "xrsb_flux"],
    "swarm": ["B_NEC", "F", "QDLat"],
    "leo_index": ["leo_index_global", "leo_high_lat"],
    "supermag": ["n_nez", "e_nez", "z_nez", "dbdt_horizontal_magnitude", "dbdt_magnitude"],
}


def _yyyymm(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"


def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year, month, 1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    else:
        end = pd.Timestamp(year, month + 1, 1, tz="UTC")
    return start, end


def _expected_month_rows(year: int, month: int) -> int:
    start, end = _month_bounds(year, month)
    return int((end - start).total_seconds() // 60)


def _format_path(template: str, year: int, month: int) -> Path:
    return Path(template.format(year=year, month=month, yyyymm=_yyyymm(year, month)))


def _first_existing_candidate(
    candidates: list[tuple[str, list[str], list[str]]],
    year: int,
    month: int,
) -> tuple[Path | None, list[str], list[str], list[Path]]:
    checked: list[Path] = []
    for template, required, any_of in candidates:
        path = _format_path(template, year, month)
        checked.append(path)
        if path.exists():
            return path, required, any_of, checked
    required = candidates[0][1] if candidates else []
    any_of = candidates[0][2] if candidates else []
    return None, required, any_of, checked


def _read_parquet(path: Path, dataset: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"{dataset}: could not read {path}: {exc}") from exc

    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _has_any_column(df: pd.DataFrame, candidates: list[str]) -> bool:
    return any(col in df.columns for col in candidates)


def _validate_time_coverage(
    df: pd.DataFrame,
    dataset: str,
    year: int,
    month: int,
    warnings_list: list[str],
) -> None:
    if "timestamp" not in df.columns or df.empty:
        return

    start, end = _month_bounds(year, month)
    ts = df["timestamp"].dropna()
    outside = int(((ts < start) | (ts >= end)).sum())
    if outside:
        warnings_list.append(f"{dataset}: {outside} rows outside {year:04d}-{month:02d}")

    if dataset.startswith("swarm"):
        return

    row_basis = int(df["timestamp"].nunique()) if "station" in df.columns else len(df)
    if row_basis != _expected_month_rows(year, month):
        warnings_list.append(
            f"{dataset}: timestamp count {row_basis} differs from expected "
            f"{_expected_month_rows(year, month)} one-minute rows"
        )


def _validate_dataset_columns(
    df: pd.DataFrame,
    dataset: str,
    required: list[str],
    any_of: list[str],
) -> list[str]:
    errors: list[str] = []
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        errors.append(f"{dataset}: missing required columns {missing_required}")

    if any_of and not _has_any_column(df, any_of):
        errors.append(f"{dataset}: expected at least one of {any_of}")
    return errors


def _finish_validation(errors: list[str], warnings_list: list[str], fail_on_error: bool) -> bool:
    for warning in warnings_list:
        log.warning(warning)
    if errors:
        message = "Validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        if fail_on_error:
            raise ValueError(message)
        log.error(message)
        return False
    return True


def validate_sources(year: int, month: int, fail_on_error: bool = True) -> bool:
    """Validate monthly source products before feature construction.

    The validator is deliberately structural: it checks loadability, timestamp
    schema, duplicate keys, month coverage, and source-specific core columns.
    It does not mutate files or fill missing data.

    Parameters
    ----------
    year, month:
        Calendar month to validate.
    fail_on_error:
        Raise on hard failures when True; otherwise log and return False.

    Returns
    -------
    bool
        True when all required source checks pass, False otherwise.
    """
    month_str = _yyyymm(year, month)
    errors: list[str] = []
    warnings_list: list[str] = []

    log.info("Validating source products for %s...", month_str)

    for dataset, candidates in _SOURCE_CANDIDATES.items():
        if dataset == "swarm":
            # Swarm has one required file per satellite.
            for template, required, any_of in candidates:
                path = _format_path(template, year, month)
                satellite_label = path.stem.split("_")[0]
                label = f"swarm:{satellite_label}"
                if not path.exists():
                    errors.append(f"{label}: missing file {path}")
                    continue
                try:
                    df = _read_parquet(path, label)
                    validate_output_schema(df, label)
                    if dataset in ("omni", "goes_mag", "supermag"):
                        validate_physical_ranges(df, dataset)
                    errors.extend(_validate_dataset_columns(df, label, required, any_of))
                    _validate_time_coverage(df, label, year, month, warnings_list)
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
            continue

        path, required, any_of, checked = _first_existing_candidate(candidates, year, month)
        if path is None:
            errors.append(
                f"{dataset}: missing file; checked "
                + ", ".join(str(candidate) for candidate in checked)
            )
            continue

        try:
            df = _read_parquet(path, dataset)
            unique_subset = ["timestamp", "station"] if "station" in required else None
            validate_output_schema(df, dataset, unique_subset=unique_subset)
            if dataset in ("omni", "goes_mag", "supermag"):
                validate_physical_ranges(df, dataset)
            errors.extend(_validate_dataset_columns(df, dataset, required, any_of))
            _validate_time_coverage(df, dataset, year, month, warnings_list)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")

    return _finish_validation(errors, warnings_list, fail_on_error)


def _feature_path(year: int, month: int) -> Path:
    month_str = _yyyymm(year, month)
    return Path("data") / "processed" / "features" / f"{year:04d}" / f"{month:02d}" / f"features_{month_str}.parquet"


def _validate_cyclical_bounds(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in ["ut_sin", "ut_cos", "doy_sin", "doy_cos"]:
        if col not in df.columns:
            continue
        finite = df[col].dropna()
        if finite.empty:
            continue
        if not finite.between(-1.000001, 1.000001).all():
            errors.append(f"features: {col} contains values outside [-1, 1]")
    return errors


def _feature_target_columns(df: pd.DataFrame) -> list[str]:
    canonical = [
        col for col in df.columns
        if col.startswith("dbdt_horizontal_magnitude_")
    ]
    legacy = [
        col for col in df.columns
        if col.endswith("_dbdt_magnitude") or col.endswith("_dbdt_horizontal_magnitude")
    ]
    return canonical + legacy


def validate_feature_matrix(year: int, month: int, fail_on_error: bool = True) -> bool:
    """Validate the fused monthly feature matrix.

    Checks the structural schema, month-aligned one-minute cadence, required
    temporal encodings, core driver feature presence, cyclical bounds, and
    existence of at least one station target column once target integration is
    available.
    """
    month_str = _yyyymm(year, month)
    path = _feature_path(year, month)
    errors: list[str] = []
    warnings_list: list[str] = []

    log.info("Validating feature matrix for %s...", month_str)

    if not path.exists():
        errors.append(f"features: missing file {path}")
        return _finish_validation(errors, warnings_list, fail_on_error)

    try:
        df = _read_parquet(path, "features")
        validate_output_schema(df, f"features-{month_str}")
    except Exception as exc:
        errors.append(f"features: {exc}")
        return _finish_validation(errors, warnings_list, fail_on_error)

    errors.extend(_validate_dataset_columns(df, "features", _FEATURE_REQUIRED_COLUMNS, []))
    for semantic_name, candidates in _FEATURE_CORE_ANY_COLUMNS.items():
        if not _has_any_column(df, candidates):
            errors.append(f"features: missing core feature {semantic_name}; expected one of {candidates}")

    if "timestamp" in df.columns:
        start, end = _month_bounds(year, month)
        ts = df["timestamp"].dropna()
        if int(ts.duplicated().sum()) > 0:
            errors.append("features: duplicate timestamps present")
        if not ts.is_monotonic_increasing:
            warnings_list.append("features: timestamps are not monotonic increasing")
        expected_rows = _expected_month_rows(year, month)
        if len(df) != expected_rows:
            warnings_list.append(
                f"features: row count {len(df)} differs from expected {expected_rows}"
            )
        if len(ts) > 0:
            if ts.min() != start:
                warnings_list.append(f"features: first timestamp {ts.min()} != expected {start}")
            if ts.max() != end - pd.Timedelta(minutes=1):
                warnings_list.append(
                    f"features: last timestamp {ts.max()} != expected {end - pd.Timedelta(minutes=1)}"
                )

    errors.extend(_validate_cyclical_bounds(df))

    target_cols = _feature_target_columns(df)
    if not target_cols:
        warnings_list.append(
            "features: no multi-station dB/dt target columns found yet; "
            "expected after P0-S5"
        )
    else:
        all_nan_targets = [col for col in target_cols if df[col].isna().all()]
        if all_nan_targets:
            warnings_list.append(
                "features: all-NaN station target columns: "
                + ", ".join(all_nan_targets[:10])
                + ("..." if len(all_nan_targets) > 10 else "")
            )

    return _finish_validation(errors, warnings_list, fail_on_error)


# ---------------------------------------------------------------------------
# P0-A4: Per-month data completeness reporting
# ---------------------------------------------------------------------------

def _validation_report_dir(year: int, month: int) -> Path:
    return Path("results") / "validation" / f"{year:04d}" / f"{month:02d}"


def _expected_minute_index(year: int, month: int) -> pd.DatetimeIndex:
    start, end = _month_bounds(year, month)
    return pd.date_range(start=start, end=end, freq="1min", inclusive="left", tz="UTC")


def _timestamp_gap_ranges(
    timestamps: pd.Series,
    year: int,
    month: int,
    max_ranges: int = 25,
) -> list[dict[str, str | int]]:
    """Return compact missing 1-minute timestamp ranges for a month."""
    expected = _expected_minute_index(year, month)
    if timestamps.empty:
        missing = expected
    else:
        observed = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().unique())
        observed = observed[(observed >= expected[0]) & (observed <= expected[-1])]
        missing = expected.difference(observed)

    if len(missing) == 0:
        return []

    ranges: list[dict[str, str | int]] = []
    start = missing[0]
    previous = missing[0]
    for ts in missing[1:]:
        if ts - previous == pd.Timedelta(minutes=1):
            previous = ts
            continue
        ranges.append({
            "start": start.isoformat(),
            "end": previous.isoformat(),
            "minutes": int((previous - start).total_seconds() // 60) + 1,
        })
        if len(ranges) >= max_ranges:
            break
        start = ts
        previous = ts

    if len(ranges) < max_ranges:
        ranges.append({
            "start": start.isoformat(),
            "end": previous.isoformat(),
            "minutes": int((previous - start).total_seconds() // 60) + 1,
        })

    return ranges


def _selected_primary_columns(df: pd.DataFrame, dataset: str) -> list[str]:
    return [col for col in _COMPLETENESS_PRIMARY_COLUMNS.get(dataset, []) if col in df.columns]


def _valid_fraction(df: pd.DataFrame, columns: list[str]) -> float | None:
    if df.empty or not columns:
        return None
    return float(df[columns].notna().all(axis=1).mean())


def _source_completeness_summary(
    dataset: str,
    path: Path | None,
    year: int,
    month: int,
    required: list[str] | None = None,
    any_of: list[str] | None = None,
    unique_subset: list[str] | None = None,
) -> dict:
    """Build a completeness summary for one monthly source file."""
    summary = {
        "dataset": dataset,
        "path": str(path) if path is not None else None,
        "available": False,
        "schema_ok": False,
        "rows": 0,
        "unique_timestamps": 0,
        "expected_minutes": _expected_month_rows(year, month),
        "timestamp_coverage_fraction": 0.0,
        "valid_fraction": None,
        "primary_columns": [],
        "gap_locations": [],
        "errors": [],
        "warnings": [],
    }

    if path is None or not path.exists():
        summary["errors"].append("missing_file")
        return summary

    summary["available"] = True
    try:
        df = _read_parquet(path, dataset)
    except Exception as exc:
        summary["errors"].append(str(exc))
        return summary

    summary["rows"] = int(len(df))

    try:
        validate_output_schema(df, dataset, unique_subset=unique_subset)
        summary["schema_ok"] = True
    except Exception as exc:
        summary["errors"].append(str(exc))

    column_errors = _validate_dataset_columns(df, dataset, required or [], any_of or [])
    summary["errors"].extend(column_errors)

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        expected = _expected_month_rows(year, month)
        unique_timestamps = int(timestamps.nunique())
        summary["unique_timestamps"] = unique_timestamps
        summary["timestamp_coverage_fraction"] = float(min(unique_timestamps / expected, 1.0)) if expected else 0.0
        summary["gap_locations"] = _timestamp_gap_ranges(timestamps, year, month)

    primary_columns = _selected_primary_columns(df, dataset)
    if any_of:
        primary_columns = [col for col in any_of if col in df.columns] or primary_columns
    summary["primary_columns"] = primary_columns
    summary["valid_fraction"] = _valid_fraction(df, primary_columns)

    if summary["valid_fraction"] is None:
        summary["warnings"].append("no_primary_columns_for_valid_fraction")

    return summary


def _load_cached_supermag_inventory(year: int, month: int) -> dict:
    month_str = _yyyymm(year, month)
    path = Path("data") / "external" / "station_metadata" / f"supermag_inventory_{month_str}.json"
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "station_count": 0,
            "stations": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as fh:
            inventory = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "path": str(path),
            "station_count": 0,
            "stations": [],
            "error": str(exc),
        }

    stations = [str(station).upper() for station in inventory.get("stations", [])]
    return {
        "available": True,
        "path": str(path),
        "station_count": int(inventory.get("station_count", len(stations))),
        "stations": stations,
        "query_timestamp": inventory.get("query_timestamp"),
    }


def _supermag_station_completeness(
    path: Path | None,
    year: int,
    month: int,
) -> dict:
    result = {
        "available": False,
        "station_count": 0,
        "stations": {},
        "lowest_completeness": [],
        "errors": [],
    }
    if path is None or not path.exists():
        result["errors"].append("missing_supermag_file")
        return result

    try:
        df = _read_parquet(path, "supermag")
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    if "station" not in df.columns or "timestamp" not in df.columns:
        result["errors"].append("missing_station_or_timestamp_column")
        return result

    expected = _expected_month_rows(year, month)
    primary_cols = _selected_primary_columns(df, "supermag")
    station_summaries: dict[str, dict] = {}

    for station, group in df.groupby("station", sort=True):
        timestamps = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").dropna()
        unique_timestamps = int(timestamps.nunique())
        station_primary = [col for col in primary_cols if col in group.columns]
        station_summaries[str(station)] = {
            "rows": int(len(group)),
            "unique_timestamps": unique_timestamps,
            "timestamp_coverage_fraction": float(min(unique_timestamps / expected, 1.0)) if expected else 0.0,
            "valid_fraction": _valid_fraction(group, station_primary),
            "gap_locations": _timestamp_gap_ranges(timestamps, year, month, max_ranges=10),
        }

    result["available"] = True
    result["station_count"] = len(station_summaries)
    result["stations"] = station_summaries
    result["lowest_completeness"] = sorted(
        (
            {
                "station": station,
                "timestamp_coverage_fraction": summary["timestamp_coverage_fraction"],
                "valid_fraction": summary["valid_fraction"],
            }
            for station, summary in station_summaries.items()
        ),
        key=lambda item: (
            item["timestamp_coverage_fraction"],
            -1.0 if item["valid_fraction"] is None else item["valid_fraction"],
        ),
    )[:20]
    return result


def _satellite_availability(year: int, month: int) -> dict:
    month_str = _yyyymm(year, month)
    goes_dir = Path("data") / "raw" / "goes"
    swarm_dir = Path("data") / "raw" / "swarm" / f"{year:04d}" / f"{month:02d}"

    goes_files = sorted(goes_dir.glob(f"**/*{month_str}*.parquet")) if goes_dir.exists() else []
    swarm_files = sorted(swarm_dir.glob(f"swarm*_LR1B_{month_str}.parquet")) if swarm_dir.exists() else []

    goes_satellites = sorted({
        token.upper().replace("GOES", "GOES-")
        for path in goes_files
        for token in path.stem.replace("-", "").split("_")
        if token.lower().startswith("goes") and token.lower() not in {"goes", "goesmag"}
    })
    swarm_satellites = sorted({
        path.stem.split("_")[0].replace("swarm", "Swarm-")
        for path in swarm_files
    })

    return {
        "goes": {
            "available_file_count": len(goes_files),
            "satellites": goes_satellites,
            "files": [str(path) for path in goes_files],
        },
        "swarm": {
            "available_file_count": len(swarm_files),
            "satellites": swarm_satellites,
            "files": [str(path) for path in swarm_files],
        },
    }


def generate_completeness_report(year: int, month: int) -> dict:
    """Generate and write a monthly data completeness report.

    The report is observational only: it reads existing source files and cached
    inventory metadata, computes completeness diagnostics, writes JSON under
    ``results/validation/YYYY/MM/``, and returns the same dictionary.
    """
    month_str = _yyyymm(year, month)
    log.info("Generating completeness report for %s...", month_str)

    sources: dict[str, dict] = {}

    for dataset, candidates in _SOURCE_CANDIDATES.items():
        if dataset == "swarm":
            satellite_summaries = {}
            for template, required, any_of in candidates:
                path = _format_path(template, year, month)
                label = path.stem.split("_")[0]
                satellite_summaries[label] = _source_completeness_summary(
                    "swarm",
                    path,
                    year,
                    month,
                    required=required,
                    any_of=any_of,
                )
            sources["swarm"] = {
                "available": any(summary["available"] for summary in satellite_summaries.values()),
                "satellites": satellite_summaries,
            }
            continue

        path, required, any_of, checked = _first_existing_candidate(candidates, year, month)
        unique_subset = ["timestamp", "station"] if "station" in required else None
        summary = _source_completeness_summary(
            dataset,
            path,
            year,
            month,
            required=required,
            any_of=any_of,
            unique_subset=unique_subset,
        )
        if path is None:
            summary["checked_paths"] = [str(candidate) for candidate in checked]
        sources[dataset] = summary

    supermag_path = Path(sources["supermag"]["path"]) if sources.get("supermag", {}).get("path") else None
    inventory = _load_cached_supermag_inventory(year, month)
    station_completeness = _supermag_station_completeness(supermag_path, year, month)

    report = {
        "year": year,
        "month": month,
        "month_str": month_str,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "expected_minutes": _expected_month_rows(year, month),
        "sources": sources,
        "supermag_inventory": inventory,
        "station_count": {
            "inventory": inventory.get("station_count", 0),
            "retrieved": station_completeness.get("station_count", 0),
        },
        "per_station_completeness": station_completeness,
        "satellite_availability": _satellite_availability(year, month),
    }

    out_dir = _validation_report_dir(year, month)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"completeness_report_{month_str}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    log.info("Completeness report saved to %s", output_path)
    return report
