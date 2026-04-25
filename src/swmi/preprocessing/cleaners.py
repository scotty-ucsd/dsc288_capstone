"""
cleaners.py
Gap-aware dB/dt computation and X-ray normalization for the SWMI pipeline.

Key functions:
    compute_dbdt_gap_aware   — single-station gap-aware dB/dt from NEZ components
    compute_all_station_dbdt — batch wrapper over all stations in long-format data
    normalize_goes_xray      — cross-satellite XRS log/baseline normalization

Scientific contract:
    - DBDT_METHOD = "backward" (no future leakage)
    - dB/dt uses ACTUAL elapsed time between samples, not a constant 60 s
    - Gaps > gap_threshold_sec produce NaN derivatives with dbdt_gap_flag = 1
    - dbdt_horizontal_magnitude = sqrt(dbdt_n² + dbdt_e²)
    - Coordinates must be NEZ; .mag and .geo are forbidden (Warning #16, #17)

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from swmi.preprocessing.validation import validate_output_schema
from swmi.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default gap threshold: 90 seconds = 1.5 × expected 60 s cadence.
# Tolerates minor timestamp jitter (±5 s rounding in some SuperMAG records)
# but correctly identifies genuine data gaps.
# ---------------------------------------------------------------------------
_DEFAULT_GAP_THRESHOLD_SEC = 90.0
_XRSA_VALID_RANGE = (1e-9, 3e-3)
_XRSB_VALID_RANGE = (1e-9, 2e-1)
_DEFAULT_XRAY_BASELINE_QUANTILE = 0.10

# NOAA cross-version factors are applied in log space as log10(factor). The
# current science-quality v2-2-1 products are already continuity-corrected for
# the study satellites, so defaults are identity but callers can override them
# when NOAA publishes satellite-specific continuity factors.
_DEFAULT_XRAY_SCALE_FACTORS = {
    "GOES-13": 1.0,
    "GOES-14": 1.0,
    "GOES-15": 1.0,
    "GOES-16": 1.0,
    "GOES-17": 1.0,
    "GOES-18": 1.0,
    "GOES-19": 1.0,
}


def _load_goes_xray_input(
    xray_data: str | os.PathLike | pd.DataFrame,
) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(xray_data, pd.DataFrame):
        return xray_data.copy(), None

    path = Path(xray_data)
    if not path.exists():
        raise FileNotFoundError(f"GOES X-ray parquet not found: {path}")
    return pd.read_parquet(path), path


def _infer_xray_output_path(raw_path: Path) -> Path:
    month_str = raw_path.stem.rsplit("_", maxsplit=1)[-1]
    if len(month_str) != 6 or not month_str.isdigit():
        raise ValueError(
            f"Cannot infer year/month from GOES X-ray path: {raw_path}. "
            "Pass output_path explicitly."
        )
    year = month_str[:4]
    month = month_str[4:6]
    return Path("data") / "processed" / "goes" / year / month / f"goes_xray_normalized_{month_str}.parquet"


def _infer_xray_plot_path(raw_path: Path | None, output_path: Path | None) -> Path | None:
    stem_source = output_path or raw_path
    if stem_source is None:
        return None
    month_str = stem_source.stem.rsplit("_", maxsplit=1)[-1]
    if len(month_str) != 6 or not month_str.isdigit():
        month_str = stem_source.stem
    return Path("results") / "figures" / f"goes_xray_normalization_{month_str}.png"


def _quality_mask_from_summary(flags: pd.Series, channel: str) -> pd.Series:
    """Return channel-valid mask from canonical retriever quality strings."""
    if flags is None:
        return pd.Series(True)

    text = flags.fillna("").astype(str)
    channel_key = f"{channel}="
    channel_ok = ~text.str.contains(fr"{channel_key}(?!0(?:;|$))\d+", regex=True)

    electron_values = text.str.extract(r"electron=(\d+)")[0]
    electron = pd.to_numeric(electron_values, errors="coerce").fillna(0).astype("int64")
    electron_bad = electron.isin((1, 4))
    for bit in (8, 16, 32, 64, 128, 256):
        electron_bad |= (electron & bit) != 0

    return channel_ok & ~electron_bad


def _valid_xray_flux(flux: pd.Series, valid_range: tuple[float, float]) -> pd.Series:
    valid_min, valid_max = valid_range
    return pd.to_numeric(flux, errors="coerce").between(valid_min, valid_max)


def _satellite_baselines(
    df: pd.DataFrame,
    log_col: str,
    baseline_quantile: float,
) -> pd.Series:
    baselines = (
        df.groupby("xray_source_satellite", dropna=False)[log_col]
        .transform(lambda values: values.quantile(baseline_quantile))
    )
    global_baseline = df[log_col].quantile(baseline_quantile)
    return baselines.fillna(global_baseline)


def _scale_offsets(
    satellites: pd.Series,
    scale_factors: dict[str, float],
) -> pd.Series:
    factors = satellites.map(scale_factors).fillna(1.0).astype(float)
    if (factors <= 0).any():
        bad = sorted(satellites.loc[factors <= 0].dropna().unique())
        raise ValueError(f"GOES X-ray scale factors must be positive. Bad satellites: {bad}")
    return np.log10(factors)


def _write_xray_validation_plot(df: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is unavailable; skipping GOES X-ray validation plot.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_cols = [
        "goes_xray_short_normalized",
        "goes_xray_long_normalized",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in plot_cols:
        if col in df.columns:
            ax.plot(df["timestamp"], df[col], label=col, linewidth=0.8)
    ax.set_title("GOES X-ray cross-satellite normalized log flux")
    ax.set_xlabel("Time")
    ax.set_ylabel("log10 flux minus quiet baseline")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def normalize_goes_xray(
    xray_data: str | os.PathLike | pd.DataFrame,
    output_path: str | os.PathLike | None = None,
    *,
    write_output: bool | None = None,
    validation_plot_path: str | os.PathLike | None = None,
    write_validation_plot: bool = True,
    scale_factors: dict[str, float] | None = None,
    baseline_quantile: float = _DEFAULT_XRAY_BASELINE_QUANTILE,
) -> pd.DataFrame:
    """Normalize canonical GOES X-ray flux across satellites.

    Pipeline:
    1. Re-apply physical/quality filtering to canonical XRS flux.
    2. Compute log10 flux for both channels.
    3. Subtract a per-satellite quiet-Sun baseline.
    4. Apply NOAA continuity scale factors in log space.
    5. Optionally write a validation plot.
    """
    if not 0.0 <= baseline_quantile <= 1.0:
        raise ValueError(f"baseline_quantile must be in [0, 1], got {baseline_quantile}")

    df, raw_path = _load_goes_xray_input(xray_data)
    if write_output is None:
        write_output = raw_path is not None

    required = {
        "timestamp",
        "xrsa_flux",
        "xrsb_flux",
        "xray_source_satellite",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"GOES X-ray data missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("GOES X-ray data contains null or unparseable timestamps.")
    if out["timestamp"].duplicated().any():
        raise ValueError("GOES X-ray data contains duplicate timestamps.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    if "xray_quality_flags" not in out.columns:
        out["xray_quality_flags"] = pd.Series(pd.NA, index=out.index, dtype="string")

    short_quality = _quality_mask_from_summary(out["xray_quality_flags"], "a")
    long_quality = _quality_mask_from_summary(out["xray_quality_flags"], "b")

    short_valid = short_quality & _valid_xray_flux(out["xrsa_flux"], _XRSA_VALID_RANGE)
    long_valid = long_quality & _valid_xray_flux(out["xrsb_flux"], _XRSB_VALID_RANGE)

    out["xrsa_flux"] = pd.to_numeric(out["xrsa_flux"], errors="coerce").where(short_valid, np.nan)
    out["xrsb_flux"] = pd.to_numeric(out["xrsb_flux"], errors="coerce").where(long_valid, np.nan)
    out["goes_xray_short_log"] = np.log10(out["xrsa_flux"])
    out["goes_xray_long_log"] = np.log10(out["xrsb_flux"])

    out["goes_xray_short_baseline"] = _satellite_baselines(
        out,
        "goes_xray_short_log",
        baseline_quantile,
    )
    out["goes_xray_long_baseline"] = _satellite_baselines(
        out,
        "goes_xray_long_log",
        baseline_quantile,
    )

    factors = {**_DEFAULT_XRAY_SCALE_FACTORS, **(scale_factors or {})}
    offsets = _scale_offsets(out["xray_source_satellite"], factors)

    out["goes_xray_short_normalized"] = (
        out["goes_xray_short_log"] - out["goes_xray_short_baseline"] + offsets
    )
    out["goes_xray_long_normalized"] = (
        out["goes_xray_long_log"] - out["goes_xray_long_baseline"] + offsets
    )
    out["xray_normalized_missing_flag"] = (
        out["goes_xray_short_normalized"].isna()
        & out["goes_xray_long_normalized"].isna()
    ).astype("int8")

    if write_output:
        if output_path is None and raw_path is None:
            raise ValueError("output_path is required when write_output=True and xray_data is a DataFrame.")
        final_output_path = Path(output_path) if output_path is not None else _infer_xray_output_path(raw_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        validate_output_schema(out, f"GOES-xray-normalized-{final_output_path.stem}")
        out.to_parquet(final_output_path, index=False)
        log.info("Saved normalized GOES X-ray → %s (%d rows)", final_output_path, len(out))
    else:
        final_output_path = Path(output_path) if output_path is not None else None

    if write_validation_plot:
        plot_path = (
            Path(validation_plot_path)
            if validation_plot_path is not None
            else _infer_xray_plot_path(raw_path, final_output_path)
        )
        if plot_path is not None:
            _write_xray_validation_plot(out, plot_path)

    return out


def compute_dbdt_gap_aware(
    b_nez: pd.DataFrame,
    timestamps: pd.Series,
    gap_threshold_sec: float = _DEFAULT_GAP_THRESHOLD_SEC,
) -> pd.DataFrame:
    """Compute dB/dt components using backward difference with gap awareness.

    Unlike a naive ``diff() / 60.0``, this function:
    1. Uses **actual elapsed time** between consecutive samples.
    2. Masks derivatives where the time gap exceeds ``gap_threshold_sec``.
    3. Produces NaN (never zero-fill) when either component is missing.

    Parameters
    ----------
    b_nez : pd.DataFrame
        Must contain columns ``n_nez``, ``e_nez``, ``z_nez`` (float, nT).
        These are the SuperMAG NEZ magnetic field components with IGRF
        baseline removed.  Do **not** pass ``.mag`` or ``.geo`` columns.
    timestamps : pd.Series
        Timezone-aware ``datetime64[ns, UTC]`` timestamps aligned 1:1
        with the rows of ``b_nez``.  Must be sorted in ascending order.
    gap_threshold_sec : float, default 90.0
        Maximum allowable time interval (seconds) between consecutive
        samples.  Intervals exceeding this threshold are flagged as gaps
        and their dB/dt values are set to NaN.

    Returns
    -------
    pd.DataFrame
        Columns:
            - ``dbdt_n``   : dB_N/dt  (nT / min)
            - ``dbdt_e``   : dB_E/dt  (nT / min)
            - ``dbdt_z``   : dB_Z/dt  (nT / min)
            - ``dbdt_horizontal_magnitude`` : sqrt(dbdt_n² + dbdt_e²)
            - ``dbdt_gap_flag`` : int8, 1 where gap detected, 0 otherwise

    Raises
    ------
    KeyError
        If ``b_nez`` is missing any of the required NEZ columns.
    ValueError
        If ``timestamps`` contains timezone-naive datetimes.
        If ``timestamps`` length does not match ``b_nez`` row count.
        If ``gap_threshold_sec`` is not positive.

    Notes
    -----
    - The first row always produces NaN (no predecessor for backward diff).
    - ``dbdt_horizontal_magnitude`` is NaN if *either* ``dbdt_n`` or
      ``dbdt_e`` is NaN.  Zero is a valid quiet-time value and is never
      used as a fill sentinel.
    - Units: input B in nT, output dB/dt in nT/min.  The division by
      ``dt_seconds`` followed by multiplication by 60 converts from
      nT/s to nT/min.

    Complexity
    ----------
    Time:  $O(N)$ — single pass through the arrays.
    Space: $O(N)$ — output DataFrame same size as input.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts = pd.to_datetime(["2015-03-17 04:00", "2015-03-17 04:01",
    ...                      "2015-03-17 04:02"], utc=True)
    >>> b = pd.DataFrame({"n_nez": [100.0, 110.0, 105.0],
    ...                    "e_nez": [50.0, 52.0, 48.0],
    ...                    "z_nez": [30.0, 31.0, 29.0]})
    >>> result = compute_dbdt_gap_aware(b, ts)
    >>> assert np.isnan(result["dbdt_n"].iloc[0])          # first row
    >>> assert result["dbdt_n"].iloc[1] == 10.0            # (110-100)/60*60
    >>> assert result["dbdt_gap_flag"].iloc[0] == 0        # first row isn't a "gap"
    """
    # ------------------------------------------------------------------
    # Input validation — fail fast on contract violations
    # ------------------------------------------------------------------
    required_cols = {"n_nez", "e_nez", "z_nez"}
    missing = required_cols - set(b_nez.columns)
    if missing:
        raise KeyError(
            f"b_nez is missing required NEZ columns: {sorted(missing)}. "
            f"Present columns: {list(b_nez.columns)}. "
            "Ensure you are passing NEZ coordinates, not .mag or .geo."
        )

    if len(timestamps) != len(b_nez):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) does not match "
            f"b_nez row count ({len(b_nez)}). They must be aligned 1:1."
        )

    if gap_threshold_sec <= 0:
        raise ValueError(
            f"gap_threshold_sec must be positive, got {gap_threshold_sec}."
        )

    # Check timezone awareness
    if hasattr(timestamps.dtype, "tz") and timestamps.dtype.tz is None:
        raise ValueError(
            "timestamps must be timezone-aware (UTC).  "
            "Use pd.to_datetime(col, utc=True) to convert."
        )

    n_rows = len(b_nez)

    # ------------------------------------------------------------------
    # Edge case: 0 or 1 rows — return structurally valid empty/NaN result
    # ------------------------------------------------------------------
    if n_rows == 0:
        return pd.DataFrame({
            "dbdt_n": pd.Series([], dtype=float),
            "dbdt_e": pd.Series([], dtype=float),
            "dbdt_z": pd.Series([], dtype=float),
            "dbdt_horizontal_magnitude": pd.Series([], dtype=float),
            "dbdt_gap_flag": pd.Series([], dtype="int8"),
        })

    if n_rows == 1:
        return pd.DataFrame({
            "dbdt_n": [np.nan],
            "dbdt_e": [np.nan],
            "dbdt_z": [np.nan],
            "dbdt_horizontal_magnitude": [np.nan],
            "dbdt_gap_flag": np.array([0], dtype=np.int8),
        })

    # ------------------------------------------------------------------
    # Core computation — vectorized for O(N) performance
    # ------------------------------------------------------------------
    # Step 1: Actual elapsed time between consecutive samples (seconds)
    # timestamps may be a DatetimeIndex (whose .diff() → TimedeltaIndex, no .dt)
    # or a Series (whose .diff() → Series, has .dt). Convert uniformly.
    ts_series = pd.Series(timestamps.values, dtype="datetime64[ns, UTC]")
    dt_sec = ts_series.diff().dt.total_seconds().values  # float64 array, NaN at idx 0

    # Step 2: Backward differences of B-field components
    db_n = b_nez["n_nez"].diff().values
    db_e = b_nez["e_nez"].diff().values
    db_z = b_nez["z_nez"].diff().values

    # Step 3: dB/dt in nT/min  =  (delta_B / delta_t_seconds) * 60
    # Guard against division by zero (shouldn't happen with valid timestamps
    # but defensive programming requires it).
    with np.errstate(divide="ignore", invalid="ignore"):
        dbdt_n = np.where(dt_sec > 0, (db_n / dt_sec) * 60.0, np.nan)
        dbdt_e = np.where(dt_sec > 0, (db_e / dt_sec) * 60.0, np.nan)
        dbdt_z = np.where(dt_sec > 0, (db_z / dt_sec) * 60.0, np.nan)

    # Step 4: Gap masking — intervals > threshold produce NaN
    is_gap = np.zeros(n_rows, dtype=bool)
    is_gap[1:] = dt_sec[1:] > gap_threshold_sec  # idx 0 is NaN, never a gap

    dbdt_n[is_gap] = np.nan
    dbdt_e[is_gap] = np.nan
    dbdt_z[is_gap] = np.nan

    # Step 5: First row has no predecessor — always NaN
    dbdt_n[0] = np.nan
    dbdt_e[0] = np.nan
    dbdt_z[0] = np.nan

    # Step 6: Horizontal magnitude — NaN propagates naturally via sqrt
    # If either dbdt_n or dbdt_e is NaN, the magnitude must be NaN.
    dbdt_h_mag = np.sqrt(dbdt_n**2 + dbdt_e**2)

    # Step 7: Gap flag (int8) — 1 where gap was detected, 0 otherwise
    gap_flag = is_gap.astype(np.int8)

    # ------------------------------------------------------------------
    # Log diagnostics
    # ------------------------------------------------------------------
    n_gaps = int(is_gap.sum())
    n_valid = int(np.isfinite(dbdt_h_mag).sum())
    if n_gaps > 0:
        gap_pct = n_gaps / max(n_rows - 1, 1) * 100  # exclude first row
        log.info(
            "dB/dt gap-aware: %d gaps detected (%.1f%% of intervals), "
            "%d valid derivatives out of %d rows.",
            n_gaps, gap_pct, n_valid, n_rows,
        )

    return pd.DataFrame({
        "dbdt_n": dbdt_n,
        "dbdt_e": dbdt_e,
        "dbdt_z": dbdt_z,
        "dbdt_horizontal_magnitude": dbdt_h_mag,
        "dbdt_gap_flag": gap_flag,
    })


def _empty_all_station_dbdt() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
        "station": pd.Series([], dtype=str),
        "dbdt_n": pd.Series([], dtype=float),
        "dbdt_e": pd.Series([], dtype=float),
        "dbdt_z": pd.Series([], dtype=float),
        "dbdt_horizontal_magnitude": pd.Series([], dtype=float),
        "dbdt_gap_flag": pd.Series([], dtype="int8"),
    })


def _infer_dbdt_output_path(raw_path: Path) -> Path:
    month_str = raw_path.stem.rsplit("_", maxsplit=1)[-1]
    if len(month_str) != 6 or not month_str.isdigit():
        raise ValueError(
            f"Cannot infer year/month from raw SuperMAG path: {raw_path}. "
            "Pass output_path explicitly."
        )
    year = month_str[:4]
    month = month_str[4:6]
    return Path("data") / "interim" / "cleaned" / year / month / f"supermag_dbdt_{month_str}.parquet"


def _load_raw_supermag(raw_parquet: str | os.PathLike | pd.DataFrame) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(raw_parquet, pd.DataFrame):
        return raw_parquet.copy(), None

    raw_path = Path(raw_parquet)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw SuperMAG parquet not found: {raw_path}")
    return pd.read_parquet(raw_path), raw_path


def _resolve_dbdt_output_path(output_path: str | os.PathLike | None, raw_path: Path | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    if raw_path is None:
        raise ValueError("output_path is required when write_output=True and raw_parquet is a DataFrame.")
    return _infer_dbdt_output_path(raw_path)


def compute_all_station_dbdt(
    raw_parquet: str | os.PathLike | pd.DataFrame,
    gap_threshold_sec: float = _DEFAULT_GAP_THRESHOLD_SEC,
    output_path: str | os.PathLike | None = None,
    write_output: bool | None = None,
) -> pd.DataFrame:
    """Apply gap-aware dB/dt computation to all stations in long-format data.

    This is the batch wrapper for ``compute_dbdt_gap_aware()``.  It groups
    by station, applies the gap-aware derivative to each group independently,
    and returns a unified long-format DataFrame.

    Parameters
    ----------
    raw_parquet : str, Path, or pd.DataFrame
        Raw SuperMAG monthly Parquet path, or an already-loaded long-format
        DataFrame with columns:
        ``timestamp, station, n_nez, e_nez, z_nez`` (minimum).
        Must be sorted by ``[station, timestamp]`` or will be sorted internally.
    gap_threshold_sec : float, default 90.0
        Passed to ``compute_dbdt_gap_aware()``.
    output_path : str or Path, optional
        Destination for cleaned dB/dt Parquet. If omitted for path input,
        writes to ``data/interim/cleaned/YYYY/MM/supermag_dbdt_YYYYMM.parquet``.
    write_output : bool, optional
        Defaults to True for path input and False for DataFrame input.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp, station, dbdt_n, dbdt_e, dbdt_z,
        dbdt_horizontal_magnitude, dbdt_gap_flag``
        Sorted by ``[timestamp, station]``.

    Raises
    ------
    KeyError
        If the raw SuperMAG data is missing required columns.
    FileNotFoundError
        If ``raw_parquet`` is a path that does not exist.

    Notes
    -----
    Each station is processed independently — gaps in one station do not
    affect dB/dt computation for other stations.  This is correct because
    SuperMAG station outages are station-specific events.

    Complexity
    ----------
    Time:  $O(N)$ where N is total rows across all stations.
    Space: $O(N)$ for the output DataFrame.
    """
    raw_df, raw_path = _load_raw_supermag(raw_parquet)
    if write_output is None:
        write_output = raw_path is not None

    required_cols = {"timestamp", "station", "n_nez", "e_nez", "z_nez"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise KeyError(
            f"raw_df missing required columns: {sorted(missing)}. "
            f"Present: {list(raw_df.columns)}"
        )

    if raw_df.empty:
        log.warning("compute_all_station_dbdt received empty DataFrame.")
        result = _empty_all_station_dbdt()
        if write_output:
            final_output_path = _resolve_dbdt_output_path(output_path, raw_path)
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            validate_output_schema(result, "SuperMAG-dbdt-empty", unique_subset=["timestamp", "station"])
            result.to_parquet(final_output_path, index=False)
        return result

    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
    if raw_df["timestamp"].isna().any():
        raise ValueError("raw SuperMAG data contains null or unparseable timestamps.")

    duplicate_mask = raw_df.duplicated(subset=["timestamp", "station"], keep=False)
    if duplicate_mask.any():
        examples = raw_df.loc[duplicate_mask, ["timestamp", "station"]].head(5).to_dict(orient="records")
        raise ValueError(
            "raw SuperMAG data contains duplicate (timestamp, station) rows. "
            f"Examples: {examples}"
        )

    # Ensure sort order: station then timestamp for correct per-station diffs
    raw_df = raw_df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    frames = []
    stations = raw_df["station"].unique()
    log.info(
        "Computing gap-aware dB/dt for %d stations (%d total rows)...",
        len(stations), len(raw_df),
    )

    for station in stations:
        mask = raw_df["station"] == station
        station_data = raw_df.loc[mask]

        b_nez = station_data[["n_nez", "e_nez", "z_nez"]].reset_index(drop=True)
        ts = station_data["timestamp"].reset_index(drop=True)

        dbdt_result = compute_dbdt_gap_aware(b_nez, ts, gap_threshold_sec)

        station_out = pd.DataFrame({
            "timestamp": station_data["timestamp"].reset_index(drop=True),
            "station": station,
            "dbdt_n": dbdt_result["dbdt_n"].values,
            "dbdt_e": dbdt_result["dbdt_e"].values,
            "dbdt_z": dbdt_result["dbdt_z"].values,
            "dbdt_horizontal_magnitude": dbdt_result["dbdt_horizontal_magnitude"].values,
            "dbdt_gap_flag": dbdt_result["dbdt_gap_flag"].values,
        })
        frames.append(station_out)

    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values(["timestamp", "station"]).reset_index(drop=True)

    n_stations = len(stations)
    n_valid = int(result["dbdt_horizontal_magnitude"].notna().sum())
    n_gaps = int(result["dbdt_gap_flag"].sum())
    log.info(
        "dB/dt complete: %d stations, %d total rows, %d valid, %d gap-masked.",
        n_stations, len(result), n_valid, n_gaps,
    )

    if write_output:
        final_output_path = _resolve_dbdt_output_path(output_path, raw_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        validate_output_schema(
            result,
            f"SuperMAG-dbdt-{final_output_path.stem}",
            unique_subset=["timestamp", "station"],
        )
        result.to_parquet(final_output_path, index=False)
        log.info("Saved SuperMAG dB/dt → %s (%d rows)", final_output_path, len(result))

    return result
