"""
validate_feature_matrix.py

Stage 3 validation for fused monthly feature matrices.

Purpose
-------
Validate the fused feature matrix after source joins and feature engineering,
but before any target alignment, sequence construction, or model training.

Checks
------
- File existence and loadability
- Timestamp uniqueness, monotonicity, and full-month expected row count
- Required fused feature presence
- NaN fractions for major feature groups
- Cyclical feature bounds in [-1, 1]
- Presence of Newell coupling and rolling features
- Presence and completeness of rolling valid-point support columns
- Presence and completeness of pivoted SuperMAG station columns
- Basic duplicate-column / constant-column diagnostics

Outputs
-------
data/results/validation/YYYY/MM/feature_validation_summary_YYYYMM.csv
data/results/validation/YYYY/MM/feature_validation_flags_YYYYMM.json
data/results/validation/YYYY/MM/feature_validation_station_cols_YYYYMM.csv
"""

import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)


def _feature_path(year: int, month: int) -> str:
    month_str = f"{year:04d}{month:02d}"
    return os.path.join(
        config.FEATURES_DIR,
        f"{year:04d}",
        f"{month:02d}",
        f"features_{month_str}.parquet",
    )


def _out_dir(year: int, month: int) -> str:
    out_dir = os.path.join(
        "data",
        "results",
        "validation",
        f"{year:04d}",
        f"{month:02d}",
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _month_bounds(year: int, month: int):
    start = pd.Timestamp(year, month, 1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    else:
        end = pd.Timestamp(year, month + 1, 1, tz="UTC")
    return start, end


def _expected_rows(year: int, month: int) -> int:
    start, end = _month_bounds(year, month)
    return int((end - start).total_seconds() // 60)


def _as_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _add_row(rows: List[dict], dataset: str, check: str, status: str, value):
    rows.append({
        "dataset": dataset,
        "check": check,
        "status": status,
        "value": value,
    })


def validate_feature_matrix(year: int, month: int, fail_on_error: bool = True) -> dict:
    month_str = f"{year:04d}{month:02d}"
    month_label = f"{year:04d}-{month:02d}"
    path = _feature_path(year, month)
    out_dir = _out_dir(year, month)

    log.info("=" * 60)
    log.info("STAGE 3: FEATURE MATRIX VALIDATION -- %s", month_label)
    log.info("=" * 60)

    rows: List[dict] = []
    errors: List[str] = []
    warnings_list: List[str] = []

    if not os.path.exists(path):
        msg = f"Missing feature matrix: {path}"
        log.error(msg)
        if fail_on_error:
            raise FileNotFoundError(msg)
        return {
            "year": year,
            "month": month,
            "status": "FAIL",
            "errors": [msg],
            "warnings": [],
        }

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        msg = f"Could not load feature matrix {path}: {exc}"
        log.error(msg)
        if fail_on_error:
            raise RuntimeError(msg) from exc
        return {
            "year": year,
            "month": month,
            "status": "FAIL",
            "errors": [msg],
            "warnings": [],
        }

    df = _as_utc(df)

    # ------------------------------------------------------------------
    # Schema-level validation
    # ------------------------------------------------------------------
    try:
        validate_output_schema(df, f"features-{month_str}")
        _add_row(rows, "features", "schema_validation", "PASS", "ok")
    except Exception as exc:
        _add_row(rows, "features", "schema_validation", "FAIL", str(exc))

    # ------------------------------------------------------------------
    # Timestamp checks
    # ------------------------------------------------------------------
    required_cols = ["timestamp", "ut_sin", "ut_cos", "doy_sin", "doy_cos"]
    missing_required = [c for c in required_cols if c not in df.columns]
    _add_row(
        rows,
        "features",
        "required_columns_present",
        "FAIL" if missing_required else "PASS",
        ",".join(missing_required) if missing_required else "ok",
    )

    _add_row(rows, "features", "row_count", "INFO", int(len(df)))

    if "timestamp" in df.columns:
        nat_count = int(df["timestamp"].isna().sum())
        _add_row(rows, "features", "timestamp_nat_count", "FAIL" if nat_count > 0 else "PASS", nat_count)

        ts = df["timestamp"].dropna()
        dup_count = int(ts.duplicated().sum())
        _add_row(rows, "features", "duplicate_timestamps", "FAIL" if dup_count > 0 else "PASS", dup_count)

        is_sorted = bool(ts.is_monotonic_increasing)
        _add_row(rows, "features", "timestamp_monotonic", "WARN" if not is_sorted else "PASS", is_sorted)

        exp_rows = _expected_rows(year, month)
        _add_row(
            rows,
            "features",
            "expected_month_rows",
            "WARN" if len(df) != exp_rows else "PASS",
            f"actual={len(df)} expected={exp_rows}",
        )

        if len(ts) > 0:
            start_expected, end_expected = _month_bounds(year, month)
            ts_min = ts.min()
            ts_max = ts.max()
            _add_row(
                rows,
                "features",
                "timestamp_min",
                "WARN" if ts_min != start_expected else "PASS",
                str(ts_min),
            )
            _add_row(
                rows,
                "features",
                "timestamp_max",
                "WARN" if ts_max != (end_expected - pd.Timedelta(minutes=1)) else "PASS",
                str(ts_max),
            )

    # ------------------------------------------------------------------
    # Core fused-feature presence
    # ------------------------------------------------------------------
    core_cols = [
        "omni_bz_gsm",
        "omni_by_gsm",
        "omni_vx",
        "newell_phi",
        "leo_high_lat",
        "ut_sin",
        "ut_cos",
        "doy_sin",
        "doy_cos",
    ]
    for col in core_cols:
        _add_row(
            rows,
            "features",
            f"has_{col}",
            "PASS" if col in df.columns else "WARN",
            col in df.columns,
        )

    # ------------------------------------------------------------------
    # Missingness by group
    # ------------------------------------------------------------------
    def _missing_frac(col: str):
        return float(df[col].isna().mean()) if col in df.columns else np.nan

    major_cols = [
        "omni_bz_gsm",
        "omni_by_gsm",
        "omni_vx",
        "omni_proton_density",
        "omni_pressure",
        "goes_bz_gsm",
        "leo_high_lat",
        "newell_phi",
    ]
    for col in major_cols:
        if col in df.columns:
            frac = _missing_frac(col)
            status = "WARN" if frac > 0.25 else "PASS"
            if col in {"omni_bz_gsm", "omni_by_gsm", "omni_vx", "newell_phi"} and frac > 0.10:
                status = "WARN"
            _add_row(rows, "features", f"missing_frac_{col}", status, frac)

    # ------------------------------------------------------------------
    # Cyclical feature bounds
    # ------------------------------------------------------------------
    for col in ["ut_sin", "ut_cos", "doy_sin", "doy_cos"]:
        if col in df.columns:
            bad = int(((df[col] < -1.000001) | (df[col] > 1.000001)).fillna(False).sum())
            _add_row(rows, "features", f"{col}_in_unit_range", "FAIL" if bad > 0 else "PASS", bad)

    # ------------------------------------------------------------------
    # Newell and rolling feature checks
    # ------------------------------------------------------------------
    expected_rolling = [
        "omni_bz_gsm_mean_10m",
        "omni_bz_gsm_std_10m",
        "omni_bz_gsm_mean_30m",
        "omni_bz_gsm_std_30m",
        "omni_vx_mean_10m",
        "omni_vx_std_10m",
        "omni_vx_mean_30m",
        "omni_vx_std_30m",
        "newell_phi_mean_10m",
        "newell_phi_std_10m",
        "newell_phi_mean_30m",
        "newell_phi_std_30m",
    ]
    missing_roll = [c for c in expected_rolling if c not in df.columns]
    _add_row(
        rows,
        "features",
        "rolling_features_present",
        "WARN" if missing_roll else "PASS",
        ",".join(missing_roll) if missing_roll else "ok",
    )

    # ------------------------------------------------------------------
    # Rolling support-count feature checks
    # ------------------------------------------------------------------
    rolling_base_cols = ["omni_bz_gsm", "omni_vx", "newell_phi"]
    rolling_windows = [10, 30]

    expected_valid_count_cols = [
        f"{col}_valid_points_{w}m"
        for col in rolling_base_cols
        for w in rolling_windows
    ]

    missing_valid_count_cols = [c for c in expected_valid_count_cols if c not in df.columns]
    _add_row(
        rows,
        "features",
        "rolling_valid_point_columns_present",
        "WARN" if missing_valid_count_cols else "PASS",
        ",".join(missing_valid_count_cols) if missing_valid_count_cols else "ok",
    )

    for col in rolling_base_cols:
        for w in rolling_windows:
            count_col = f"{col}_valid_points_{w}m"
            mean_col = f"{col}_mean_{w}m"
            std_col = f"{col}_std_{w}m"

            if count_col in df.columns:
                vals = pd.to_numeric(df[count_col], errors="coerce")

                bad_low = int((vals < 0).fillna(False).sum())
                bad_high = int((vals > w).fillna(False).sum())
                non_integer_like = int(((vals.dropna() % 1) != 0).sum())

                _add_row(
                    rows,
                    "features",
                    f"{count_col}_range_low",
                    "FAIL" if bad_low > 0 else "PASS",
                    bad_low,
                )
                _add_row(
                    rows,
                    "features",
                    f"{count_col}_range_high",
                    "FAIL" if bad_high > 0 else "PASS",
                    bad_high,
                )
                _add_row(
                    rows,
                    "features",
                    f"{count_col}_integer_like",
                    "WARN" if non_integer_like > 0 else "PASS",
                    non_integer_like,
                )
                _add_row(
                    rows,
                    "features",
                    f"{count_col}_mean",
                    "INFO",
                    float(vals.mean()) if len(vals.dropna()) else np.nan,
                )

            if count_col in df.columns and mean_col in df.columns:
                vals = pd.to_numeric(df[count_col], errors="coerce")
                inconsistent_mean = int(
                    ((vals == 0) & df[mean_col].notna()).fillna(False).sum()
                )
                _add_row(
                    rows,
                    "features",
                    f"{mean_col}_consistent_with_{count_col}",
                    "FAIL" if inconsistent_mean > 0 else "PASS",
                    inconsistent_mean,
                )

            if count_col in df.columns and std_col in df.columns:
                vals = pd.to_numeric(df[count_col], errors="coerce")
                inconsistent_std = int(
                    ((vals == 0) & df[std_col].notna()).fillna(False).sum()
                )
                _add_row(
                    rows,
                    "features",
                    f"{std_col}_consistent_with_{count_col}",
                    "FAIL" if inconsistent_std > 0 else "PASS",
                    inconsistent_std,
                )

    # ------------------------------------------------------------------
    # Gap flag checks
    # ------------------------------------------------------------------
    gap_cols = [c for c in df.columns if c.endswith("_missing") or c.endswith("_ffill_applied")]
    _add_row(rows, "features", "gap_flag_column_count", "INFO", len(gap_cols))

    if "l1_any_missing" in df.columns:
        frac = float(df["l1_any_missing"].mean())
        _add_row(rows, "features", "mean_l1_any_missing", "WARN" if frac > 0.25 else "PASS", frac)

    if "geo_any_missing" in df.columns:
        frac = float(df["geo_any_missing"].mean())
        _add_row(rows, "features", "mean_geo_any_missing", "WARN" if frac > 0.50 else "PASS", frac)

    # ------------------------------------------------------------------
    # Station-derived SuperMAG pivot columns
    # ------------------------------------------------------------------
    station_metric_suffixes = [
        "_b_n",
        "_b_e",
        "_b_z",
        "_dbn_dt",
        "_dbe_dt",
        "_dbdt_magnitude",
        "_dbdt_missing_flag",
    ]
    station_cols = [c for c in df.columns if any(c.endswith(sfx) for sfx in station_metric_suffixes)]

    _add_row(rows, "features", "station_derived_column_count", "WARN" if len(station_cols) == 0 else "PASS", len(station_cols))

    station_records = []
    station_names = sorted({
        c[: -len("_dbdt_magnitude")]
        for c in df.columns
        if c.endswith("_dbdt_magnitude")
    })

    for station in station_names:
        mag_col = f"{station}_dbdt_magnitude"
        miss_col = f"{station}_dbdt_missing_flag"
        rec = {
            "station": station,
            "has_dbdt_magnitude": mag_col in df.columns,
            "missing_frac_dbdt_magnitude": float(df[mag_col].isna().mean()) if mag_col in df.columns else np.nan,
            "mean_dbdt_missing_flag": float(df[miss_col].mean()) if miss_col in df.columns else np.nan,
        }
        station_records.append(rec)

        if mag_col in df.columns:
            frac = float(df[mag_col].isna().mean())
            _add_row(
                rows,
                "features",
                f"missing_frac_{mag_col}",
                "WARN" if frac > 0.50 else "PASS",
                frac,
            )

    station_df = pd.DataFrame(station_records)
    station_out = os.path.join(out_dir, f"feature_validation_station_cols_{month_str}.csv")
    station_df.to_csv(station_out, index=False)
    log.info("Saved feature station-column summary -> %s", station_out)

    # ------------------------------------------------------------------
    # Constant-column diagnostics
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0 and vals.nunique() <= 1:
            constant_cols.append(col)

    _add_row(
        rows,
        "features",
        "constant_numeric_column_count",
        "WARN" if len(constant_cols) > 10 else "PASS",
        len(constant_cols),
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)

    errors = [
        f"{r['dataset']}::{r['check']} -> {r['value']}"
        for _, r in summary_df.iterrows()
        if r["status"] == "FAIL"
    ]
    warnings_list = [
        f"{r['dataset']}::{r['check']} -> {r['value']}"
        for _, r in summary_df.iterrows()
        if r["status"] == "WARN"
    ]

    summary_out = os.path.join(out_dir, f"feature_validation_summary_{month_str}.csv")
    summary_df.to_csv(summary_out, index=False)

    flags = {
        "year": year,
        "month": month,
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings_list else "PASS",
        "n_errors": len(errors),
        "n_warnings": len(warnings_list),
        "errors": errors,
        "warnings": warnings_list[:200],
        "summary_csv": summary_out,
        "station_summary_csv": station_out,
    }

    flags_out = os.path.join(out_dir, f"feature_validation_flags_{month_str}.json")
    with open(flags_out, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)

    log.info("Saved feature validation summary -> %s", summary_out)
    log.info("Saved feature validation flags   -> %s", flags_out)

    if warnings_list:
        log.warning("Feature validation warnings for %s: %d", month_label, len(warnings_list))
        for msg in warnings_list[:20]:
            log.warning("  %s", msg)

    if errors:
        log.error("Feature validation errors for %s: %d", month_label, len(errors))
        for msg in errors[:20]:
            log.error("  %s", msg)
        if fail_on_error:
            raise RuntimeError(f"Feature matrix validation failed for {month_label}. See {flags_out}")

    return flags


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate fused feature matrix for one month.")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2015")
    parser.add_argument("--month", type=int, required=True, help="Calendar month, e.g. 3")
    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Do not raise RuntimeError on hard validation failures.",
    )
    args = parser.parse_args()

    validate_feature_matrix(
        year=args.year,
        month=args.month,
        fail_on_error=not args.no_fail_on_error,
    )
