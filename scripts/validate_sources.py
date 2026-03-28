"""
validate_sources.py

Stage 2 validation for raw/processed pipeline datasets.

Purpose
-------
Validate each source independently before source fusion in build_feature_matrix.py.

Checks
------
- File existence and loadability
- Timestamp presence, monotonicity, and duplicate handling
- Required-column checks
- Missingness and row-count summaries
- SuperMAG composite-key uniqueness on ["station", "timestamp"]
- Per-station completeness summaries
- LEO freshness / decay-age summaries
- LEO vs AL / AE correlation sanity checks
- Optional storm-window physics plot for a known interval

Outputs
-------
data/results/validation/YYYY/MM/validation_summary_YYYYMM.csv
data/results/validation/YYYY/MM/validation_flags_YYYYMM.json
data/results/validation/YYYY/MM/validation_station_completeness_YYYYMM.csv
data/results/validation/YYYY/MM/validation_leo_corr_YYYYMM.csv
data/results/validation/YYYY/MM/validation_stormplot_YYYYMM.png
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)


HARD_FAIL_MISSING = {
    "omni_l1",
    "leo_index",
    "supermag",
}

PRIMARY_COLS = {
    "omni_l1": "BZ_GSM",
    "goes_gsm": "goes_bz_gsm",
    "leo_index": "leo_high_lat",
    "supermag": "dbdt_magnitude",
    "swarm_a_raw": "F",
    "swarm_b_raw": "F",
    "swarm_c_raw": "F",
}

REQUIRED_COLS = {
    "omni_l1": ["timestamp", "BZ_GSM"],
    "goes_gsm": ["timestamp"],
    "leo_index": ["timestamp", "leo_high_lat"],
    "supermag": ["timestamp", "station"],
    "swarm_a_raw": ["timestamp"],
    "swarm_b_raw": ["timestamp"],
    "swarm_c_raw": ["timestamp"],
}


def _month_paths(year: int, month: int) -> Dict[str, str]:
    ms = f"{year:04d}{month:02d}"
    y = f"{year:04d}"
    m = f"{month:02d}"

    return {
        "omni_l1": os.path.join(config.PROCESSED_DIR, "omni", y, m, f"omni_{ms}.parquet"),
        "goes_gsm": os.path.join(config.PROCESSED_DIR, "goes", y, m, f"goes15_{ms}.parquet"),
        "leo_index": os.path.join(config.PROCESSED_DIR, "swarm", y, m, f"swarm_leo_index_{ms}.parquet"),
        "supermag": os.path.join(config.PROCESSED_DIR, "supermag", y, m, f"supermag_{ms}.parquet"),
        "swarm_a_raw": os.path.join(config.RAW_DATA_DIR, "swarm", y, m, f"swarmA_LR1B_{ms}.parquet"),
        "swarm_b_raw": os.path.join(config.RAW_DATA_DIR, "swarm", y, m, f"swarmB_LR1B_{ms}.parquet"),
        "swarm_c_raw": os.path.join(config.RAW_DATA_DIR, "swarm", y, m, f"swarmC_LR1B_{ms}.parquet"),
    }


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


def _as_utc_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _safe_read_parquet(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.error("Failed to load %s: %s", path, exc)
        return None


def _pearson_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 10:
        return {
            "n_overlap": int(len(tmp)),
            "pearson": np.nan,
            "spearman": np.nan,
        }

    return {
        "n_overlap": int(len(tmp)),
        "pearson": float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="pearson")),
        "spearman": float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman")),
    }


def _dataset_basic_checks(
    name: str,
    df: pd.DataFrame,
) -> List[dict]:
    rows = []

    required = REQUIRED_COLS.get(name, [])
    missing_required = [c for c in required if c not in df.columns]
    rows.append({
        "dataset": name,
        "check": "required_columns_present",
        "status": "FAIL" if missing_required else "PASS",
        "value": ",".join(missing_required) if missing_required else "ok",
    })

    rows.append({
        "dataset": name,
        "check": "row_count",
        "status": "INFO",
        "value": int(len(df)),
    })

    pcol = PRIMARY_COLS.get(name)
    if pcol and pcol in df.columns:
        missing = int(df[pcol].isna().sum())
        frac = float(missing / max(len(df), 1))
        rows.append({
            "dataset": name,
            "check": f"missing_frac_{pcol}",
            "status": "WARN" if frac > 0.25 else "PASS",
            "value": frac,
        })

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        rows.append({
            "dataset": name,
            "check": "timestamp_nat_count",
            "status": "FAIL" if ts.isna().any() else "PASS",
            "value": int(ts.isna().sum()),
        })

        ts_valid = ts.dropna()
        is_sorted = bool(ts_valid.is_monotonic_increasing)
        rows.append({
            "dataset": name,
            "check": "timestamp_monotonic",
            "status": "WARN" if not is_sorted else "PASS",
            "value": is_sorted,
        })

        if name != "supermag":
            dup_count = int(ts_valid.duplicated().sum())
            rows.append({
                "dataset": name,
                "check": "duplicate_timestamps",
                "status": "FAIL" if dup_count > 0 else "PASS",
                "value": dup_count,
            })

    return rows


def _validate_omni(df: pd.DataFrame) -> List[dict]:
    rows = []
    for col in ["BZ_GSM", "BY_GSM", "Vx", "proton_density", "Pressure", "SYM_H", "AL_INDEX", "AU_INDEX"]:
        if col in df.columns:
            frac = float(df[col].isna().mean())
            rows.append({
                "dataset": "omni_l1",
                "check": f"missing_frac_{col}",
                "status": "WARN" if frac > 0.10 else "PASS",
                "value": frac,
            })

    if "omni_bz_gsm" in df.columns:
        bad = int(((df["BZ_GSM"] < -200) | (df["BZ_GSM"] > 200)).fillna(False).sum())
        rows.append({
            "dataset": "omni_l1",
            "check": "bz_range_sanity",
            "status": "WARN" if bad > 0 else "PASS",
            "value": bad,
        })

    return rows


def _validate_goes(df: pd.DataFrame) -> List[dict]:
    rows = []
    if "goes_bz_gsm" in df.columns:
        frac = float(df["goes_bz_gsm"].isna().mean())
        rows.append({
            "dataset": "goes_gsm",
            "check": "missing_frac_goes_bz_gsm",
            "status": "WARN" if frac > 0.25 else "PASS",
            "value": frac,
        })

    if "goes_missing_flag" in df.columns:
        frac = float(df["goes_missing_flag"].mean())
        rows.append({
            "dataset": "goes_gsm",
            "check": "goes_missing_flag_mean",
            "status": "WARN" if frac > 0.25 else "PASS",
            "value": frac,
        })

    return rows


def _validate_supermag(df: pd.DataFrame, year: int, month: int, out_dir: str) -> List[dict]:
    rows = []

    try:
        validate_output_schema(
            _as_utc_timestamp(df),
            f"supermag-{year:04d}{month:02d}",
            unique_subset=["station", "timestamp"],
        )
        rows.append({
            "dataset": "supermag",
            "check": "unique_station_timestamp",
            "status": "PASS",
            "value": "ok",
        })
    except Exception as exc:
        rows.append({
            "dataset": "supermag",
            "check": "unique_station_timestamp",
            "status": "FAIL",
            "value": str(exc),
        })

    if "station" in df.columns and "timestamp" in df.columns:
        df = _as_utc_timestamp(df).dropna(subset=["timestamp"]).copy()

        month_start = pd.Timestamp(year, month, 1, tz="UTC")
        if month == 12:
            month_end = pd.Timestamp(year + 1, 1, 1, tz="UTC")
        else:
            month_end = pd.Timestamp(year, month + 1, 1, tz="UTC")
        expected_minutes = int((month_end - month_start).total_seconds() // 60)

        station_rows = []
        for station, grp in df.groupby("station"):
            n_rows = int(len(grp))
            completeness = float(n_rows / max(expected_minutes, 1))
            record = {
                "station": station,
                "rows": n_rows,
                "completeness_frac": completeness,
            }

            if "dbdt_magnitude" in grp.columns:
                vals = grp["dbdt_magnitude"].dropna()
                record["dbdt_p50"] = float(vals.quantile(0.50)) if len(vals) else np.nan
                record["dbdt_p95"] = float(vals.quantile(0.95)) if len(vals) else np.nan
                record["dbdt_p99"] = float(vals.quantile(0.99)) if len(vals) else np.nan
                record["missing_frac_dbdt"] = float(grp["dbdt_magnitude"].isna().mean())
            station_rows.append(record)

        station_df = pd.DataFrame(station_rows).sort_values(["completeness_frac", "station"], ascending=[False, True])
        station_out = os.path.join(out_dir, f"validation_station_completeness_{year:04d}{month:02d}.csv")
        station_df.to_csv(station_out, index=False)
        log.info("Saved station completeness → %s", station_out)

        low_comp = int((station_df["completeness_frac"] < 0.80).sum()) if not station_df.empty else 0
        rows.append({
            "dataset": "supermag",
            "check": "stations_below_80pct_completeness",
            "status": "WARN" if low_comp > 0 else "PASS",
            "value": low_comp,
        })

    return rows


def _validate_leo(df: pd.DataFrame, omni_df: Optional[pd.DataFrame], year: int, month: int, out_dir: str) -> List[dict]:
    rows = []

    for col in ["leo_high_lat", "leo_mid_lat", "leo_dayside", "leo_nightside"]:
        if col in df.columns:
            frac = float(df[col].isna().mean())
            rows.append({
                "dataset": "leo_index",
                "check": f"missing_frac_{col}",
                "status": "WARN" if frac > 0.80 else "PASS",
                "value": frac,
            })

    fresh_cols = [c for c in df.columns if c.endswith("_is_fresh")]
    for col in fresh_cols:
        fresh_frac = float(df[col].mean())
        rows.append({
            "dataset": "leo_index",
            "check": f"fresh_frac_{col}",
            "status": "WARN" if fresh_frac < 0.05 else "PASS",
            "value": fresh_frac,
        })

    age_cols = [c for c in df.columns if c.endswith("_decay_age")]
    for col in age_cols:
        med_age = float(df[col].dropna().median()) if df[col].notna().any() else np.nan
        rows.append({
            "dataset": "leo_index",
            "check": f"median_{col}",
            "status": "WARN" if np.isfinite(med_age) and med_age > 120 else "PASS",
            "value": med_age,
        })

    corr_rows = []
    if omni_df is not None and "timestamp" in omni_df.columns:
        omni = _as_utc_timestamp(omni_df).copy()
        leo = _as_utc_timestamp(df).copy()

        if "timestamp" in leo.columns:
            merged = leo.merge(
                omni[["timestamp"] + [c for c in ["omni_al", "omni_ae"] if c in omni.columns]],
                on="timestamp",
                how="inner",
            )

            for leo_col in ["leo_high_lat", "leo_nightside"]:
                if leo_col not in merged.columns:
                    continue
                for omni_col in ["omni_al", "omni_ae"]:
                    if omni_col not in merged.columns:
                        continue

                    stats = _pearson_spearman(merged[leo_col], merged[omni_col])
                    corr_rows.append({
                        "leo_col": leo_col,
                        "omni_col": omni_col,
                        **stats,
                    })

                    warn = False
                    if stats["n_overlap"] >= 100 and np.isfinite(stats["spearman"]):
                        warn = abs(stats["spearman"]) < 0.15

                    rows.append({
                        "dataset": "leo_index",
                        "check": f"corr_{leo_col}_vs_{omni_col}_spearman",
                        "status": "WARN" if warn else "PASS",
                        "value": stats["spearman"],
                    })

    corr_df = pd.DataFrame(corr_rows)
    corr_out = os.path.join(out_dir, f"validation_leo_corr_{year:04d}{month:02d}.csv")
    corr_df.to_csv(corr_out, index=False)
    log.info("Saved LEO correlation summary → %s", corr_out)

    return rows


def _validate_swarm_raw(name: str, df: pd.DataFrame) -> List[dict]:
    rows = []

    for col in ["B_NEC", "QDLat", "MLT", "Latitude", "Longitude", "Radius"]:
        if col in df.columns:
            frac = float(df[col].isna().mean())
            rows.append({
                "dataset": name,
                "check": f"missing_frac_{col}",
                "status": "WARN" if frac > 0.10 else "PASS",
                "value": frac,
            })

    if "Flags_B" in df.columns:
        bad_frac = float((df["Flags_B"] != 0).mean())
        rows.append({
            "dataset": name,
            "check": "bad_flags_frac",
            "status": "WARN" if bad_frac > 0.25 else "PASS",
            "value": bad_frac,
        })

    return rows


def _slice_ts(df: Optional[pd.DataFrame], col: str, start: str, end: str) -> Optional[pd.DataFrame]:
    if df is None or col not in df.columns or "timestamp" not in df.columns:
        return None
    work = _as_utc_timestamp(df).dropna(subset=["timestamp"]).copy()
    work = work.set_index("timestamp").sort_index()
    return work.loc[start:end]


def _save_storm_plot(
    datasets: Dict[str, pd.DataFrame],
    year: int,
    month: int,
    out_dir: str,
) -> Optional[str]:
    if (year, month) != (2015, 3):
        return None

    start_plot = "2015-03-16"
    end_plot = "2015-03-19"

    omni_s = _slice_ts(datasets.get("omni_l1"), "omni_bz_gsm", start_plot, end_plot)
    goes_s = _slice_ts(datasets.get("goes_gsm"), "goes_bz_gsm", start_plot, end_plot)
    leo_s = _slice_ts(datasets.get("leo_index"), "leo_high_lat", start_plot, end_plot)

    smag_raw = datasets.get("supermag")
    smag_abk = None
    if smag_raw is not None and "station" in smag_raw.columns and "dbdt_magnitude" in smag_raw.columns:
        smag_abk = _slice_ts(smag_raw[smag_raw["station"] == "ABK"].copy(), "dbdt_magnitude", start_plot, end_plot)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("St. Patrick's Day Storm (2015-03-17) -- Validation", fontsize=13)

    if omni_s is not None and not omni_s.empty:
        axes[0].plot(omni_s.index, omni_s["omni_bz_gsm"], color="red", label="IMF Bz (GSM)")
    axes[0].set_ylabel("IMF Bz")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if goes_s is not None and not goes_s.empty:
        axes[1].plot(goes_s.index, goes_s["goes_bz_gsm"], color="purple", label="GOES Bz")
    axes[1].set_ylabel("GOES Bz")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if smag_abk is not None and not smag_abk.empty:
        axes[2].plot(smag_abk.index, smag_abk["dbdt_magnitude"], color="black", label="ABK dB/dt")
    axes[2].set_ylabel("ABK dB/dt")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if leo_s is not None and not leo_s.empty:
        axes[3].plot(leo_s.index, leo_s["leo_high_lat"], color="blue", label="LEO high-lat")
    axes[3].set_ylabel("LEO res_H")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"validation_stormplot_{year:04d}{month:02d}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    log.info("Saved validation storm plot → %s", plot_path)
    return plot_path


def validate_datasets(year: int, month: int, fail_on_error: bool = True) -> dict:
    month_str = f"{year:04d}-{month:02d}"
    ms = f"{year:04d}{month:02d}"
    files = _month_paths(year, month)
    out_dir = _out_dir(year, month)

    log.info("=" * 60)
    log.info("STAGE 2: INDEPENDENT SOURCE VALIDATION -- %s", month_str)
    log.info("=" * 60)

    datasets: Dict[str, pd.DataFrame] = {}
    summary_rows: List[dict] = []
    errors: List[str] = []
    warnings_list: List[str] = []

    for name, path in files.items():
        if not os.path.exists(path):
            msg = f"Missing file: {name} -> {path}"
            if name in HARD_FAIL_MISSING:
                errors.append(msg)
                log.error(msg)
            else:
                warnings_list.append(msg)
                log.warning(msg)
            continue

        df = _safe_read_parquet(path)
        if df is None:
            msg = f"Unreadable file: {name} -> {path}"
            if name in HARD_FAIL_MISSING:
                errors.append(msg)
            else:
                warnings_list.append(msg)
            continue

        df = _as_utc_timestamp(df)
        datasets[name] = df
        summary_rows.extend(_dataset_basic_checks(name, df))

    if "omni_l1" in datasets:
        summary_rows.extend(_validate_omni(datasets["omni_l1"]))

    if "goes_gsm" in datasets:
        summary_rows.extend(_validate_goes(datasets["goes_gsm"]))

    if "supermag" in datasets:
        summary_rows.extend(_validate_supermag(datasets["supermag"], year, month, out_dir))

    if "leo_index" in datasets:
        summary_rows.extend(_validate_leo(datasets["leo_index"], datasets.get("omni_l1"), year, month, out_dir))

    for raw_name in ["swarm_a_raw", "swarm_b_raw", "swarm_c_raw"]:
        if raw_name in datasets:
            summary_rows.extend(_validate_swarm_raw(raw_name, datasets[raw_name]))

    summary_df = pd.DataFrame(summary_rows)

    for _, row in summary_df.iterrows():
        if row["status"] == "FAIL":
            errors.append(f"{row['dataset']}::{row['check']} -> {row['value']}")
        elif row["status"] == "WARN":
            warnings_list.append(f"{row['dataset']}::{row['check']} -> {row['value']}")

    summary_out = os.path.join(out_dir, f"validation_summary_{ms}.csv")
    summary_df.to_csv(summary_out, index=False)

    plot_path = _save_storm_plot(datasets, year, month, out_dir)

    flags = {
        "year": year,
        "month": month,
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings_list else "PASS",
        "n_errors": len(errors),
        "n_warnings": len(warnings_list),
        "errors": errors,
        "warnings": warnings_list[:200],
        "summary_csv": summary_out,
        "storm_plot": plot_path,
    }

    flags_out = os.path.join(out_dir, f"validation_flags_{ms}.json")
    with open(flags_out, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)

    log.info("Saved validation summary → %s", summary_out)
    log.info("Saved validation flags   → %s", flags_out)

    if warnings_list:
        log.warning("Validation warnings for %s: %d", month_str, len(warnings_list))
        for msg in warnings_list[:20]:
            log.warning("  %s", msg)

    if errors:
        log.error("Validation errors for %s: %d", month_str, len(errors))
        for msg in errors[:20]:
            log.error("  %s", msg)
        if fail_on_error:
            raise RuntimeError(f"Source validation failed for {month_str}. See {flags_out}")

    return flags


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate raw/processed source datasets for one month.")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2015")
    parser.add_argument("--month", type=int, required=True, help="Calendar month, e.g. 3")
    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Do not raise RuntimeError on hard validation failures.",
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    validate_datasets(
        year=args.year,
        month=args.month,
        fail_on_error=not args.no_fail_on_error,
    )
