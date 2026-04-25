"""
validate_sources.py

Stage 2 validation for March 2015 MVP datasets.

What this script checks
- Duplicate timestamps in each source product
- Timestamp range and out-of-month rows for OMNI
- Manifest-style completeness checks for OMNI, SuperMAG, GOES, raw Swarm, and aligned LEO index
- Swarm vector availability separate from quality-flag filtering
- Swarm quality-flag distributions with MVP-safe interpretation
- Physics sanity plot across L1, GEO, ground, and LEO during the March 2015 storm

Assumptions
- GOES file already contains a GSM component selected during preprocessing, so no coordinate rotation
  is performed here.
- Swarm Tier 1 index has already been computed in a separate script and saved as an aligned 1-minute product.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MARCH_2015_ROWS = 31 * 24 * 60  # 44640
MARCH_START = pd.Timestamp("2015-03-01 00:00:00", tz="UTC")
MARCH_END = pd.Timestamp("2015-03-31 23:59:00", tz="UTC")


def ensure_utc_timestamp(df, col="timestamp"):
    if col not in df.columns:
        raise KeyError(f"Expected timestamp column '{col}' not found.")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def set_ts_index(df):
    df = ensure_utc_timestamp(df)
    return df.sort_values("timestamp").set_index("timestamp")


def find_col(df, candidates, dataset_name):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"{dataset_name}: none of these columns were found: {candidates}")


def report_duplicate_timestamps(df, name, ts_col="timestamp", max_examples=5):
    if ts_col not in df.columns:
        raise KeyError(f"{name}: timestamp column '{ts_col}' not found.")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    null_ts = int(ts.isna().sum())
    dup_mask = ts.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    unique_dup_timestamps = ts[dup_mask].drop_duplicates().sort_values()

    print(
        f"{name:<22} | null_timestamps={null_ts:<5} | "
        f"duplicate_rows={dup_count:<6} | unique_duplicate_times={len(unique_dup_timestamps)}"
    )

    if len(unique_dup_timestamps) > 0:
        print(f"  Duplicate timestamp examples for {name}:")
        for t in unique_dup_timestamps[:max_examples]:
            n = int((ts == t).sum())
            print(f"    {t}  ({n} rows)")


def report_time_range(df, name, ts_col="timestamp", window_start=None, window_end=None, max_examples=10):
    if ts_col not in df.columns:
        raise KeyError(f"{name}: timestamp column '{ts_col}' not found.")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    ts_min = ts.min()
    ts_max = ts.max()

    print(f"{name:<22} | min_timestamp={ts_min} | max_timestamp={ts_max}")

    if window_start is not None and window_end is not None:
        outside_mask = (ts < window_start) | (ts > window_end)
        outside_count = int(outside_mask.sum())
        print(f"{name:<22} | outside_window_rows={outside_count}")

        if outside_count > 0:
            print(f"  Outside-window timestamp examples for {name}:")
            sample = ts[outside_mask].sort_values().head(max_examples)
            for t in sample:
                print(f"    {t}")


def debug_swarm_schema(df, name, max_examples=3):
    print(f"\n--- DEBUG: {name} SCHEMA ---")
    print(f"Columns: {list(df.columns)}")
    print("Dtypes:")
    print(df.dtypes.astype(str).to_string())

    candidate_cols = [
        "B_NEC", "BNEC",
        "B_N", "B_E", "B_C",
        "BN", "BE", "BC",
        "B_NEC_N", "B_NEC_E", "B_NEC_C",
        "F", "Flags_B", "FlagsB", "QDLat", "MLT", "Latitude", "Longitude", "Radius"
    ]
    existing = [c for c in candidate_cols if c in df.columns]
    print(f"Candidate science/debug columns present: {existing}")

    for col in ["B_NEC", "BNEC", "F", "Flags_B", "FlagsB", "QDLat"]:
        if col in df.columns:
            nonnull = df[col].dropna()
            print(f"\nSample values from {col}:")
            if nonnull.empty:
                print("  <all null>")
            else:
                for i, val in enumerate(nonnull.head(max_examples)):
                    print(f"  [{i}] type={type(val)} value={repr(val)[:200]}")

    print(f"--- END DEBUG: {name} ---\n")


def infer_swarm_vector_present_mask(df):
    vector_cols = ["B_NEC", "BNEC"]
    for col in vector_cols:
        if col in df.columns:
            def is_valid_vec(x):
                if x is None:
                    return False
                try:
                    arr = np.asarray(x, dtype=float).reshape(-1)
                    return len(arr) == 3 and np.isfinite(arr).all()
                except Exception:
                    return False
            return df[col].apply(is_valid_vec)

    component_sets = [
        ["B_N", "B_E", "B_C"],
        ["BN", "BE", "BC"],
        ["B_NEC_N", "B_NEC_E", "B_NEC_C"],
    ]
    for cols in component_sets:
        if all(c in df.columns for c in cols):
            return df[cols].notna().all(axis=1)

    raise KeyError(
        "Could not find Swarm vector field columns. "
        "Expected one of: B_NEC, BNEC, or separate NEC component columns."
    )


def infer_swarm_qc_pass_mask(df):
    flag_col = None
    for c in ["Flags_B", "FlagsB", "flags_b", "flagsb"]:
        if c in df.columns:
            flag_col = c
            break

    if flag_col is None:
        return pd.Series(True, index=df.index)

    flags = df[flag_col].fillna(255).astype(int)

    # MVP policy:
    # 0 = nominal VFM
    # 1 = ASM off; still acceptable for vector-based B_NEC use
    # 255 = not enough VFM samples to generate B_VFM/B_NEC -> reject
    # all other values = anomalous/provisional reject for this raw QC summary
    return flags.isin([0, 1])


def summarize_series_missing(df, name, primary_col, rows_expected=None):
    total_rows = len(df)
    if primary_col not in df.columns:
        raise KeyError(f"{name}: expected column '{primary_col}' not found.")

    missing = int(df[primary_col].isna().sum())
    missing_frac = missing / total_rows if total_rows else math.nan

    msg = (
        f"{name:<22} | rows={total_rows:<8} | "
        f"missing {primary_col}={missing_frac:>7.2%} ({missing})"
    )

    if rows_expected is not None:
        row_gap = rows_expected - total_rows
        msg += f" | expected={rows_expected} | row_gap={row_gap}"

    print(msg)


def summarize_swarm_raw(df, name, debug_threshold=0.50):
    total_rows = len(df)

    vector_present_mask = infer_swarm_vector_present_mask(df)
    vector_present_rows = int(vector_present_mask.sum())
    vector_missing_rows = total_rows - vector_present_rows
    vector_missing_frac = vector_missing_rows / total_rows if total_rows else math.nan

    qc_pass_mask = infer_swarm_qc_pass_mask(df)
    qc_pass_rows = int((vector_present_mask & qc_pass_mask).sum())
    qc_fail_rows = total_rows - qc_pass_rows
    qc_fail_frac = qc_fail_rows / total_rows if total_rows else math.nan

    msg = (
        f"{name:<22} | rows={total_rows:<8} | "
        f"missing/invalid vectors={vector_missing_frac:>7.2%} ({vector_missing_rows}) | "
        f"qc_fail_rows={qc_fail_frac:>7.2%} ({qc_fail_rows})"
    )

    if "QDLat" in df.columns:
        qdlat_missing = int(df["QDLat"].isna().sum())
        msg += f" | missing QDLat={qdlat_missing / total_rows:>7.2%} ({qdlat_missing})"

    print(msg)

    flag_col = None
    for c in ["Flags_B", "FlagsB", "flags_b", "flagsb"]:
        if c in df.columns:
            flag_col = c
            break

    if flag_col is not None:
        flag_counts = df[flag_col].value_counts(dropna=False).sort_index()
        print(f"  {name} {flag_col} distribution:")
        for k, v in flag_counts.items():
            print(f"    {k}: {v}")

    if vector_missing_frac >= debug_threshold:
        debug_swarm_schema(df, name)


def summarize_leo_index(df, name, rows_expected=None):
    total_rows = len(df)

    required = ["timestamp", "leo_index_global"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise KeyError(f"{name}: missing required columns {missing_required}")

    msg = [f"{name:<22} | rows={total_rows:<8}"]

    if rows_expected is not None:
        row_gap = rows_expected - total_rows
        msg.append(f"expected={rows_expected}")
        msg.append(f"row_gap={row_gap}")

    idx_missing = int(df["leo_index_global"].isna().sum())
    msg.append(f"missing leo_index_global={idx_missing / total_rows:>7.2%} ({idx_missing})")

    fresh_col = None
    for c in ["is_fresh", "isfresh"]:
        if c in df.columns:
            fresh_col = c
            break

    count_col = None
    for c in ["leo_count", "leocount"]:
        if c in df.columns:
            count_col = c
            break

    if fresh_col is not None:
        fresh_frac = df[fresh_col].fillna(0).mean()
        msg.append(f"fresh_frac={fresh_frac:>7.2%}")
        msg.append(f"persisted_frac={1.0 - fresh_frac:>7.2%}")

    if count_col is not None:
        occupied_frac = df[count_col].fillna(0).gt(0).mean()
        msg.append(f"occupied_frac={occupied_frac:>7.2%}")

    print(" | ".join(msg))


def trim_to_month(df, name, ts_col="timestamp", month_start=MARCH_START, month_end=MARCH_END):
    if ts_col not in df.columns:
        raise KeyError(f"{name}: timestamp column '{ts_col}' not found.")
    mask = (df[ts_col] >= month_start) & (df[ts_col] <= month_end)
    trimmed = df.loc[mask].copy()
    dropped = len(df) - len(trimmed)
    if dropped > 0:
        print(f"{name:<22} | trimmed_outside_month_rows={dropped}")
    return trimmed


def validate_datasets():
    print("=" * 70)
    print("STAGE 2: INDEPENDENT SOURCE VALIDATION (MARCH 2015 MVP)")
    print("=" * 70)

    print("Loading OMNI...")
    df_omni = pd.read_parquet(
        "data/raw/omni/2015/omni_1min_2015-03-01_to_2015-03-31.parquet"
    )

    print("Loading SuperMAG (ABK)...")
    df_smag = pd.read_parquet(
        "data/raw/supermag/2015/supermag_ABK_2015-03-01_to_2015-03-31.parquet"
    )

    print("Loading GOES-15...")
    df_goes = pd.read_parquet(
        "data/processed/aligned_1min/2015/03/goes15_1min_201503.parquet"
    )

    print("Loading Swarm A, B, C raw...")
    df_swarm_a = pd.read_parquet("data/raw/swarm/2015/03/swarmA_LR1B_201503.parquet")
    df_swarm_b = pd.read_parquet("data/raw/swarm/2015/03/swarmB_LR1B_201503.parquet")
    df_swarm_c = pd.read_parquet("data/raw/swarm/2015/03/swarmC_LR1B_201503.parquet")

    print("Loading aligned LEO index...")
    df_leo = pd.read_parquet(
        "data/processed/aligned_1min/2015/03/leo_index_global_201503.parquet"
    )

    df_omni = ensure_utc_timestamp(df_omni)
    df_smag = ensure_utc_timestamp(df_smag)
    df_goes = ensure_utc_timestamp(df_goes)
    df_swarm_a = ensure_utc_timestamp(df_swarm_a)
    df_swarm_b = ensure_utc_timestamp(df_swarm_b)
    df_swarm_c = ensure_utc_timestamp(df_swarm_c)
    df_leo = ensure_utc_timestamp(df_leo)

    print("\n--- DUPLICATE TIMESTAMP REPORT ---")
    for name, df in [
        ("OMNI (L1)", df_omni),
        ("SuperMAG (ABK)", df_smag),
        ("GOES-15 (GEO)", df_goes),
        ("Swarm A raw (LEO)", df_swarm_a),
        ("Swarm B raw (LEO)", df_swarm_b),
        ("Swarm C raw (LEO)", df_swarm_c),
        ("LEO index (aligned)", df_leo),
    ]:
        report_duplicate_timestamps(df, name)

    print("\n--- OMNI TIME RANGE CHECK ---")
    report_time_range(df_omni, "OMNI (L1)", window_start=MARCH_START, window_end=MARCH_END)

    # Trim OMNI to strict March window if the retrieval included a boundary row.
    df_omni = trim_to_month(df_omni, "OMNI (L1)", month_start=MARCH_START, month_end=MARCH_END)

    print("\n--- MANIFEST-STYLE VALIDATION REPORT ---")
    omni_bz_col = find_col(df_omni, ["BZ_GSM", "Bz_GSM", "BZGSM"], "OMNI")
    smag_dbdt_col = find_col(
        df_smag, ["dbdt_magnitude", "dBdt_magnitude", "dbdt_mag"], "SuperMAG"
    )
    goes_bz_col = find_col(
        df_goes, ["B_Z_GSM", "BZ_GSM", "Bz_GSM", "goes_bz_gsm"], "GOES-15"
    )

    summarize_series_missing(df_omni, "OMNI (L1)", omni_bz_col, rows_expected=MARCH_2015_ROWS)
    summarize_series_missing(df_smag, "SuperMAG (ABK)", smag_dbdt_col, rows_expected=MARCH_2015_ROWS)
    summarize_series_missing(df_goes, "GOES-15 (GEO)", goes_bz_col, rows_expected=MARCH_2015_ROWS)

    summarize_swarm_raw(df_swarm_a, "Swarm A raw (LEO)")
    summarize_swarm_raw(df_swarm_b, "Swarm B raw (LEO)")
    summarize_swarm_raw(df_swarm_c, "Swarm C raw (LEO)")

    summarize_leo_index(df_leo, "LEO index (aligned)", rows_expected=MARCH_2015_ROWS)

    print("\n--- GENERATING PHYSICS VALIDATION PLOT ---")
    df_omni_i = set_ts_index(df_omni.drop_duplicates(subset=["timestamp"], keep="first"))
    df_smag_i = set_ts_index(df_smag.drop_duplicates(subset=["timestamp"], keep="first"))
    df_goes_i = set_ts_index(df_goes.drop_duplicates(subset=["timestamp"], keep="first"))
    df_swarm_a_i = set_ts_index(df_swarm_a.drop_duplicates(subset=["timestamp"], keep="first"))
    df_leo_i = set_ts_index(df_leo.drop_duplicates(subset=["timestamp"], keep="first"))

    start_plot = "2015-03-16"
    end_plot = "2015-03-20"

    omni_slice = df_omni_i.loc[start_plot:end_plot]
    smag_slice = df_smag_i.loc[start_plot:end_plot]
    goes_slice = df_goes_i.loc[start_plot:end_plot]
    leo_slice = df_leo_i.loc[start_plot:end_plot]
    swarm_a_slice = df_swarm_a_i.loc[start_plot:end_plot].copy()

    symh_col = find_col(omni_slice.reset_index(), ["SYM_H", "SYMH", "sym_h"], "OMNI")
    vx_col = find_col(omni_slice.reset_index(), ["Vx", "VX_GSE", "vx"], "OMNI")
    leo_idx_col = find_col(
        leo_slice.reset_index(), ["leo_index_global", "leoindexglobal"], "LEO index"
    )

    swarm_f_col = None
    for c in ["F", "f"]:
        if c in swarm_a_slice.columns:
            swarm_f_col = c
            break

    if "QDLat" not in swarm_a_slice.columns:
        raise KeyError("Swarm A plot requires QDLat column.")

    swarm_plot = swarm_a_slice.iloc[::60].copy()

    fig, axes = plt.subplots(
        6, 1, figsize=(12, 14), sharex=True, gridspec_kw={"hspace": 0.15}
    )
    fig.suptitle(
        "St. Patrick's Day Storm (March 17, 2015) - Full MVP Validation",
        fontsize=14
    )

    axes[0].plot(omni_slice.index, omni_slice[omni_bz_col], label="IMF Bz (GSM)", color="red", alpha=0.75)
    axes[0].set_ylabel("Bz (nT)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8)

    ax0_twin = axes[0].twinx()
    ax0_twin.plot(omni_slice.index, omni_slice[symh_col], label="SYM-H", color="blue", linewidth=1.8)
    ax0_twin.set_ylabel("SYM-H (nT)", color="blue")
    ax0_twin.legend(loc="upper right")

    axes[1].plot(omni_slice.index, omni_slice[vx_col].abs(), label="Solar Wind Speed |Vx|", color="orange")
    axes[1].set_ylabel("Speed (km/s)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(goes_slice.index, goes_slice[goes_bz_col], label="GOES-15 Bz (GSM)", color="purple")
    axes[2].set_ylabel("GEO Bz (nT)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(smag_slice.index, smag_slice[smag_dbdt_col], label="ABK dB/dt Mag", color="black")
    axes[3].set_ylabel("dB/dt (nT/min)")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(leo_slice.index, leo_slice[leo_idx_col], label="Tier 1 LEO Index", color="teal", linewidth=1.5)
    axes[4].set_ylabel("LEO idx (nT)")
    axes[4].legend(loc="upper left")
    axes[4].grid(True, alpha=0.3)

    if "leo_count" in leo_slice.columns:
        ax4_twin = axes[4].twinx()
        ax4_twin.plot(leo_slice.index, leo_slice["leo_count"], color="gray", alpha=0.35, linewidth=1.0)
        ax4_twin.set_ylabel("Count", color="gray")
    elif "leocount" in leo_slice.columns:
        ax4_twin = axes[4].twinx()
        ax4_twin.plot(leo_slice.index, leo_slice["leocount"], color="gray", alpha=0.35, linewidth=1.0)
        ax4_twin.set_ylabel("Count", color="gray")

    if swarm_f_col is not None and swarm_a_slice[swarm_f_col].notna().any():
        sc = axes[5].scatter(
            swarm_plot.index,
            swarm_plot["QDLat"],
            c=swarm_plot[swarm_f_col],
            cmap="viridis",
            s=5,
            alpha=0.85,
            label=f"Swarm A ({swarm_f_col})"
        )
        plt.colorbar(sc, ax=axes[5], label=f"Mag Field |{swarm_f_col}| (nT)", pad=0.01)
    else:
        axes[5].scatter(
            swarm_plot.index,
            swarm_plot["QDLat"],
            color="green",
            s=5,
            alpha=0.85,
            label="Swarm A QDLat"
        )

    axes[5].set_ylabel("QD Latitude (deg)")
    axes[5].legend(loc="upper left")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = "results/plots/01_validation_st_patricks_2015_v2.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Validation plot saved to {plot_path}")


if __name__ == "__main__":
    validate_datasets()
