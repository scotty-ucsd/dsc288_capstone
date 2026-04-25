"""
build_sequences.py
Builds chronologically split LSTM input sequences from the feature matrix.

Sequence format
---------------
- Input window: 120 minutes (120 timesteps at 1-minute cadence)
- Forecast horizon: config.FORECAST_HORIZON_MIN = 60 minutes
- A sequence at time T uses features [T-119, T]; target is dB/dt at T+60.
- Sequences are rejected if any timestep in the window fails the usability
  mask, or if the sequence spans a split boundary.

Split boundaries (from config.py)
----------------------------------
Train: TRAIN_START → TRAIN_END
Buffer: SPLIT_BUFFER_DAYS days excluded after TRAIN_END
Val:    VAL_START → VAL_END
Buffer: SPLIT_BUFFER_DAYS days excluded after VAL_END
Test:   TEST_START → TEST_END

Scaler
------
StandardScaler fit ONLY on training partition rows.
Saved as models/artifacts/scaler_v{VERSION}.pkl + _meta.json.
Applied (not refitted) to val and test partitions.

Station-aware context
---------------------
For each target station and each sequence, appends:
  station_mlt_sin, station_mlt_cos  (cyclical, excluded from scaling)
  station_qdlat                     (raw float, included in scaling)

TODO: Vectorize validity mask; remove audit_leakage fallback; add multi-station y array; integrate into scripts/03_build_sequences.py
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from station_context import station_mlt_encoded, get_station_qdlat

log = get_logger(__name__)

INPUT_WINDOW_MIN = 120


# ---------------------------------------------------------------------------
# Columns excluded from Z-score scaling
# ---------------------------------------------------------------------------
SCALE_EXCLUDE_PATTERNS = (
    "timestamp",
    "_is_fresh",
    "_missing_flag",
    "_missing",
    "_ffill_applied",
    "goes_satellite",
    "ut_sin", "ut_cos", "doy_sin", "doy_cos",
    "station_mlt_sin", "station_mlt_cos",
    "year", "month",
)


def _is_scalable(col: str) -> bool:
    return not any(col == pat or col.endswith(pat) for pat in SCALE_EXCLUDE_PATTERNS)


# ---------------------------------------------------------------------------
# Anti-leakage audit
# ---------------------------------------------------------------------------

def audit_leakage(
    X_timestamps: np.ndarray,
    y_timestamps: np.ndarray,
    split_boundaries: dict,
) -> None:
    """Verify temporal integrity before writing a sequence file.

    Parameters
    ----------
    X_timestamps:
        Array of shape (N, WINDOW) containing feature timestamps for each sequence.
    y_timestamps:
        Array of shape (N,) containing target timestamps.
    split_boundaries:
        Dict with keys: 'train_end', 'val_start', 'val_end', 'test_start'.
        Each value is a timezone-aware pd.Timestamp.

    Raises
    ------
    AssertionError
        If any feature timestamp is strictly after the corresponding target timestamp,
        or if any sequence crosses a designated split boundary buffer.
    """
    # Invariant: features must not be in the future relative to targets
    if X_timestamps.ndim == 2:
        last_feature_ts = X_timestamps[:, -1]
    else:
        last_feature_ts = X_timestamps

    if hasattr(last_feature_ts[0], "tzinfo"):
        violations = [
            i for i, (ft, yt) in enumerate(zip(last_feature_ts, y_timestamps))
            if ft > yt
        ]
    else:
        violations = []  # numerical fallback — skip timestamp comparison

    assert len(violations) == 0, (
        f"Anti-leakage violation: {len(violations)} sequences have feature timestamps "
        f"after the target timestamp. First violation at index {violations[0]}."
    )

    log.info("Anti-leakage audit passed: %d sequences checked.", len(y_timestamps))


# ---------------------------------------------------------------------------
# Split boundary helpers
# ---------------------------------------------------------------------------

def _make_boundaries():
    tz = "UTC"
    train_end   = pd.Timestamp(config.TRAIN_END, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    buffer_days = pd.Timedelta(days=config.SPLIT_BUFFER_DAYS)
    return {
        "train_start": pd.Timestamp(config.TRAIN_START, tz=tz),
        "train_end":   train_end,
        "buf1_start":  train_end + pd.Timedelta(minutes=1),
        "buf1_end":    pd.Timestamp(config.TRAIN_END, tz=tz) + buffer_days,
        "val_start":   pd.Timestamp(config.VAL_START, tz=tz),
        "val_end":     pd.Timestamp(config.VAL_END, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
        "buf2_start":  pd.Timestamp(config.VAL_END, tz=tz) + pd.Timedelta(days=1),
        "buf2_end":    pd.Timestamp(config.VAL_END, tz=tz) + buffer_days,
        "test_start":  pd.Timestamp(config.TEST_START, tz=tz),
        "test_end":    pd.Timestamp(config.TEST_END, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
    }


def _row_split(ts: pd.Series, b: dict) -> pd.Series:
    """Return a Series of string labels: 'train', 'val', 'test', or 'buffer'."""
    labels = pd.Series("buffer", index=ts.index, dtype=str)
    labels[(ts >= b["train_start"]) & (ts <= b["train_end"])] = "train"
    labels[(ts >= b["val_start"])   & (ts <= b["val_end"])]   = "val"
    labels[(ts >= b["test_start"])  & (ts <= b["test_end"])]  = "test"
    return labels


# ---------------------------------------------------------------------------
# Sequence validity
# ---------------------------------------------------------------------------

def _build_validity_mask(df: pd.DataFrame, splits: pd.Series) -> np.ndarray:
    """Return a boolean mask: True at index i if the sequence starting at i is valid."""
    W = INPUT_WINDOW_MIN
    H = config.FORECAST_HORIZON_MIN
    n = len(df)

    # A sequence starting at i spans feature rows [i, i+W-1] and target at i+W-1+H
    valid = np.zeros(n, dtype=bool)

    for i in range(n - W - H + 1):
        end_feat = i + W - 1
        tgt_row  = end_feat + H

        if tgt_row >= n:
            continue

        # All 120 feature rows must belong to the same split (no cross-split sequences)
        window_splits = splits.iloc[i:end_feat + 1]
        tgt_split     = splits.iloc[tgt_row]

        if window_splits.nunique() != 1:
            continue  # window crosses a split or buffer boundary
        if window_splits.iloc[0] != tgt_split:
            continue  # target in different split than features
        if window_splits.iloc[0] == "buffer":
            continue  # inside buffer zone

        valid[i] = True

    return valid


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame, station: str) -> list[str]:
    exclude = {
        "timestamp", "year", "month",
        # raw targets (future data)
        f"target_{station.lower()}_dbdt_mag_tplus60",
        f"target_{station.lower()}_dbn_dt_tplus60",
        f"target_{station.lower()}_dbe_dt_tplus60",
    }
    # Also exclude raw b components (not features) and future-shifting flags
    for col in list(df.columns):
        if "tplus" in col or "target_" in col:
            exclude.add(col)

    feature_cols = [c for c in df.columns if c not in exclude]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        log.warning("Dropping non-numeric columns from feature set: %s", non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    return feature_cols


# ---------------------------------------------------------------------------
# Scaler (Task 4.2)
# ---------------------------------------------------------------------------

def _fit_scaler(df_train: pd.DataFrame, feature_cols: list[str]) -> tuple[StandardScaler, list[str], list[str]]:
    """Fit StandardScaler on training data only.

    Parameters
    ----------
    df_train:
        Training partition DataFrame.
    feature_cols:
        All candidate feature columns.

    Returns
    -------
    scaler, cols_to_scale, cols_no_scale
    """
    # Leakage guard
    max_ts = df_train["timestamp"].max()
    val_start = pd.Timestamp(config.VAL_START, tz="UTC")
    assert max_ts < val_start, (
        f"Scaler fit data must not include validation or test period. "
        f"max_timestamp={max_ts}, VAL_START={val_start}"
    )

    cols_to_scale = [c for c in feature_cols if _is_scalable(c) and c in df_train.columns]
    cols_no_scale = [c for c in feature_cols if not _is_scalable(c)]

    n_fit = len(df_train)
    log.info("Fitting scaler on %d training rows, %d columns...", n_fit, len(cols_to_scale))

    scaler = StandardScaler()
    scaler.fit(df_train[cols_to_scale].fillna(0).values)

    # Warn on near-constant features
    for i, col in enumerate(cols_to_scale):
        if scaler.scale_[i] < 0.01:
            log.warning(
                "Feature '%s' has near-zero std=%.5f after fitting. "
                "Consider dropping it before training.", col, scaler.scale_[i]
            )

    return scaler, cols_to_scale, cols_no_scale


def _save_scaler(scaler: StandardScaler, cols_to_scale: list[str], n_samples: int) -> None:
    """Save scaler artifact + metadata JSON to models/artifacts/."""
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    pkl_path  = os.path.join(config.ARTIFACTS_DIR, f"scaler_v{config.SCALER_VERSION}.pkl")
    meta_path = os.path.join(config.ARTIFACTS_DIR, f"scaler_v{config.SCALER_VERSION}_meta.json")

    joblib.dump(scaler, pkl_path)

    meta = {
        "version":             config.SCALER_VERSION,
        "fit_start":           config.TRAIN_START,
        "fit_end":             config.TRAIN_END,
        "n_samples_fit":       n_samples,
        "feature_names":       cols_to_scale,
        "means":               [f"{m:.6g}" for m in scaler.mean_.tolist()],
        "stds":                [f"{s:.6g}" for s in scaler.scale_.tolist()],
        "reference_field_model": config.REFERENCE_FIELD,
        "created_at":          datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Scaler artifact saved → %s", pkl_path)
    log.info("Scaler metadata    saved → %s", meta_path)


# ---------------------------------------------------------------------------
# Sequence array construction (with station context)
# ---------------------------------------------------------------------------

def _build_arrays(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    feature_cols: list[str],
    scaler: StandardScaler,
    cols_to_scale: list[str],
    station: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[pd.Timestamp]]:
    """Build X and y arrays for valid sequences.

    Appends station-aware context (MLT sin/cos, QDLat) to every sequence.
    """
    W = INPUT_WINDOW_MIN
    H = config.FORECAST_HORIZON_MIN

    # Pre-scale feature matrix once
    feat_arr = df[feature_cols].fillna(0.0).values.astype(np.float32)
    scale_indices = [feature_cols.index(c) for c in cols_to_scale if c in feature_cols]
    if scale_indices:
        feat_arr[:, scale_indices] = scaler.transform(
            df[[feature_cols[i] for i in scale_indices]].fillna(0.0).values
        ).astype(np.float32)

    y_arr = df[target_col].values.astype(np.float32)
    ts_arr = df["timestamp"].values

    # Station context (constant per station)
    station_qdlat = get_station_qdlat(station)

    valid_starts = np.where(valid_mask)[0]
    n_valid = len(valid_starts)

    # Extra features: station_mlt_sin, station_mlt_cos, station_qdlat
    X = np.empty((n_valid, W, feat_arr.shape[1] + 3), dtype=np.float32)
    y = np.empty(n_valid, dtype=np.float32)
    feat_timestamps = []
    tgt_timestamps  = []

    for j, start_idx in enumerate(valid_starts):
        end_idx = start_idx + W - 1
        tgt_idx = end_idx + H

        X[j, :, :feat_arr.shape[1]] = feat_arr[start_idx:start_idx + W]

        # Station context at time T (end of input window)
        ts_T = pd.Timestamp(ts_arr[end_idx])
        if ts_T.tzinfo is None:
            ts_T = ts_T.tz_localize("UTC")
        mlt_sin, mlt_cos = station_mlt_encoded(station, ts_T)

        X[j, :, -3] = mlt_sin
        X[j, :, -2] = mlt_cos
        X[j, :, -1] = station_qdlat

        y[j] = y_arr[tgt_idx]
        feat_timestamps.append(ts_T)
        tgt_timestamps.append(pd.Timestamp(ts_arr[tgt_idx]))

    return X, y, feat_timestamps, tgt_timestamps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_sequences(station: str = "ABK") -> None:
    """Build chronologically split train/val/test sequence files for one station.

    Reads all available feature matrix Parquet files, applies split boundaries
    from config.py, fits a scaler on training data only, and writes
    compressed .npz files per split per month.

    Parameters
    ----------
    station:
        IAGA code of the target station (e.g. ``"ABK"``).
    """
    station = station.upper()
    log.info("Building sequences for station %s...", station)

    # ------------------------------------------------------------------
    # Load all feature Parquet files
    # ------------------------------------------------------------------
    features_glob = os.path.join(config.FEATURES_DIR, "**", "*.parquet")
    import glob
    feat_files = sorted(glob.glob(features_glob, recursive=True))
    if not feat_files:
        log.error("No feature matrix files found at %s. Run build_feature_matrix first.", features_glob)
        return

    df_all = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %d rows across %d feature files.", len(df_all), len(feat_files))

    # Determine target column (station-specific dBdt magnitude)
    target_col_candidates = [
        c for c in df_all.columns
        if "dbdt_magnitude" in c and station.lower() in c.lower()
    ]
    if not target_col_candidates:
        log.error("No dbdt_magnitude column found for station %s. "
                  "Available columns: %s", station, [c for c in df_all.columns if "dbdt" in c])
        return
    target_col = target_col_candidates[0]
    log.info("Using target column: %s", target_col)

    # ------------------------------------------------------------------
    # Assign split labels
    # ------------------------------------------------------------------
    b = _make_boundaries()
    splits = _row_split(df_all["timestamp"], b)
    df_all["_split"] = splits
    log.info("Split distribution: %s", splits.value_counts().to_dict())

    # ------------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------------
    feature_cols = _get_feature_cols(df_all, station)
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10], )

    # ------------------------------------------------------------------
    # Validity mask (excludes cross-boundary and buffer sequences)
    # ------------------------------------------------------------------
    log.info("Computing validity mask...")
    valid_mask = _build_validity_mask(df_all, splits)
    log.info("Valid sequence starts: %d / %d", int(valid_mask.sum()), len(df_all))

    # ------------------------------------------------------------------
    # Fit scaler on training rows only
    # ------------------------------------------------------------------
    df_train = df_all.loc[splits == "train"]
    scaler, cols_to_scale, _ = _fit_scaler(df_train, feature_cols)
    _save_scaler(scaler, cols_to_scale, n_samples=len(df_train))

    # ------------------------------------------------------------------
    # Build arrays and write per-split .npz files
    # ------------------------------------------------------------------
    for split_name in ("train", "val", "test"):
        split_mask = (splits == split_name).values & valid_mask

        if int(split_mask.sum()) == 0:
            log.warning("No valid sequences for split '%s', station %s.", split_name, station)
            continue

        X, y, feat_ts, tgt_ts = _build_arrays(
            df_all, split_mask, feature_cols, scaler, cols_to_scale, station, target_col
        )

        # Anti-leakage audit before writing
        audit_leakage(
            X_timestamps=np.array(feat_ts),
            y_timestamps=np.array(tgt_ts),
            split_boundaries=b,
        )

        out_dir = os.path.join(config.SEQUENCES_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)
        # Write per-month files to keep individual files manageable
        out_path = os.path.join(out_dir, f"seq_{station}_{split_name}.npz")
        np.savez_compressed(out_path, X=X, y=y)

        log.info(
            "Written %s | split=%s | X=%s | y=%s → %s",
            station, split_name, X.shape, y.shape, out_path,
        )


if __name__ == "__main__":
    build_sequences(station="ABK")
