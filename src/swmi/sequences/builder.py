"""Build multi-station sequence tensors from monthly feature matrices.

P0-S6 contract:
- ``X`` has shape ``(samples, timesteps, features)`` and contains global
  timestamp-level features only.
- ``y`` has shape ``(samples, stations)`` with NaN where a station target is
  unavailable or masked.
- Station context is saved separately as ``station_context`` with shape
  ``(samples, stations, context_features)``.
- Feature windows end at time ``T``; targets are taken at
  ``T + FORECAST_HORIZON_MIN``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from swmi.utils import config
from swmi.utils.logger import get_logger

log = get_logger(__name__)

INPUT_WINDOW_MIN = 120
TARGET_PREFIX = "dbdt_horizontal_magnitude_"
TARGET_MISSING_PREFIX = "dbdt_missing_flag_"
STATION_CONTEXT_FIELDS = ("mlt", "mlat", "mlon", "glat", "glon")
SCALE_EXCLUDE_SUFFIXES = (
    "_missing_flag",
    "_missing",
    "_ffill_applied",
    "_valid_points_10m",
    "_valid_points_30m",
    "_valid_points_60m",
    "_valid_points_120m",
)
SCALE_EXCLUDE_COLUMNS = {
    "ut_sin",
    "ut_cos",
    "doy_sin",
    "doy_cos",
    "year",
    "month",
}


@dataclass(frozen=True)
class SequenceBuildResult:
    """Summary of one sequence build run."""

    output_paths: dict[str, Path]
    feature_columns: list[str]
    target_columns: list[str]
    stations: list[str]
    split_counts: dict[str, int]


def _month_boundaries() -> dict[str, pd.Timestamp]:
    train_end = pd.Timestamp(config.TRAIN_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    val_end = pd.Timestamp(config.VAL_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    test_end = pd.Timestamp(config.TEST_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    buffer_days = pd.Timedelta(days=config.SPLIT_BUFFER_DAYS)
    val_start = pd.Timestamp(config.VAL_START, tz="UTC")
    test_start = pd.Timestamp(config.TEST_START, tz="UTC")
    return {
        "train_start": pd.Timestamp(config.TRAIN_START, tz="UTC"),
        "train_end": train_end,
        "val_start": val_start + buffer_days,
        "val_end": val_end,
        "test_start": test_start + buffer_days,
        "test_end": test_end,
        "buffer_after_train_start": val_start,
        "buffer_after_train_end": val_start + buffer_days - pd.Timedelta(minutes=1),
        "buffer_after_val_start": test_start,
        "buffer_after_val_end": test_start + buffer_days - pd.Timedelta(minutes=1),
    }


def _row_split(timestamps: pd.Series, boundaries: dict[str, pd.Timestamp] | None = None) -> pd.Series:
    """Assign each row to train, val, test, or buffer."""
    b = _month_boundaries() if boundaries is None else boundaries
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    labels = pd.Series("buffer", index=timestamps.index, dtype="string")
    labels[(ts >= b["train_start"]) & (ts <= b["train_end"])] = "train"
    labels[(ts >= b["val_start"]) & (ts <= b["val_end"])] = "val"
    labels[(ts >= b["test_start"]) & (ts <= b["test_end"])] = "test"
    return labels.astype(str)


def _station_from_target_col(column: str) -> str:
    if not column.startswith(TARGET_PREFIX):
        raise ValueError(f"Not a canonical target column: {column}")
    return column.removeprefix(TARGET_PREFIX)


def _target_columns(df: pd.DataFrame) -> list[str]:
    columns = sorted(col for col in df.columns if col.startswith(TARGET_PREFIX))
    if not columns:
        raise ValueError(f"No target columns found with prefix {TARGET_PREFIX!r}. Run P0-S5 first.")
    return columns


def _is_station_context_col(column: str, stations: list[str]) -> bool:
    return any(column == f"{field}_{station}" for field in STATION_CONTEXT_FIELDS for station in stations)


def _feature_columns(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    stations = [_station_from_target_col(col) for col in target_cols]
    excluded = {"timestamp", "feature_schema_version", "year", "month", "_split"}
    excluded.update(target_cols)
    excluded.update(col for col in df.columns if col.startswith(TARGET_MISSING_PREFIX))
    excluded.update(col for col in df.columns if _is_station_context_col(col, stations))

    feature_cols: list[str] = []
    dropped_non_numeric: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            dropped_non_numeric.append(col)
            continue
        feature_cols.append(col)

    if dropped_non_numeric:
        log.warning("Dropping non-numeric sequence features: %s", dropped_non_numeric)
    if not feature_cols:
        raise ValueError("No numeric feature columns remain after excluding targets and metadata.")
    return feature_cols


def _is_scalable(column: str) -> bool:
    if column in SCALE_EXCLUDE_COLUMNS:
        return False
    return not any(column.endswith(suffix) for suffix in SCALE_EXCLUDE_SUFFIXES)


def _load_feature_matrices(features_dir: str | Path) -> pd.DataFrame:
    paths = sorted(Path(features_dir).glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature matrix Parquet files found under {features_dir}")

    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        raise KeyError("Feature matrix is missing required 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Feature matrix contains null or unparseable timestamps.")
    if df["timestamp"].duplicated().any():
        examples = df.loc[df["timestamp"].duplicated(keep=False), "timestamp"].head(5).tolist()
        raise ValueError(f"Feature matrix contains duplicate timestamps. Examples: {examples}")
    return df.sort_values("timestamp").reset_index(drop=True)


def _fit_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: pd.Series,
    scaler_dir: str | Path | None,
) -> tuple[StandardScaler, list[str]]:
    train_rows = splits == "train"
    if not train_rows.any():
        raise ValueError("Cannot fit scaler: no training rows available.")

    train_max = df.loc[train_rows, "timestamp"].max()
    val_start = pd.Timestamp(config.VAL_START, tz="UTC")
    if train_max >= val_start:
        raise ValueError(f"Scaler fit rows leak into validation period: max train timestamp={train_max}")

    cols_to_scale = [col for col in feature_cols if _is_scalable(col)]
    scaler = StandardScaler()
    scaler.fit(df.loc[train_rows, cols_to_scale].fillna(0.0).to_numpy())

    if scaler_dir is not None:
        output_dir = Path(scaler_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, output_dir / f"scaler_v{config.SCALER_VERSION}.pkl")
        metadata = {
            "version": config.SCALER_VERSION,
            "fit_start": config.TRAIN_START,
            "fit_end": config.TRAIN_END,
            "n_rows_fit": int(train_rows.sum()),
            "feature_columns": feature_cols,
            "scaled_columns": cols_to_scale,
            "reference_field_model": config.REFERENCE_FIELD,
        }
        with open(output_dir / f"scaler_v{config.SCALER_VERSION}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

    return scaler, cols_to_scale


def _scaled_feature_array(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    cols_to_scale: list[str],
) -> np.ndarray:
    features = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32).copy()
    scale_indices = [feature_cols.index(col) for col in cols_to_scale]
    if scale_indices:
        scaled_values = scaler.transform(df[cols_to_scale].fillna(0.0).to_numpy()).astype(np.float32)
        features[:, scale_indices] = scaled_values
    return features


def _target_array(df: pd.DataFrame, target_cols: list[str]) -> np.ndarray:
    y = df[target_cols].to_numpy(dtype=np.float32).copy()
    stations = [_station_from_target_col(col) for col in target_cols]
    for idx, station in enumerate(stations):
        missing_col = f"{TARGET_MISSING_PREFIX}{station}"
        if missing_col in df.columns:
            missing = pd.to_numeric(df[missing_col], errors="coerce").fillna(1).to_numpy(dtype=bool)
            y[missing, idx] = np.nan
    return y


def _context_columns(stations: list[str], df: pd.DataFrame) -> list[str]:
    return [field for field in STATION_CONTEXT_FIELDS if any(f"{field}_{station}" in df.columns for station in stations)]


def _station_context_at_rows(
    df: pd.DataFrame,
    row_indices: np.ndarray,
    stations: list[str],
    context_fields: list[str],
) -> np.ndarray:
    context = np.full((len(row_indices), len(stations), len(context_fields)), np.nan, dtype=np.float32)
    for station_idx, station in enumerate(stations):
        for context_idx, field in enumerate(context_fields):
            col = f"{field}_{station}"
            if col in df.columns:
                context[:, station_idx, context_idx] = pd.to_numeric(
                    df.iloc[row_indices][col],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
    return context


def _row_has_feature_gap(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].isna().any(axis=1).to_numpy(dtype=bool)


def _valid_sequence_starts(
    df: pd.DataFrame,
    splits: pd.Series,
    y_values: np.ndarray,
    feature_cols: list[str],
    *,
    input_window_min: int,
    forecast_horizon_min: int,
    max_gap_fraction: float,
    stride_min: int,
) -> dict[str, list[int]]:
    if not 0.0 <= max_gap_fraction <= 1.0:
        raise ValueError(f"max_gap_fraction must be in [0, 1], got {max_gap_fraction}")
    if input_window_min <= 0 or forecast_horizon_min <= 0 or stride_min <= 0:
        raise ValueError("input_window_min, forecast_horizon_min, and stride_min must be positive.")

    row_gap = _row_has_feature_gap(df, feature_cols)
    starts_by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    n_rows = len(df)
    max_start = n_rows - input_window_min - forecast_horizon_min + 1
    for start in range(0, max(0, max_start), stride_min):
        end = start + input_window_min - 1
        target_idx = end + forecast_horizon_min
        window_splits = splits.iloc[start : end + 1]
        split = window_splits.iloc[0]
        if split not in starts_by_split:
            continue
        if window_splits.nunique() != 1 or splits.iloc[target_idx] != split:
            continue
        if row_gap[start : end + 1].mean() > max_gap_fraction:
            continue
        if np.isnan(y_values[target_idx]).all():
            continue
        starts_by_split[split].append(start)
    return starts_by_split


def audit_leakage(
    feature_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    *,
    forecast_horizon_min: int = config.FORECAST_HORIZON_MIN,
) -> None:
    """Fail hard if feature and target timestamps violate forecast timing."""
    if feature_timestamps.ndim != 2:
        raise ValueError("feature_timestamps must have shape (samples, timesteps).")
    if target_timestamps.ndim != 1:
        raise ValueError("target_timestamps must have shape (samples,).")
    if feature_timestamps.shape[0] != target_timestamps.shape[0]:
        raise ValueError("feature_timestamps and target_timestamps sample counts differ.")
    if feature_timestamps.shape[0] == 0:
        return
    if np.issubdtype(feature_timestamps.dtype, np.number) or np.issubdtype(target_timestamps.dtype, np.number):
        raise ValueError("Leakage audit requires timestamp arrays; numeric timestamp fallback is forbidden.")

    feature_df = pd.DataFrame(feature_timestamps)
    feature_ts = feature_df.apply(pd.to_datetime, utc=True, errors="coerce")
    target_ts = pd.to_datetime(pd.Series(target_timestamps), utc=True, errors="coerce")
    if feature_ts.isna().any().any() or target_ts.isna().any():
        raise ValueError("Leakage audit requires parseable timestamp arrays; no numeric fallback is allowed.")

    non_monotonic = feature_ts.diff(axis=1).iloc[:, 1:].le(pd.Timedelta(0)).any(axis=1)
    if non_monotonic.any():
        first = int(np.flatnonzero(non_monotonic.to_numpy())[0])
        raise ValueError(f"Feature timestamps are not strictly increasing in sequence {first}.")

    cadence_bad = feature_ts.diff(axis=1).iloc[:, 1:].ne(pd.Timedelta(minutes=1)).any(axis=1)
    if cadence_bad.any():
        first = int(np.flatnonzero(cadence_bad.to_numpy())[0])
        raise ValueError(f"Feature timestamps are not at 1-minute cadence in sequence {first}.")

    last_feature_ts = feature_ts.iloc[:, -1]
    expected_target_ts = last_feature_ts + pd.Timedelta(minutes=forecast_horizon_min)
    mismatched = target_ts.reset_index(drop=True) != expected_target_ts.reset_index(drop=True)
    if mismatched.any():
        first = int(np.flatnonzero(mismatched.to_numpy())[0])
        raise ValueError(
            "Anti-leakage violation: target timestamp does not equal "
            f"T+{forecast_horizon_min} min for sequence {first}."
        )


def _build_split_arrays(
    df: pd.DataFrame,
    starts: list[int],
    feature_values: np.ndarray,
    y_values: np.ndarray,
    stations: list[str],
    context_fields: list[str],
    *,
    input_window_min: int,
    forecast_horizon_min: int,
) -> dict[str, np.ndarray]:
    starts_arr = np.asarray(starts, dtype=np.int64)
    n_samples = len(starts)
    n_features = feature_values.shape[1]

    X = np.empty((n_samples, input_window_min, n_features), dtype=np.float32)
    y = np.empty((n_samples, len(stations)), dtype=np.float32)
    feature_ts = np.empty((n_samples, input_window_min), dtype="datetime64[ns]")
    target_ts = np.empty(n_samples, dtype="datetime64[ns]")
    end_indices = starts_arr + input_window_min - 1
    target_indices = end_indices + forecast_horizon_min

    ts_values = df["timestamp"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    for sample_idx, start in enumerate(starts_arr):
        end = start + input_window_min
        target_idx = start + input_window_min - 1 + forecast_horizon_min
        X[sample_idx] = feature_values[start:end]
        y[sample_idx] = y_values[target_idx]
        feature_ts[sample_idx] = ts_values[start:end]
        target_ts[sample_idx] = ts_values[target_idx]

    audit_leakage(feature_ts, target_ts, forecast_horizon_min=forecast_horizon_min)

    target_mask = ~np.isnan(y)
    current_y = y_values[end_indices].astype(np.float32).copy()
    current_target_mask = ~np.isnan(current_y)
    station_context = _station_context_at_rows(df, end_indices, stations, context_fields)
    return {
        "X": X,
        "y": y,
        "target_mask": target_mask,
        "current_y": current_y,
        "current_target_mask": current_target_mask,
        "station_context": station_context,
        "sequence_start_times": ts_values[starts_arr],
        "sequence_end_times": ts_values[end_indices],
        "target_times": ts_values[target_indices],
    }


def build_sequences(
    features_dir: str | Path = config.FEATURES_DIR,
    output_dir: str | Path = config.SEQUENCES_DIR,
    *,
    scaler_dir: str | Path | None = config.ARTIFACTS_DIR,
    input_window_min: int = INPUT_WINDOW_MIN,
    forecast_horizon_min: int = config.FORECAST_HORIZON_MIN,
    max_gap_fraction: float = 0.10,
    stride_min: int = 1,
) -> SequenceBuildResult:
    """Build train/val/test NPZ sequence files from feature matrices."""
    df = _load_feature_matrices(features_dir)
    target_cols = _target_columns(df)
    stations = [_station_from_target_col(col) for col in target_cols]
    feature_cols = _feature_columns(df, target_cols)
    context_fields = _context_columns(stations, df)
    splits = _row_split(df["timestamp"])

    scaler, cols_to_scale = _fit_scaler(df, feature_cols, splits, scaler_dir)
    feature_values = _scaled_feature_array(df, feature_cols, scaler, cols_to_scale)
    y_values = _target_array(df, target_cols)
    starts_by_split = _valid_sequence_starts(
        df,
        splits,
        y_values,
        feature_cols,
        input_window_min=input_window_min,
        forecast_horizon_min=forecast_horizon_min,
        max_gap_fraction=max_gap_fraction,
        stride_min=stride_min,
    )

    output_root = Path(output_dir)
    output_paths: dict[str, Path] = {}
    split_counts: dict[str, int] = {}
    for split, starts in starts_by_split.items():
        split_counts[split] = len(starts)
        if not starts:
            log.warning("No valid %s sequences produced.", split)
            continue
        arrays = _build_split_arrays(
            df,
            starts,
            feature_values,
            y_values,
            stations,
            context_fields,
            input_window_min=input_window_min,
            forecast_horizon_min=forecast_horizon_min,
        )
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / f"sequences_{split}.npz"
        np.savez_compressed(
            output_path,
            **arrays,
            stations=np.asarray(stations, dtype="U"),
            feature_columns=np.asarray(feature_cols, dtype="U"),
            target_columns=np.asarray(target_cols, dtype="U"),
            station_context_columns=np.asarray(context_fields, dtype="U"),
            forecast_horizon_min=np.asarray(forecast_horizon_min, dtype=np.int16),
            input_window_min=np.asarray(input_window_min, dtype=np.int16),
        )
        output_paths[split] = output_path
        log.info("Wrote %s sequences: X=%s y=%s -> %s", split, arrays["X"].shape, arrays["y"].shape, output_path)

    if not output_paths:
        raise ValueError("No sequence files were written; all splits had zero valid sequences.")

    return SequenceBuildResult(
        output_paths=output_paths,
        feature_columns=feature_cols,
        target_columns=target_cols,
        stations=stations,
        split_counts=split_counts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build multi-station SWMI sequences.")
    parser.add_argument("--features-dir", default=config.FEATURES_DIR)
    parser.add_argument("--output-dir", default=config.SEQUENCES_DIR)
    parser.add_argument("--scaler-dir", default=config.ARTIFACTS_DIR)
    parser.add_argument("--input-window-min", type=int, default=INPUT_WINDOW_MIN)
    parser.add_argument("--forecast-horizon-min", type=int, default=config.FORECAST_HORIZON_MIN)
    parser.add_argument("--max-gap-fraction", type=float, default=0.10)
    parser.add_argument("--stride-min", type=int, default=1)
    args = parser.parse_args(argv)

    build_sequences(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        scaler_dir=args.scaler_dir,
        input_window_min=args.input_window_min,
        forecast_horizon_min=args.forecast_horizon_min,
        max_gap_fraction=args.max_gap_fraction,
        stride_min=args.stride_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
