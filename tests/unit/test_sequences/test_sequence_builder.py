"""Tests for multi-station sequence generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.sequences.builder import (  # noqa: E402
    _feature_columns,
    _target_columns,
    _valid_sequence_starts,
    audit_leakage,
    build_sequences,
)


def _feature_df(periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "omni_bz_gsm": np.linspace(-5.0, 5.0, periods),
            "newell_phi": np.linspace(1.0, 2.0, periods),
            "ut_sin": np.sin(np.arange(periods)),
            "ut_cos": np.cos(np.arange(periods)),
            "feature_schema_version": ["feature_schema_v1"] * periods,
            "dbdt_horizontal_magnitude_ABK": np.linspace(0.0, 1.0, periods),
            "dbdt_horizontal_magnitude_TRO": np.linspace(1.0, 2.0, periods),
            "dbdt_missing_flag_ABK": [0] * periods,
            "dbdt_missing_flag_TRO": [0] * periods,
            "mlt_ABK": [12.0] * periods,
            "mlt_TRO": [13.0] * periods,
            "mlat_ABK": [65.0] * periods,
            "mlat_TRO": [66.0] * periods,
        }
    )


def test_build_sequences_writes_multi_output_npz(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "sequences"
    scaler_dir = tmp_path / "artifacts"
    features_dir.mkdir()
    df = _feature_df()
    # First sequence with window=5, horizon=2 targets row 6.
    df.loc[6, "dbdt_missing_flag_TRO"] = 1
    df.to_parquet(features_dir / "features_202001.parquet", index=False)

    result = build_sequences(
        features_dir=features_dir,
        output_dir=output_dir,
        scaler_dir=scaler_dir,
        input_window_min=5,
        forecast_horizon_min=2,
        stride_min=10,
    )

    assert result.stations == ["ABK", "TRO"]
    assert result.split_counts["train"] > 0
    train_path = output_dir / "train" / "sequences_train.npz"
    assert train_path.exists()
    assert (scaler_dir / "scaler_v1.pkl").exists()

    data = np.load(train_path)
    assert data["X"].shape[1] == 5
    assert data["y"].shape[1] == 2
    assert data["target_mask"].shape == data["y"].shape
    assert np.isnan(data["y"][0, 1])
    assert not data["target_mask"][0, 1]
    assert data["station_context"].shape[:2] == data["y"].shape
    assert list(data["stations"]) == ["ABK", "TRO"]


def test_feature_columns_exclude_targets_masks_and_station_context() -> None:
    df = _feature_df(periods=10)
    targets = _target_columns(df)

    features = _feature_columns(df, targets)

    assert "omni_bz_gsm" in features
    assert "dbdt_horizontal_magnitude_ABK" not in features
    assert "dbdt_missing_flag_ABK" not in features
    assert "mlt_ABK" not in features
    assert "feature_schema_version" not in features


def test_valid_sequence_starts_reject_excess_feature_gaps() -> None:
    df = _feature_df(periods=12)
    df.loc[0:2, "omni_bz_gsm"] = np.nan
    targets = _target_columns(df)
    features = _feature_columns(df, targets)
    y = df[targets].to_numpy(dtype=np.float32)
    splits = pd.Series(["train"] * len(df), index=df.index)

    starts = _valid_sequence_starts(
        df,
        splits,
        y,
        features,
        input_window_min=5,
        forecast_horizon_min=1,
        max_gap_fraction=0.10,
        stride_min=1,
    )

    assert 0 not in starts["train"]
    assert 3 in starts["train"]


def test_audit_leakage_rejects_numeric_timestamp_fallback() -> None:
    with pytest.raises(ValueError, match="numeric timestamp fallback"):
        audit_leakage(np.array([[1, 2, 3]]), np.array([4]), forecast_horizon_min=1)


def test_audit_leakage_rejects_wrong_horizon() -> None:
    feature_ts = np.array(
        [[
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T00:01"),
        ]]
    )
    target_ts = np.array([np.datetime64("2020-01-01T00:05")])

    with pytest.raises(ValueError, match="target timestamp"):
        audit_leakage(feature_ts, target_ts, forecast_horizon_min=1)
