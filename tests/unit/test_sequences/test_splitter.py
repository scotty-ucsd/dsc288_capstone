"""Chronological split boundaries, buffer, and no cross-split windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.sequences import builder  # noqa: E402


def test_month_boundaries_respect_split_buffer() -> None:
    """SPLIT_BUFFER_DAYS respected between train/val and val/test."""
    from swmi.utils import config

    b = builder._month_boundaries()
    val_start = pd.Timestamp(config.VAL_START, tz="UTC")
    test_start = pd.Timestamp(config.TEST_START, tz="UTC")
    buf = pd.Timedelta(days=config.SPLIT_BUFFER_DAYS)
    assert b["val_start"] == val_start + buf
    assert b["test_start"] == test_start + buf


def test_row_split_respects_chronology() -> None:
    ts = pd.date_range("2014-12-01", "2024-12-01", freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})
    labels = builder._row_split(df["timestamp"])
    train_mask = labels == "train"
    test_mask = labels == "test"
    if train_mask.any() and test_mask.any():
        assert df.loc[train_mask, "timestamp"].max() < df.loc[test_mask, "timestamp"].min()


def test_valid_sequence_starts_rejects_mixed_split_window() -> None:
    """Window must be entirely inside one split (no leakage across label change)."""
    n = 500
    ts = pd.date_range("2015-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "omni_bz_gsm": np.random.default_rng(1).standard_normal(n),
        }
    )
    labels = builder._row_split(df["timestamp"])
    y = np.zeros((n, 1), dtype=np.float32)
    starts = builder._valid_sequence_starts(
        df,
        labels,
        y,
        ["omni_bz_gsm"],
        input_window_min=10,
        forecast_horizon_min=60,
        max_gap_fraction=0.5,
        stride_min=1,
    )
    for group in ("train", "val", "test"):
        for s in starts.get(group, []):
            w = labels.iloc[s : s + 10]
            assert w.nunique() == 1
            tgt = s + 10 - 1 + 60
            assert labels.iloc[s] == labels.iloc[tgt]
