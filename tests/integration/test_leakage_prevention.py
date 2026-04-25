"""Temporal leakage checks: splits, forecast horizon, audit_leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from swmi.sequences.builder import _month_boundaries, _row_split, audit_leakage  # noqa: E402


def test_test_timestamps_after_val_plus_buffer() -> None:
    b = _month_boundaries()
    test_ts = pd.Timestamp("2023-08-01", tz="UTC")
    val_ts = pd.Timestamp("2022-12-01", tz="UTC")
    assert test_ts > val_ts
    assert test_ts >= b["test_start"]
    assert val_ts <= b["val_end"]


def test_audit_leakage_accepts_aligned_target_time() -> None:
    feat = np.array(
        [[np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T00:01")]]
    )
    target = np.array([np.datetime64("2020-01-01T01:01")])  # T + 60 min
    audit_leakage(feat, target, forecast_horizon_min=60)


def test_row_labels_align_with_buffer() -> None:
    idx = pd.date_range("2021-12-20", "2022-01-20", freq="1h", tz="UTC")
    labels = _row_split(pd.Series(idx, index=range(len(idx))))
    assert (labels == "buffer").any()
