"""Cyclical encodings from UT hour and day-of-year — bounds in [-1, 1]."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.features.builder import _add_cyclical_features  # noqa: E402


def test_cyclical_ut_doy_in_unit_interval() -> None:
    ts = pd.date_range("2015-06-15 12:00", periods=5, freq="1h", tz="UTC")
    df = _add_cyclical_features(pd.DataFrame({"timestamp": ts}))
    for col in ("ut_sin", "ut_cos", "doy_sin", "doy_cos"):
        s = df[col].dropna()
        assert s.min() >= -1.0001
        assert s.max() <= 1.0001
