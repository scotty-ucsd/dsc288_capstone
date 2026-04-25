"""Rolling feature window sizes (L1: 10/30 m; GOES mag: 60/120 m)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.features.builder import (  # noqa: E402
    GOES_MAG_ROLLING_WINDOWS_MIN,
    _add_rolling_features,
    add_goes_features,
)


def _minute_df(n: int) -> pd.DataFrame:
    ts = pd.date_range("2015-03-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "omni_bz_gsm": np.linspace(-5.0, 5.0, n),
            "omni_vx": -400.0 * np.ones(n),
            "newell_phi": np.ones(n),
        }
    )


def test_rolling_l1_windows_10_and_30_minutes() -> None:
    df = _add_rolling_features(_minute_df(40))
    assert "omni_bz_gsm_mean_10m" in df.columns
    assert "omni_bz_gsm_std_10m" in df.columns
    assert "omni_bz_gsm_valid_points_10m" in df.columns
    assert "omni_bz_gsm_mean_30m" in df.columns
    # Window 10: at index 9, mean should equal first 10 raw values' mean
    expect = float(df["omni_bz_gsm"].iloc[:10].mean())
    assert np.isclose(df["omni_bz_gsm_mean_10m"].iloc[9], expect, rtol=1e-5)


def test_goes_rolling_uses_60_and_120_minute_windows() -> None:
    assert set(GOES_MAG_ROLLING_WINDOWS_MIN) == {60, 120}
    n = 200
    ts = pd.date_range("2015-03-01", periods=n, freq="1min", tz="UTC")
    g = pd.DataFrame({"timestamp": ts, "goes_bz_gsm": np.random.default_rng(0).standard_normal(n)})
    out = add_goes_features(g)
    assert "goes_bz_gsm_mean_60m" in out.columns
    assert "goes_bz_gsm_max_120m" in out.columns
