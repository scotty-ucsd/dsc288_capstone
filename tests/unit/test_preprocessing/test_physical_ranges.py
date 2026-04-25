"""P0-F2 physical range validation on monthly source tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.preprocessing.validation import validate_physical_ranges  # noqa: E402


def test_omni_passes_plausible_l1() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=n, freq="1min", tz="UTC"),
            "BZ_GSM": np.linspace(-20, 20, n),
            "Vx": -400.0 * np.ones(n),
            "Vy": np.zeros(n),
            "Vz": np.zeros(n),
        }
    )
    validate_physical_ranges(df, "omni")


def test_omni_raises_on_extreme_bz() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC"),
            "BZ_GSM": [0.0, 150.0, 0.0],
            "Vx": [-400.0, -400.0, -400.0],
        }
    )
    with pytest.raises(ValueError, match="Bz"):
        validate_physical_ranges(df, "omni")


def test_supermag_raises_on_extreme_dbdt_horizontal() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC"),
            "station": ["ABK"] * 3,
            "dbdt_horizontal_magnitude": [0.0, 20000.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="dbdt_horizontal_magnitude"):
        validate_physical_ranges(df, "supermag")
