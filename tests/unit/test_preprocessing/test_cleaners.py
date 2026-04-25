"""Tests for preprocessing cleaners."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.preprocessing.cleaners import normalize_goes_xray  # noqa: E402


def _xray_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-01-01", periods=6, freq="1min", tz="UTC"),
            "xrsa_flux": [1e-8, 1e-7, 1e-6, 1e-8, 1e-7, 1e-6],
            "xrsb_flux": [1e-6, 1e-5, 1e-4, 1e-6, 1e-5, 1e-4],
            "xray_quality_flags": [
                "a=0;b=0",
                "a=0;b=0",
                "a=0;b=0",
                "a=0;b=0;electron=0",
                "a=0;b=0;electron=0",
                "a=0;b=0;electron=0",
            ],
            "xray_source_satellite": [
                "GOES-15",
                "GOES-15",
                "GOES-15",
                "GOES-16",
                "GOES-16",
                "GOES-16",
            ],
            "xray_missing_flag": [0, 0, 0, 0, 0, 0],
        }
    )


def test_normalize_goes_xray_log_baseline_and_scale_factor() -> None:
    normalized = normalize_goes_xray(
        _xray_df(),
        write_output=False,
        write_validation_plot=False,
        baseline_quantile=0.0,
        scale_factors={"GOES-15": 10.0, "GOES-16": 1.0},
    )

    assert "goes_xray_long_log" in normalized.columns
    assert normalized.loc[0, "goes_xray_long_log"] == pytest.approx(-6.0)
    assert normalized.loc[2, "goes_xray_long_normalized"] == pytest.approx(3.0)
    assert normalized.loc[5, "goes_xray_long_normalized"] == pytest.approx(2.0)
    assert int(normalized["xray_normalized_missing_flag"].sum()) == 0


def test_normalize_goes_xray_filters_quality_and_physical_ranges() -> None:
    df = _xray_df().iloc[:4].copy()
    df.loc[0, "xray_quality_flags"] = "a=1;b=0"
    df.loc[1, "xray_quality_flags"] = "a=0;b=0;electron=8"
    df.loc[2, "xrsa_flux"] = -9999.0
    df.loc[3, "xrsb_flux"] = 0.3

    normalized = normalize_goes_xray(
        df,
        write_output=False,
        write_validation_plot=False,
        baseline_quantile=0.0,
    )

    assert pd.isna(normalized.loc[0, "goes_xray_short_log"])
    assert normalized.loc[0, "goes_xray_long_log"] == pytest.approx(-6.0)
    assert pd.isna(normalized.loc[1, "goes_xray_short_log"])
    assert pd.isna(normalized.loc[1, "goes_xray_long_log"])
    assert pd.isna(normalized.loc[2, "goes_xray_short_log"])
    assert pd.isna(normalized.loc[3, "goes_xray_long_log"])


def test_normalize_goes_xray_writes_output_and_validates_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "goes_xray_normalized_201501.parquet"

    normalized = normalize_goes_xray(
        _xray_df(),
        output_path=output_path,
        write_output=True,
        write_validation_plot=False,
    )

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert len(saved) == len(normalized)
    assert "goes_xray_short_normalized" in saved.columns
    assert "goes_xray_long_normalized" in saved.columns


def test_normalize_goes_xray_rejects_duplicate_timestamps() -> None:
    df = _xray_df()
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        normalize_goes_xray(df, write_output=False, write_validation_plot=False)


def test_normalize_goes_xray_requires_output_path_for_dataframe_write() -> None:
    with pytest.raises(ValueError, match="output_path is required"):
        normalize_goes_xray(_xray_df(), write_output=True, write_validation_plot=False)
