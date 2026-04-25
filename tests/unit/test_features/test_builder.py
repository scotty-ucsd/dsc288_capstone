"""Tests for scoped feature-builder interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.features.builder import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    _prepare_supermag_targets,
    _transform_partition,
    add_goes_features,
    add_xray_features,
)


def test_add_goes_features_preserves_source_fields_and_adds_rolling_state() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=121, freq="1min", tz="UTC"),
            "goes_bz_gsm": list(range(121)),
            "goes_source_satellite": ["GOES-15"] * 121,
            "goes_mag_missing_flag": [0] * 121,
        }
    )

    out = add_goes_features(df, min_valid_fraction=0.5)

    assert "goes_source_satellite" in out.columns
    assert "goes_mag_missing_flag" in out.columns
    assert "goes_bz_gsm_mean_60m" in out.columns
    assert "goes_bz_gsm_std_120m" in out.columns
    assert "goes_bz_gsm_min_60m" in out.columns
    assert "goes_bz_gsm_max_120m" in out.columns
    assert out.loc[59, "goes_bz_gsm_mean_60m"] == pytest.approx(29.5)
    assert out.loc[119, "goes_bz_gsm_valid_points_120m"] == pytest.approx(120.0)


def test_add_goes_features_requires_canonical_bz() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2015-01-01", periods=2, freq="1min", tz="UTC")})

    with pytest.raises(KeyError, match="goes_bz_gsm"):
        add_goes_features(df)


def test_add_xray_features_are_event_driven_and_preserve_quality_metadata() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-01-01", periods=5, freq="1min", tz="UTC"),
            "xrsb_flux": [1e-7, 2e-6, 2e-5, 2e-4, 1e-7],
            "goes_xray_long_log": [-7.0, -5.7, -4.7, -3.7, -7.0],
            "xray_quality_flags": ["a=0;b=0"] * 5,
            "xray_source_satellite": ["GOES-15"] * 5,
            "xray_missing_flag": [0] * 5,
        }
    )

    out = add_xray_features(df)

    assert "xray_quality_flags" in out.columns
    assert "xray_source_satellite" in out.columns
    assert "goes_xray_long_dlog_dt" in out.columns
    assert "goes_xray_time_since_last_c_flare" in out.columns
    assert "goes_xray_time_since_last_m_flare" in out.columns
    assert "goes_xray_time_since_last_x_flare" in out.columns
    assert "goes_xray_cumulative_m_class_24h" in out.columns
    assert "goes_xray_max_flux_24h" in out.columns
    assert not any(col.startswith("goes_xray_long_mean_") for col in out.columns)
    assert out.loc[1, "goes_xray_time_since_last_c_flare"] == pytest.approx(0.0)
    assert out.loc[4, "goes_xray_time_since_last_m_flare"] == pytest.approx(1.0)
    assert out.loc[4, "goes_xray_cumulative_m_class_24h"] == pytest.approx(2.0)
    assert out.loc[4, "goes_xray_max_flux_24h"] == pytest.approx(2e-4)


def test_add_xray_features_can_use_normalized_long_log() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="1min", tz="UTC"),
            "goes_xray_long_normalized": [-1.0, 0.0],
            "xray_source_satellite": ["GOES-16", "GOES-16"],
        }
    )

    out = add_xray_features(df)

    assert list(out["goes_xray_long_log"]) == [-1.0, 0.0]
    assert "xray_missing_flag" in out.columns


def test_prepare_supermag_targets_emits_canonical_station_outputs() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2015-03-01T00:00:00Z",
                    "2015-03-01T00:00:00Z",
                    "2015-03-01T00:01:00Z",
                    "2015-03-01T00:01:00Z",
                ]
            ),
            "station": ["abk", "TRO", "ABK", "TRO"],
            "dbdt_horizontal_magnitude": [1.0, 2.0, 3.0, 4.0],
            "dbdt_gap_flag": [0, 0, 1, 0],
            "n_nez": [10.0, 20.0, 30.0, 40.0],
            "e_nez": [1.0, 2.0, 3.0, 4.0],
            "mlt": [12.0, 13.0, 12.1, 13.1],
            "mlat": [65.0, 66.0, 65.0, 66.0],
            "glat": [68.0, 69.0, 68.0, 69.0],
        }
    )

    out = _prepare_supermag_targets(df)

    assert "dbdt_horizontal_magnitude_ABK" in out.columns
    assert "dbdt_horizontal_magnitude_TRO" in out.columns
    assert "dbdt_missing_flag_ABK" in out.columns
    assert "mlt_ABK" in out.columns
    assert "mlat_TRO" in out.columns
    assert "n_nez_ABK" not in out.columns
    assert "e_nez_TRO" not in out.columns
    assert out.loc[1, "dbdt_missing_flag_ABK"] == 1
    assert pd.isna(out.loc[1, "dbdt_horizontal_magnitude_ABK"])
    assert out.loc[1, "dbdt_horizontal_magnitude_TRO"] == pytest.approx(4.0)


def test_prepare_supermag_targets_accepts_legacy_magnitude_alias() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=2, freq="1min", tz="UTC"),
            "station": ["ABK", "ABK"],
            "dbdt_magnitude": [None, 0.25],
            "dbdt_missing_flag": [1, 0],
        }
    )

    out = _prepare_supermag_targets(df)

    assert "dbdt_horizontal_magnitude_ABK" in out.columns
    assert pd.isna(out.loc[0, "dbdt_horizontal_magnitude_ABK"])
    assert out.loc[0, "dbdt_missing_flag_ABK"] == 1
    assert out.loc[1, "dbdt_horizontal_magnitude_ABK"] == pytest.approx(0.25)


def test_transform_partition_preserves_station_flags_and_schema_version() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC"),
            "omni_bz_gsm": [-5.0, -4.0, -3.0],
            "omni_by_gsm": [1.0, 2.0, 3.0],
            "omni_vx": [-400.0, -410.0, -420.0],
            "dbdt_horizontal_magnitude_ABK": [0.1, None, 0.3],
            "dbdt_missing_flag_ABK": [0, None, 0],
        }
    )

    out = _transform_partition(df)

    assert out.loc[1, "dbdt_missing_flag_ABK"] == 1
    assert out["feature_schema_version"].eq(FEATURE_SCHEMA_VERSION).all()
    assert "newell_phi" in out.columns
