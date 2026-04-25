"""Tests for LEO index validation diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.evaluation.leo_index_validation import (  # noqa: E402
    add_global_supermag_target,
    lag_correlation_table,
    run_leo_index_validation,
    station_correlation_table,
)


def _feature_df(periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2015-03-01", periods=periods, freq="1min", tz="UTC")
    leo = np.sin(np.linspace(0.0, 8.0 * np.pi, periods))
    target = np.roll(leo, 2)
    target[:2] = np.nan
    return pd.DataFrame(
        {
            "timestamp": ts,
            "leo_high_lat": leo,
            "leo_mid_lat": leo[::-1],
            "dbdt_horizontal_magnitude_ABK": target,
            "dbdt_horizontal_magnitude_TRO": target * 2.0,
            "glat_ABK": [68.0] * periods,
            "glon_ABK": [20.0] * periods,
            "mlat_ABK": [65.0] * periods,
            "mlon_ABK": [110.0] * periods,
            "glat_TRO": [69.0] * periods,
            "glon_TRO": [19.0] * periods,
            "mlat_TRO": [66.0] * periods,
            "mlon_TRO": [111.0] * periods,
        }
    )


def test_lag_correlation_identifies_leading_leo_signal() -> None:
    df = add_global_supermag_target(_feature_df())
    table = lag_correlation_table(df, lags_min=[0, 1, 2, 3])

    best = table.iloc[0]

    assert best["leo_feature"] == "leo_high_lat"
    assert best["lag_min"] == 2
    assert best["variance_explained"] > 0.99


def test_station_correlation_includes_spatial_context() -> None:
    df = add_global_supermag_target(_feature_df())
    table = station_correlation_table(df, "leo_high_lat", 2)

    assert table["station"].tolist() == ["ABK", "TRO"]
    assert table.set_index("station").loc["ABK", "glat"] == 68.0
    assert table["correlation"].min() > 0.99


def test_run_leo_index_validation_writes_outputs(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "validation"
    features_dir.mkdir()
    _feature_df().to_parquet(features_dir / "features_201503.parquet", index=False)

    result = run_leo_index_validation(
        features_dir=features_dir,
        output_dir=output_dir,
        lags_min=[0, 1, 2, 3],
    )

    assert result.best_lag_min == 2
    assert result.variance_explained > 0.99
    assert result.dmsp_recommendation == "defer DMSP"
    assert (output_dir / "leo_lag_correlations.csv").exists()
    assert (output_dir / "leo_station_correlations.csv").exists()
    assert (output_dir / "leo_lag_scan.png").exists()
    assert (output_dir / "leo_station_spatial_correlations.png").exists()
    assert (output_dir / "leo_index_validation_report.md").exists()
