"""Tests for the fused dataset mode in run_pipeline.py and swmi.features.fused."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swmi.features.fused import (
    add_source_completeness_flags,
    build_fused_month,
    generate_fused_summary_report,
    select_columns_for_format,
    _add_month_cyclical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline_script():
    """Load run_pipeline.py via importlib (it lives under scripts/utils/)."""
    path = ROOT / "scripts" / "utils" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_feature_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a minimal feature matrix DataFrame for testing."""
    ts = pd.date_range("2015-03-01", periods=n_rows, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": ts,
        # OMNI columns
        "omni_bz_gsm": rng.normal(0, 5, n_rows),
        "omni_by_gsm": rng.normal(0, 3, n_rows),
        "omni_vx": rng.normal(-400, 50, n_rows),
        "omni_proton_density": rng.normal(5, 2, n_rows),
        "omni_pressure": rng.normal(2, 1, n_rows),
        "omni_bx_gse": rng.normal(0, 3, n_rows),
        "omni_f": rng.normal(5, 2, n_rows),
        "omni_sym_h": rng.normal(-10, 20, n_rows),
        "omni_al": rng.normal(-100, 50, n_rows),
        "omni_au": rng.normal(50, 30, n_rows),
        # GOES columns
        "goes_bz_gsm": rng.normal(0, 3, n_rows),
        "goes_source_satellite": "GOES-16",
        "goes_mag_missing_flag": np.int8(0),
        "goes_xray_long_log": rng.normal(-6, 1, n_rows),
        "goes_xray_long_dlog_dt": rng.normal(0, 0.01, n_rows),
        "goes_xray_time_since_last_c_flare": rng.uniform(0, 1000, n_rows),
        "goes_xray_time_since_last_m_flare": rng.uniform(0, 5000, n_rows),
        "goes_xray_time_since_last_x_flare": rng.uniform(0, 50000, n_rows),
        "goes_xray_cumulative_m_class_24h": rng.integers(0, 5, n_rows).astype(float),
        "goes_xray_max_flux_24h": rng.uniform(1e-7, 1e-4, n_rows),
        "xray_source_satellite": "GOES-16",
        "xray_missing_flag": np.int8(0),
        # LEO columns
        "leo_high_lat": rng.normal(20, 10, n_rows),
        "leo_mid_lat": rng.normal(10, 5, n_rows),
        "leo_dayside": rng.normal(15, 8, n_rows),
        "leo_nightside": rng.normal(25, 12, n_rows),
        "leo_high_lat_decay_age": np.float64(0),
        "leo_high_lat_is_fresh": np.int8(1),
        "leo_high_lat_count": 5,
        # SuperMAG targets
        "dbdt_horizontal_magnitude_ABK": rng.exponential(5, n_rows),
        "dbdt_horizontal_magnitude_TRO": rng.exponential(5, n_rows),
        "dbdt_missing_flag_ABK": np.int8(0),
        "dbdt_missing_flag_TRO": np.int8(0),
        # Station context
        "mlt_ABK": rng.uniform(0, 24, n_rows),
        "mlat_ABK": 66.0,
        # Newell coupling
        "newell_phi": rng.exponential(1000, n_rows),
        # Cyclical
        "ut_sin": np.sin(np.linspace(0, 2 * np.pi, n_rows)),
        "ut_cos": np.cos(np.linspace(0, 2 * np.pi, n_rows)),
        "doy_sin": np.sin(np.linspace(0, 2 * np.pi, n_rows)),
        "doy_cos": np.cos(np.linspace(0, 2 * np.pi, n_rows)),
        # Rolling (quality flags)
        "omni_bz_gsm_mean_10m": rng.normal(0, 3, n_rows),
        "omni_bz_gsm_valid_points_10m": np.float32(10),
        "goes_bz_gsm_valid_points_60m": np.float32(60),
        # Gap flags
        "l1_any_missing": np.int8(0),
        "geo_any_missing": np.int8(0),
        "omni_bz_gsm_missing": np.int8(0),
        "omni_vx_ffill_applied": np.int8(0),
        # Metadata
        "year": 2015,
        "month": 3,
        "feature_schema_version": "feature_schema_v1",
    })


def _write_feature_matrix(tmp_path: Path, year: int = 2015, month: int = 3) -> Path:
    """Write a fake feature matrix Parquet file in the canonical location."""
    features_dir = tmp_path / "data" / "processed" / "features" / f"{year:04d}" / f"{month:02d}"
    features_dir.mkdir(parents=True, exist_ok=True)
    feat_path = features_dir / f"features_{year:04d}{month:02d}.parquet"
    df = _make_feature_df()
    df.to_parquet(feat_path, index=False)
    return feat_path


# ---------------------------------------------------------------------------
# Test: select_columns_for_format
# ---------------------------------------------------------------------------

class TestSelectColumnsForFormat:

    def test_debug_includes_all_columns(self):
        df = _make_feature_df()
        result = select_columns_for_format(df, "debug")
        assert list(result.columns) == list(df.columns)

    def test_eda_excludes_metadata(self):
        df = _make_feature_df()
        result = select_columns_for_format(df, "eda")
        assert "year" not in result.columns
        assert "month" not in result.columns
        assert "feature_schema_version" not in result.columns
        # EDA still includes quality flags and station context.
        assert "goes_mag_missing_flag" in result.columns
        assert "mlt_ABK" in result.columns
        assert "goes_source_satellite" in result.columns

    def test_minimal_excludes_metadata_and_flags(self):
        df = _make_feature_df()
        result = select_columns_for_format(df, "minimal")
        # Metadata excluded.
        assert "year" not in result.columns
        assert "feature_schema_version" not in result.columns
        # Quality flags excluded.
        assert "goes_mag_missing_flag" not in result.columns
        assert "omni_bz_gsm_valid_points_10m" not in result.columns
        assert "omni_vx_ffill_applied" not in result.columns
        assert "leo_high_lat_decay_age" not in result.columns
        assert "leo_high_lat_is_fresh" not in result.columns
        assert "leo_high_lat_count" not in result.columns
        # Station context excluded.
        assert "mlt_ABK" not in result.columns
        assert "mlat_ABK" not in result.columns
        # Source satellite excluded.
        assert "goes_source_satellite" not in result.columns
        assert "xray_source_satellite" not in result.columns
        # Core features still included.
        assert "timestamp" in result.columns
        assert "omni_bz_gsm" in result.columns
        assert "newell_phi" in result.columns
        assert "dbdt_horizontal_magnitude_ABK" in result.columns
        assert "ut_sin" in result.columns

    def test_invalid_format_raises(self):
        df = _make_feature_df()
        with pytest.raises(ValueError, match="format_mode"):
            select_columns_for_format(df, "invalid_format")

    def test_source_filtering_omni_only(self):
        df = _make_feature_df()
        result = select_columns_for_format(df, "debug", sources=["omni"])
        # OMNI and source-agnostic columns present.
        assert "omni_bz_gsm" in result.columns
        assert "timestamp" in result.columns
        assert "ut_sin" in result.columns
        # GOES and SuperMAG columns excluded.
        assert "goes_bz_gsm" not in result.columns
        assert "dbdt_horizontal_magnitude_ABK" not in result.columns
        assert "leo_high_lat" not in result.columns

    def test_source_filtering_multiple(self):
        df = _make_feature_df()
        result = select_columns_for_format(df, "debug", sources=["omni", "goes"])
        assert "omni_bz_gsm" in result.columns
        assert "goes_bz_gsm" in result.columns
        assert "dbdt_horizontal_magnitude_ABK" not in result.columns


# ---------------------------------------------------------------------------
# Test: add_source_completeness_flags
# ---------------------------------------------------------------------------

class TestSourceCompletenessFlags:

    def test_completeness_flags_computed_correctly(self):
        df = _make_feature_df()
        result = add_source_completeness_flags(df)

        assert "source_completeness_omni" in result.columns
        assert "source_completeness_goes" in result.columns
        assert "source_completeness_swarm" in result.columns
        assert "source_completeness_supermag" in result.columns
        assert "any_source_missing" in result.columns

        # All data is present, so completeness should be 1.0 for all rows.
        assert (result["source_completeness_omni"] == 1.0).all()
        assert (result["source_completeness_goes"] == 1.0).all()
        assert (result["source_completeness_swarm"] == 1.0).all()
        assert (result["source_completeness_supermag"] == 1.0).all()
        assert (result["any_source_missing"] == 0).all()

    def test_completeness_flags_with_missing_data(self):
        df = _make_feature_df()
        # Inject NaN into OMNI.
        df.loc[0:9, "omni_bz_gsm"] = np.nan
        df.loc[0:9, "omni_vx"] = np.nan

        result = add_source_completeness_flags(df)

        # First 10 rows should have 0 OMNI completeness.
        assert (result["source_completeness_omni"].iloc[:10] == 0.0).all()
        # And those rows should flag any_source_missing.
        assert (result["any_source_missing"].iloc[:10] == 1).all()
        # Remaining rows should be fine.
        assert (result["source_completeness_omni"].iloc[10:] == 1.0).all()

    def test_completeness_flags_empty_indicator_columns(self):
        """When indicator columns are entirely absent, completeness is 0."""
        df = _make_feature_df()
        df = df.drop(columns=["omni_bz_gsm", "omni_vx"])
        result = add_source_completeness_flags(df)
        assert (result["source_completeness_omni"] == 0.0).all()


# ---------------------------------------------------------------------------
# Test: build_fused_month
# ---------------------------------------------------------------------------

class TestBuildFusedMonth:

    def test_fused_pipeline_writes_monthly_parquet(self, tmp_path, monkeypatch):
        """Verify that build_fused_month writes output to the correct path."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        _write_feature_matrix(tmp_path)

        output_dir = str(tmp_path / "data" / "fused")
        result = build_fused_month(2015, 3, output_dir=output_dir, format_mode="eda")

        assert result is not None
        assert result.exists()
        assert result.name == "fused_201503.parquet"

        # Verify the output is readable and has expected columns.
        df = pd.read_parquet(result)
        assert "timestamp" in df.columns
        assert "source_completeness_omni" in df.columns
        assert "sin_month" in df.columns

    def test_fused_pipeline_dry_run_does_not_write(self, tmp_path, monkeypatch):
        """Verify dry-run prints plan without writing files."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        _write_feature_matrix(tmp_path)

        output_dir = str(tmp_path / "data" / "fused")
        result = build_fused_month(2015, 3, output_dir=output_dir, dry_run=True)

        assert result is None
        fused_path = Path(output_dir) / "2015" / "03" / "fused_201503.parquet"
        assert not fused_path.exists()

    def test_fused_pipeline_skips_existing_unless_force(self, tmp_path, monkeypatch):
        """Verify idempotency: existing output is skipped unless --force."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        _write_feature_matrix(tmp_path)

        output_dir = str(tmp_path / "data" / "fused")

        # First run: creates file.
        result1 = build_fused_month(2015, 3, output_dir=output_dir)
        assert result1 is not None

        # Second run: skips (returns existing path).
        result2 = build_fused_month(2015, 3, output_dir=output_dir)
        assert result2 is not None
        assert result2 == result1

        # Force run: overwrites.
        result3 = build_fused_month(2015, 3, output_dir=output_dir, force=True)
        assert result3 is not None

    def test_fused_pipeline_raises_on_missing_features(self, tmp_path, monkeypatch):
        """Verify clear error when feature matrix doesn't exist."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        output_dir = str(tmp_path / "data" / "fused")

        with pytest.raises(FileNotFoundError, match="Feature matrix not found"):
            build_fused_month(2015, 3, output_dir=output_dir)

    def test_fused_format_minimal_output(self, tmp_path, monkeypatch):
        """Verify minimal format produces fewer columns than eda."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        _write_feature_matrix(tmp_path)

        output_dir_eda = str(tmp_path / "data" / "fused_eda")
        output_dir_min = str(tmp_path / "data" / "fused_min")

        build_fused_month(2015, 3, output_dir=output_dir_eda, format_mode="eda")
        build_fused_month(2015, 3, output_dir=output_dir_min, format_mode="minimal")

        df_eda = pd.read_parquet(Path(output_dir_eda) / "2015" / "03" / "fused_201503.parquet")
        df_min = pd.read_parquet(Path(output_dir_min) / "2015" / "03" / "fused_201503.parquet")

        assert len(df_min.columns) < len(df_eda.columns)
        # Both should have the same number of rows.
        assert len(df_min) == len(df_eda)


# ---------------------------------------------------------------------------
# Test: generate_fused_summary_report
# ---------------------------------------------------------------------------

class TestFusedSummaryReport:

    def test_summary_report_generated(self, tmp_path, monkeypatch):
        """Verify the summary report is written and contains expected keys."""
        monkeypatch.setattr("swmi.utils.config.FEATURES_DIR", str(tmp_path / "data" / "processed" / "features"))
        _write_feature_matrix(tmp_path)

        output_dir = str(tmp_path / "data" / "fused")
        build_fused_month(2015, 3, output_dir=output_dir)

        report = generate_fused_summary_report(
            output_dir,
            [(2015, 3, True)],
        )

        assert report["total_months"] == 1
        assert report["succeeded"] == 1
        assert report["failed"] == 0
        assert len(report["months"]) == 1
        assert report["months"][0]["success"] is True
        assert report["months"][0]["rows"] > 0
        assert report["months"][0]["station_count"] == 2  # ABK, TRO

        # Verify JSON was written.
        report_path = Path(output_dir) / "fused_summary_report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as fh:
            disk_report = json.load(fh)
        assert disk_report["total_months"] == 1


# ---------------------------------------------------------------------------
# Test: month cyclical encoding
# ---------------------------------------------------------------------------

class TestMonthCyclical:

    def test_month_cyclical_added(self):
        df = _make_feature_df()
        assert "sin_month" not in df.columns
        result = _add_month_cyclical(df)
        assert "sin_month" in result.columns
        assert "cos_month" in result.columns
        # March = month 3 → sin(3 * 2π / 12) = sin(π/2) = 1.0
        assert result["sin_month"].iloc[0] == pytest.approx(np.sin(3 * 2 * np.pi / 12), abs=1e-5)

    def test_month_cyclical_not_duplicated(self):
        """If sin_month/cos_month already exist, don't overwrite."""
        df = _make_feature_df()
        df["sin_month"] = 0.999
        df["cos_month"] = -0.999
        result = _add_month_cyclical(df)
        assert (result["sin_month"] == 0.999).all()


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:

    def test_month_range_basic(self):
        pipeline = _load_pipeline_script()
        months = pipeline._month_range("2015-01", "2015-03")
        assert months == [(2015, 1), (2015, 2), (2015, 3)]

    def test_month_range_cross_year(self):
        pipeline = _load_pipeline_script()
        months = pipeline._month_range("2015-11", "2016-02")
        assert months == [(2015, 11), (2015, 12), (2016, 1), (2016, 2)]

    def test_month_range_single_month(self):
        pipeline = _load_pipeline_script()
        months = pipeline._month_range("2015-03", "2015-03")
        assert months == [(2015, 3)]

    def test_month_range_invalid_format_raises(self):
        pipeline = _load_pipeline_script()
        with pytest.raises(ValueError, match="Invalid date format"):
            pipeline._month_range("2015/01", "2015/03")

    def test_month_range_start_after_end_raises(self):
        pipeline = _load_pipeline_script()
        with pytest.raises(ValueError, match="after end date"):
            pipeline._month_range("2016-01", "2015-01")
