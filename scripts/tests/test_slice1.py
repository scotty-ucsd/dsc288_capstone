#!/usr/bin/env python3
"""
Validation script for P0 Slice 1 tasks:
  P0-A5: validate_output_schema call fix
  P0-A3: gap-aware dB/dt computation
  P0-B1: YAML config loader

Run: uv run python scripts/tests/test_slice1.py
"""

import sys
import os
import traceback

# Resolve project imports the same way the codebase does
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_project_root, "src", "swmi", "utils"))
sys.path.insert(0, os.path.join(_project_root, "src", "swmi", "preprocessing"))

import numpy as np
import pandas as pd

# Track results
passed = 0
failed = 0
errors = []


def test(name):
    """Decorator to track test results."""
    def decorator(func):
        global passed, failed
        try:
            func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))
    return decorator


# ===================================================================
# P0-A5: validate_output_schema with unique_subset
# ===================================================================
print("\n" + "="*70)
print("P0-A5: validate_output_schema call fix")
print("="*70)

from validation import validate_output_schema


@test("Long-format SuperMAG data passes with unique_subset=[timestamp, station]")
def _():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2015-03-17 04:00", "2015-03-17 04:00", "2015-03-17 04:01", "2015-03-17 04:01"],
            utc=True,
        ),
        "station": ["ABK", "TRO", "ABK", "TRO"],
        "n_nez": [100.0, 200.0, 110.0, 210.0],
    })
    # Should NOT raise — duplicate timestamps are OK when stations differ
    validate_output_schema(df, "SuperMAG-test", unique_subset=["timestamp", "station"])


@test("validate_output_schema rejects actual duplicates with unique_subset")
def _():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2015-03-17 04:00", "2015-03-17 04:00"],
            utc=True,
        ),
        "station": ["ABK", "ABK"],  # same station = true duplicate
        "n_nez": [100.0, 100.0],
    })
    try:
        validate_output_schema(df, "SuperMAG-dup", unique_subset=["timestamp", "station"])
        raise AssertionError("Should have raised ValueError for true duplicates")
    except ValueError:
        pass  # expected


@test("validate_output_schema still works with default (timestamp-only) uniqueness")
def _():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2015-03-17 04:00", "2015-03-17 04:01"], utc=True),
        "goes_bz_gsm": [-5.0, -7.0],
    })
    validate_output_schema(df, "GOES-test")


@test("validate_output_schema rejects allow_duplicates parameter (no such arg)")
def _():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2015-03-17 04:00"], utc=True),
        "x": [1.0],
    })
    try:
        validate_output_schema(df, "test", allow_duplicates=True)
        raise AssertionError("Should have raised TypeError for unknown kwarg")
    except TypeError:
        pass  # expected


# ===================================================================
# P0-A3: Gap-aware dB/dt computation
# ===================================================================
print("\n" + "="*70)
print("P0-A3: Gap-aware dB/dt computation")
print("="*70)

from cleaners import compute_dbdt_gap_aware, compute_all_station_dbdt


@test("Basic backward diff with uniform 1-min spacing")
def _():
    ts = pd.to_datetime([
        "2015-03-17 04:00", "2015-03-17 04:01", "2015-03-17 04:02",
    ], utc=True)
    b = pd.DataFrame({
        "n_nez": [100.0, 110.0, 105.0],
        "e_nez": [50.0,  52.0,  48.0],
        "z_nez": [30.0,  31.0,  29.0],
    })
    result = compute_dbdt_gap_aware(b, ts)

    # First row: always NaN
    assert np.isnan(result["dbdt_n"].iloc[0])
    assert np.isnan(result["dbdt_e"].iloc[0])
    assert np.isnan(result["dbdt_horizontal_magnitude"].iloc[0])

    # Second row: (110-100)/60s * 60 = 10.0 nT/min
    assert abs(result["dbdt_n"].iloc[1] - 10.0) < 1e-9
    # (52-50)/60s * 60 = 2.0
    assert abs(result["dbdt_e"].iloc[1] - 2.0) < 1e-9

    # Horizontal magnitude: sqrt(10^2 + 2^2) = sqrt(104)
    expected_mag = np.sqrt(10.0**2 + 2.0**2)
    assert abs(result["dbdt_horizontal_magnitude"].iloc[1] - expected_mag) < 1e-9

    # No gaps in uniform data
    assert result["dbdt_gap_flag"].sum() == 0


@test("Gap detection: 5-minute gap produces NaN dB/dt and gap_flag=1")
def _():
    ts = pd.to_datetime([
        "2015-03-17 04:00",
        "2015-03-17 04:01",
        "2015-03-17 04:06",  # 5-minute gap (> 90s threshold)
        "2015-03-17 04:07",
    ], utc=True)
    b = pd.DataFrame({
        "n_nez": [100.0, 110.0, 120.0, 125.0],
        "e_nez": [50.0,  52.0,  54.0,  55.0],
        "z_nez": [30.0,  31.0,  32.0,  33.0],
    })
    result = compute_dbdt_gap_aware(b, ts)

    # Index 2 crosses a 5-minute gap → must be NaN
    assert np.isnan(result["dbdt_n"].iloc[2]), "Gap should produce NaN dB/dt"
    assert np.isnan(result["dbdt_horizontal_magnitude"].iloc[2])
    assert result["dbdt_gap_flag"].iloc[2] == 1

    # Index 3 is back to normal (1-min spacing from idx 2)
    assert np.isfinite(result["dbdt_n"].iloc[3])
    assert result["dbdt_gap_flag"].iloc[3] == 0


@test("Actual dt used (not constant 60s): 30-second interval")
def _():
    ts = pd.to_datetime([
        "2015-03-17 04:00:00",
        "2015-03-17 04:00:30",  # 30-second interval
    ], utc=True)
    b = pd.DataFrame({
        "n_nez": [100.0, 110.0],
        "e_nez": [50.0,  50.0],
        "z_nez": [30.0,  30.0],
    })
    result = compute_dbdt_gap_aware(b, ts)

    # dB/dt = (110-100) / 30s * 60 = 20.0 nT/min  (NOT 10.0 which diff()/60 would give)
    assert abs(result["dbdt_n"].iloc[1] - 20.0) < 1e-9, \
        f"Expected 20.0 nT/min, got {result['dbdt_n'].iloc[1]}"


@test("NaN in B-field propagates to NaN dB/dt")
def _():
    ts = pd.to_datetime([
        "2015-03-17 04:00", "2015-03-17 04:01", "2015-03-17 04:02",
    ], utc=True)
    b = pd.DataFrame({
        "n_nez": [100.0, np.nan, 105.0],
        "e_nez": [50.0,  52.0,  48.0],
        "z_nez": [30.0,  31.0,  29.0],
    })
    result = compute_dbdt_gap_aware(b, ts)
    # idx 1: n_nez is NaN → diff is NaN → dbdt_n is NaN
    assert np.isnan(result["dbdt_n"].iloc[1])
    # idx 2: diff from NaN predecessor → dbdt_n is NaN
    assert np.isnan(result["dbdt_n"].iloc[2])
    # horizontal magnitude also NaN when dbdt_n is NaN
    assert np.isnan(result["dbdt_horizontal_magnitude"].iloc[1])


@test("Empty DataFrame returns correct schema")
def _():
    ts = pd.Series([], dtype="datetime64[ns, UTC]")
    b = pd.DataFrame({"n_nez": [], "e_nez": [], "z_nez": []})
    result = compute_dbdt_gap_aware(b, ts)
    assert len(result) == 0
    expected_cols = {"dbdt_n", "dbdt_e", "dbdt_z", "dbdt_horizontal_magnitude", "dbdt_gap_flag"}
    assert set(result.columns) == expected_cols


@test("Single-row DataFrame returns all NaN")
def _():
    ts = pd.to_datetime(["2015-03-17 04:00"], utc=True)
    b = pd.DataFrame({"n_nez": [100.0], "e_nez": [50.0], "z_nez": [30.0]})
    result = compute_dbdt_gap_aware(b, ts)
    assert len(result) == 1
    assert np.isnan(result["dbdt_n"].iloc[0])
    assert np.isnan(result["dbdt_horizontal_magnitude"].iloc[0])


@test("Missing NEZ columns raises KeyError")
def _():
    ts = pd.to_datetime(["2015-03-17 04:00"], utc=True)
    b = pd.DataFrame({"b_n": [100.0], "b_e": [50.0]})  # wrong column names
    try:
        compute_dbdt_gap_aware(b, ts)
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        pass


@test("Mismatched lengths raises ValueError")
def _():
    ts = pd.to_datetime(["2015-03-17 04:00", "2015-03-17 04:01"], utc=True)
    b = pd.DataFrame({"n_nez": [100.0], "e_nez": [50.0], "z_nez": [30.0]})
    try:
        compute_dbdt_gap_aware(b, ts)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


@test("compute_all_station_dbdt processes multiple stations correctly")
def _():
    raw = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2015-03-17 04:00", "2015-03-17 04:01", "2015-03-17 04:02",
            "2015-03-17 04:00", "2015-03-17 04:01", "2015-03-17 04:02",
        ], utc=True),
        "station": ["ABK", "ABK", "ABK", "TRO", "TRO", "TRO"],
        "n_nez": [100.0, 110.0, 105.0, 200.0, 210.0, 205.0],
        "e_nez": [50.0,  52.0,  48.0,  60.0,  62.0,  58.0],
        "z_nez": [30.0,  31.0,  29.0,  40.0,  41.0,  39.0],
    })
    result = compute_all_station_dbdt(raw)

    assert "timestamp" in result.columns
    assert "station" in result.columns
    assert "dbdt_horizontal_magnitude" in result.columns
    assert len(result) == 6
    assert set(result["station"].unique()) == {"ABK", "TRO"}

    # Each station's first row should be NaN
    for stn in ["ABK", "TRO"]:
        stn_data = result[result["station"] == stn].sort_values("timestamp")
        assert np.isnan(stn_data["dbdt_n"].iloc[0])


# ===================================================================
# P0-B1: YAML config loader
# ===================================================================
print("\n" + "="*70)
print("P0-B1: YAML config loader")
print("="*70)

import config
from config import load_config, validate_scientific_invariants


@test("load_config reads data_retrieval.yaml successfully")
def _():
    cfg = load_config(os.path.join(_project_root, "configs", "data_retrieval.yaml"))
    assert isinstance(cfg, dict)
    assert cfg["master_cadence"] == "1min"
    assert "goes" in cfg
    assert "satellite_priority" in cfg["goes"]


@test("load_config reads feature_engineering.yaml successfully")
def _():
    cfg = load_config(os.path.join(_project_root, "configs", "feature_engineering.yaml"))
    assert isinstance(cfg, dict)
    assert cfg["scientific_invariants"]["reference_field"] == "IGRF"
    assert cfg["scientific_invariants"]["qdlat_high_lat_min"] == 55.0
    assert cfg["scientific_invariants"]["decay_halflife_min"] == 10.0
    assert cfg["scientific_invariants"]["dbdt_method"] == "backward"


@test("load_config reads model_baseline.yaml successfully")
def _():
    cfg = load_config(os.path.join(_project_root, "configs", "model_baseline.yaml"))
    assert isinstance(cfg, dict)
    assert cfg["forecast_horizon_min"] == 60
    assert cfg["split_buffer_days"] == 7


@test("load_config raises FileNotFoundError for missing file")
def _():
    try:
        load_config("configs/nonexistent.yaml")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


@test("validate_scientific_invariants passes for correct feature_engineering.yaml")
def _():
    cfg = load_config(os.path.join(_project_root, "configs", "feature_engineering.yaml"))
    validate_scientific_invariants(cfg, "feature_engineering")  # should not raise


@test("validate_scientific_invariants passes for correct model_baseline.yaml")
def _():
    cfg = load_config(os.path.join(_project_root, "configs", "model_baseline.yaml"))
    validate_scientific_invariants(cfg, "model_baseline")  # should not raise


@test("validate_scientific_invariants raises on tampered invariant")
def _():
    cfg = {
        "scientific_invariants": {
            "reference_field": "CHAOS",  # WRONG — should be IGRF
            "qdlat_high_lat_min": 55.0,
            "decay_halflife_min": 10.0,
            "dbdt_method": "backward",
        }
    }
    try:
        validate_scientific_invariants(cfg, "tampered")
        raise AssertionError("Should have raised ValueError for tampered invariant")
    except ValueError as e:
        assert "REFERENCE_FIELD" in str(e)


@test("Python constants match approved values")
def _():
    assert config.REFERENCE_FIELD == "IGRF"
    assert config.QDLAT_HIGH_LAT_MIN == 55.0
    assert config.DECAY_HALFLIFE_MIN == 10.0
    assert config.DBDT_METHOD == "backward"
    assert config.FORECAST_HORIZON_MIN == 60
    assert config.MASTER_CADENCE == "1min"
    assert config.SPLIT_BUFFER_DAYS == 7


# ===================================================================
# Summary
# ===================================================================
print("\n" + "="*70)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
if errors:
    print("\nFailures:")
    for name, msg in errors:
        print(f"  ❌ {name}: {msg}")
print("="*70)

sys.exit(0 if failed == 0 else 1)
