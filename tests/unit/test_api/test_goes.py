"""Tests for the unified GOES retriever."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.api.goes import (  # noqa: E402
    GOESMagRetriever,
    GOESXrayRetriever,
    _canonicalize_satellite_xray,
    detect_goes_era,
    format_goes_satellite,
    merge_goes_satellites,
    merge_goes_xray_satellites,
    parse_goes_satellite_number,
)


def _frame(start: str, values: list[float], satellite: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(values), freq="1min", tz="UTC"),
            "goes_bz_gsm": values,
            "goes_source_satellite": satellite,
        }
    )


def _xray_frame(
    start: str,
    xrsa: list[float],
    xrsb: list[float],
    *,
    a_flag: list[int] | None = None,
    b_flag: list[int] | None = None,
    electron_flag: list[int] | None = None,
) -> pd.DataFrame:
    n = len(xrsa)
    data = {
        "timestamp": pd.date_range(start, periods=n, freq="1min", tz="UTC"),
        "xrsa_flux": xrsa,
        "xrsb_flux": xrsb,
        "xrsa_flag": a_flag or [0] * n,
        "xrsb_flag": b_flag or [0] * n,
    }
    if electron_flag is not None:
        data["electron_correction_flag"] = electron_flag
    return pd.DataFrame(data)


def test_satellite_number_parsing_and_era_detection() -> None:
    assert parse_goes_satellite_number("GOES-15") == 15
    assert parse_goes_satellite_number("g16") == 16
    assert parse_goes_satellite_number(18) == 18
    assert parse_goes_satellite_number("GOES-19") == 19
    assert format_goes_satellite("goes17") == "GOES-17"
    assert detect_goes_era("GOES-15") == "legacy"
    assert detect_goes_era("GOES-16") == "modern"
    assert detect_goes_era("GOES-19") == "modern"


def test_goes_priority_includes_2024_with_goes_19(tmp_path: Path) -> None:
    from swmi.api.goes import _load_goes_priority

    p = tmp_path / "data_retrieval.yaml"
    p.write_text(
        """
goes:
  satellite_priority:
    2024: ["GOES-16", "GOES-19", "GOES-18"]
""",
        encoding="utf-8",
    )
    assert _load_goes_priority(2024, p) == ["GOES-16", "GOES-19", "GOES-18"]


def test_merge_goes_satellites_uses_priority_and_flags_gaps() -> None:
    expected_index = pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC")
    primary = _frame("2015-03-01", [1.0, np.nan], "GOES-16")
    backup = _frame("2015-03-01", [9.0, 2.0], "GOES-15")

    merged = merge_goes_satellites(
        {"GOES-16": primary, "GOES-15": backup},
        ["GOES-16", "GOES-15"],
        expected_index=expected_index,
    )

    assert list(merged["timestamp"]) == list(expected_index)
    assert list(merged["goes_bz_gsm"][:2]) == [1.0, 2.0]
    assert pd.isna(merged.loc[2, "goes_bz_gsm"])
    assert list(merged["goes_source_satellite"][:2]) == ["GOES-16", "GOES-15"]
    assert list(merged["goes_mag_missing_flag"]) == [0, 0, 1]
    assert not merged["timestamp"].duplicated().any()
    assert merged.attrs["gap_summary"] == {
        "expected_minutes": 3,
        "valid_minutes": 2,
        "missing_minutes": 1,
        "backup_filled_minutes": 1,
    }


def test_merge_goes_satellites_loads_year_priority_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "data_retrieval.yaml"
    config_path.write_text(
        """
goes:
  satellite_priority:
    2015: ["GOES-15", "GOES-13"]
""",
        encoding="utf-8",
    )
    expected_index = pd.date_range("2015-03-01", periods=2, freq="1min", tz="UTC")
    primary = _frame("2015-03-01", [1.0, np.nan], "GOES-15")
    backup = _frame("2015-03-01", [9.0, 2.0], "GOES-13")

    merged = merge_goes_satellites(
        {"GOES-13": backup, "GOES-15": primary},
        year=2015,
        config_path=config_path,
        expected_index=expected_index,
    )

    assert list(merged["goes_bz_gsm"]) == [1.0, 2.0]
    assert list(merged["goes_source_satellite"]) == ["GOES-15", "GOES-13"]


def test_goes_mag_retrieve_merges_priority_and_writes_canonical_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_retrieve_satellite_mag(self, satellite: str, year: int, month: int) -> pd.DataFrame:
        calls.append(satellite)
        if satellite == "GOES-15":
            return _frame("2015-03-01", [10.0, np.nan, 30.0], satellite)
        if satellite == "GOES-13":
            return _frame("2015-03-01", [99.0, 20.0, 39.0], satellite)
        raise AssertionError(f"unexpected satellite {satellite}")

    monkeypatch.setattr(
        GOESMagRetriever,
        "_retrieve_satellite_mag",
        fake_retrieve_satellite_mag,
    )

    retriever = GOESMagRetriever(output_root=tmp_path)
    df = retriever.retrieve(2015, 3, force=True)

    assert calls == ["GOES-15", "GOES-13"]
    assert len(df) == 31 * 24 * 60
    assert list(df["goes_bz_gsm"].head(3)) == [10.0, 20.0, 30.0]
    assert list(df["goes_source_satellite"].head(3)) == ["GOES-15", "GOES-13", "GOES-15"]
    assert int(df["goes_mag_missing_flag"].head(3).sum()) == 0
    assert int(df["goes_mag_missing_flag"].sum()) == len(df) - 3

    output_path = tmp_path / "goes_mag_201503.parquet"
    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert list(saved.columns) == [
        "timestamp",
        "goes_bz_gsm",
        "goes_source_satellite",
        "goes_mag_missing_flag",
    ]


def test_goes_xray_retrieve_filters_quality_and_writes_canonical_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_retrieve_satellite_xray(self, satellite: str, year: int, month: int) -> pd.DataFrame:
        calls.append(satellite)
        if satellite == "GOES-16":
            return _xray_frame(
                "2017-04-01",
                [1e-8, 2e-8, 3e-8, 4e-8],
                [1e-6, 2e-6, 3e-6, 4e-6],
                electron_flag=[0, 1, 8, 0],
            )
        if satellite == "GOES-15":
            return _xray_frame(
                "2017-04-01",
                [9e-8, 8e-8, 7e-8, 6e-8],
                [9e-6, 8e-6, 7e-6, 6e-6],
                a_flag=[0, 0, 1, 0],
                b_flag=[0, 0, 0, 0],
            )
        raise AssertionError(f"unexpected satellite {satellite}")

    monkeypatch.setattr(
        GOESXrayRetriever,
        "_retrieve_satellite_xray",
        fake_retrieve_satellite_xray,
    )

    retriever = GOESXrayRetriever(output_root=tmp_path)
    df = retriever.retrieve(2017, 4, force=True)

    assert calls == ["GOES-16", "GOES-15"]
    assert len(df) == 30 * 24 * 60
    assert list(df["xrsa_flux"].head(2)) == [1e-8, 8e-8]
    assert pd.isna(df.loc[2, "xrsa_flux"])
    assert df.loc[3, "xrsa_flux"] == 4e-8
    assert list(df["xrsb_flux"].head(4)) == [1e-6, 8e-6, 7e-6, 4e-6]
    assert list(df["xray_source_satellite"].head(4)) == [
        "GOES-16",
        "GOES-15",
        "GOES-15",
        "GOES-16",
    ]
    assert int(df["xray_missing_flag"].head(4).sum()) == 0
    assert int(df["xray_missing_flag"].sum()) == len(df) - 4

    output_path = tmp_path / "goes_xray_201704.parquet"
    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert "au_factor" not in saved.columns
    assert list(saved.columns) == [
        "timestamp",
        "xrsa_flux",
        "xrsb_flux",
        "xray_quality_flags",
        "xray_source_satellite",
        "xray_missing_flag",
    ]


def test_xray_canonicalization_rejects_fill_values_and_out_of_range_flux() -> None:
    raw = _xray_frame(
        "2015-01-01",
        [1e-9, -9999.0, 1e-10, 4e-3],
        [1e-9, -9999.0, 1e-10, 3e-1],
    )

    canonical = _canonicalize_satellite_xray(raw, "GOES-15", "legacy")

    assert canonical.loc[0, "xrsa_flux"] == 1e-9
    assert canonical.loc[0, "xrsb_flux"] == 1e-9
    assert canonical["xrsa_flux"].iloc[1:].isna().all()
    assert canonical["xrsb_flux"].iloc[1:].isna().all()


def test_merge_goes_xray_satellites_uses_priority() -> None:
    expected_index = pd.date_range("2022-01-01", periods=2, freq="1min", tz="UTC")
    primary = pd.DataFrame(
        {
            "timestamp": expected_index,
            "xrsa_flux": [np.nan, 2e-8],
            "xrsb_flux": [np.nan, 2e-6],
            "xray_quality_flags": ["bad", "good"],
            "xray_source_satellite": ["GOES-16", "GOES-16"],
        }
    )
    backup = pd.DataFrame(
        {
            "timestamp": expected_index,
            "xrsa_flux": [1e-8, 9e-8],
            "xrsb_flux": [1e-6, 9e-6],
            "xray_quality_flags": ["good", "good"],
            "xray_source_satellite": ["GOES-18", "GOES-18"],
        }
    )

    merged = merge_goes_xray_satellites(
        {"GOES-16": primary, "GOES-18": backup},
        ["GOES-16", "GOES-18"],
        expected_index=expected_index,
    )

    assert list(merged["xrsa_flux"]) == [1e-8, 2e-8]
    assert list(merged["xray_source_satellite"]) == ["GOES-18", "GOES-16"]
    assert list(merged["xray_missing_flag"]) == [0, 0]


def test_parse_xray_netcdf_decodes_j2000_time(tmp_path: Path) -> None:
    import xarray as xr

    path = tmp_path / "xrs.nc"
    ds = xr.Dataset(
        data_vars={
            "xrsa_flux": ("time", [1e-8, 2e-8]),
            "xrsb_flux": ("time", [1e-6, 2e-6]),
            "xrsa_flag": ("time", [0, 0]),
            "xrsb_flag": ("time", [0, 0]),
        },
        coords={"time": ("time", [0, 60], {"units": "seconds since 2000-01-01 12:00:00"})},
    )
    ds.to_netcdf(path)

    parsed = GOESXrayRetriever._parse_xray_netcdf(path)

    assert parsed is not None
    assert list(parsed["timestamp"]) == list(
        pd.date_range("2000-01-01 12:00:00", periods=2, freq="1min", tz="UTC")
    )
    assert list(parsed["xrsa_flux"]) == [1e-8, 2e-8]
