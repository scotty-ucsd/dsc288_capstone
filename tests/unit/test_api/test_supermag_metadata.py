"""Tests for SuperMAG station metadata precompute."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.api import supermag


def _station_frame(station: str, *, glat: float, month: int) -> pd.DataFrame:
    ts = pd.date_range(f"2015-{month:02d}-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "station": [station] * len(ts),
            "n_nez": [1.0, 2.0, 3.0],
            "e_nez": [1.0, 2.0, 3.0],
            "z_nez": [1.0, 2.0, 3.0],
            "mlt": [(hour + 4.0) % 24 for hour in [0, 1, 2]],
            "mlat": [65.0] * len(ts),
            "mlon": [110.0] * len(ts),
            "glat": [glat] * len(ts),
            "glon": [20.0] * len(ts),
        }
    )


def test_precompute_station_metadata_collapses_months(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "supermag_station_coords.parquet"

    monkeypatch.setattr(
        supermag,
        "get_station_inventory",
        lambda year, month, force_refresh=False: ["ABK"] if month == 1 else ["ABK", "TRO"],
    )

    def _fake_fetch(station: str, start_dt, extent_sec: int) -> pd.DataFrame:
        month = start_dt.month
        return _station_frame(station, glat=60.0 + month, month=month)

    monkeypatch.setattr(supermag, "_fetch_station", _fake_fetch)

    metadata = supermag.precompute_station_metadata(
        2015,
        1,
        2015,
        2,
        output_path=output,
        use_cached_raw=False,
        fetch_missing_metadata=True,
    )

    assert output.exists()
    assert metadata["station"].tolist() == ["ABK", "TRO"]
    abk = metadata.set_index("station").loc["ABK"]
    assert abk["operational_start"] == "2015-01"
    assert abk["operational_end"] == "2015-02"
    assert abk["sample_months"] == 2
    assert abk["mlt_offset"] == 4.0
    assert bool(abk["metadata_available"]) is True


def test_precompute_station_metadata_can_use_cached_raw(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "metadata.parquet"
    cached = _station_frame("ABK", glat=68.0, month=3)

    monkeypatch.setattr(supermag, "_load_cached_station_month", lambda station, year, month: cached)
    monkeypatch.setattr(
        supermag,
        "_fetch_station",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Should not fetch when cached raw exists.")),
    )

    metadata = supermag.precompute_station_metadata(
        2015,
        3,
        2015,
        3,
        stations=["ABK"],
        output_path=output,
        fetch_missing_metadata=True,
    )

    assert len(metadata) == 1
    assert metadata.loc[0, "glat"] == 68.0
    assert output.exists()


def test_station_metadata_schema_rejects_duplicates() -> None:
    df = pd.DataFrame(
        {
            "station": ["ABK", "ABK"],
            "glat": [1.0, 2.0],
            "glon": [1.0, 2.0],
            "mlat": [1.0, 2.0],
            "mlon": [1.0, 2.0],
            "mlt_offset": [1.0, 2.0],
            "operational_start": ["2015-01", "2015-02"],
            "operational_end": ["2015-01", "2015-02"],
            "sample_months": [1, 1],
            "metadata_available": [True, True],
        }
    )

    try:
        supermag._validate_station_metadata_schema(df)
    except ValueError as exc:
        assert "duplicate station" in str(exc)
    else:
        raise AssertionError("Expected duplicate station validation failure.")
