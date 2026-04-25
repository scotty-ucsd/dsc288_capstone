"""Tests for download and feature-build orchestrator scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_data_config(path: Path) -> None:
    path.write_text(
        """
master_cadence: "1min"
swarm:
  satellites: ["A", "C"]
""".strip(),
        encoding="utf-8",
    )


def _write_feature_config(path: Path) -> None:
    path.write_text(
        """
scientific_invariants:
  reference_field: "IGRF"
  qdlat_high_lat_min: 55.0
  decay_halflife_min: 10.0
  dbdt_method: "backward"
dbdt:
  gap_threshold_sec: 90.0
""".strip(),
        encoding="utf-8",
    )


def test_download_orchestrator_runs_sources_and_validation(tmp_path: Path, monkeypatch) -> None:
    script = _load_script("scripts/01_download_data.py", "download_orchestrator_test")
    cfg = tmp_path / "data_retrieval.yaml"
    _write_data_config(cfg)
    calls: list[tuple[str, tuple]] = []

    monkeypatch.setattr(script, "create_directories", lambda root: calls.append(("create_directories", (root,))))
    monkeypatch.setattr(script, "retrieve_omni", lambda *args: calls.append(("omni", args)))
    monkeypatch.setattr(script, "retrieve_goes_mag", lambda *args: calls.append(("goes_mag", args)))
    monkeypatch.setattr(script, "retrieve_goes_xray", lambda *args: calls.append(("goes_xray", args)))
    monkeypatch.setattr(script, "retrieve_swarm_month", lambda *args: calls.append(("swarm", args)))
    monkeypatch.setattr(script, "retrieve_supermag_month", lambda *args: calls.append(("supermag", args)))
    monkeypatch.setattr(script, "validate_sources", lambda *args, **kwargs: calls.append(("validate", args)) or True)

    results = script.run_download_pipeline(2015, 3, config_path=cfg)

    assert all(result.ok for result in results)
    assert [name for name, _ in calls] == [
        "create_directories",
        "omni",
        "goes_mag",
        "goes_xray",
        "swarm",
        "swarm",
        "supermag",
        "validate",
    ]
    assert calls[4][1] == (2015, 3, "A")
    assert calls[5][1] == (2015, 3, "C")


def test_download_orchestrator_dry_run_does_not_call_retrievers(tmp_path: Path, monkeypatch) -> None:
    script = _load_script("scripts/01_download_data.py", "download_orchestrator_dry_test")
    cfg = tmp_path / "data_retrieval.yaml"
    _write_data_config(cfg)
    called = False

    def _mark_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(script, "retrieve_omni", _mark_called)

    results = script.run_download_pipeline(2015, 3, config_path=cfg, dry_run=True)

    assert results == []
    assert called is False


def test_feature_orchestrator_builds_derived_products(tmp_path: Path, monkeypatch) -> None:
    script = _load_script("scripts/02_build_features.py", "feature_orchestrator_test")
    cfg = tmp_path / "feature_engineering.yaml"
    _write_feature_config(cfg)
    calls: list[tuple[str, tuple, dict]] = []
    raw_xray = tmp_path / "data/raw/goes/goes_xray_201503.parquet"
    raw_supermag = tmp_path / "data/raw/supermag/2015/03/supermag_201503.parquet"
    raw_xray.parent.mkdir(parents=True)
    raw_supermag.parent.mkdir(parents=True)
    raw_xray.touch()
    raw_supermag.touch()

    monkeypatch.setattr(script, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "create_directories", lambda root: calls.append(("create_directories", (root,), {})))
    monkeypatch.setattr(
        script,
        "normalize_goes_xray",
        lambda *args, **kwargs: calls.append(("normalize_goes_xray", args, kwargs)),
    )
    monkeypatch.setattr(
        script,
        "compute_all_station_dbdt",
        lambda *args, **kwargs: calls.append(("compute_all_station_dbdt", args, kwargs)),
    )
    monkeypatch.setattr(script, "build_leo_index_month", lambda *args: calls.append(("build_leo_index_month", args, {})))
    monkeypatch.setattr(script, "build_feature_matrix", lambda *args: calls.append(("build_feature_matrix", args, {})))
    monkeypatch.setattr(script, "validate_feature_matrix", lambda *args, **kwargs: calls.append(("validate", args, kwargs)) or True)

    ok = script.run_feature_pipeline(2015, 3, config_path=cfg)

    assert ok is True
    assert [name for name, _, _ in calls] == [
        "create_directories",
        "normalize_goes_xray",
        "compute_all_station_dbdt",
        "build_leo_index_month",
        "build_feature_matrix",
        "validate",
    ]
    dbdt_kwargs = calls[2][2]
    assert dbdt_kwargs["gap_threshold_sec"] == 90.0
    assert dbdt_kwargs["output_path"] == tmp_path / "data/processed/supermag/2015/03/supermag_201503.parquet"


def test_feature_orchestrator_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    script = _load_script("scripts/02_build_features.py", "feature_orchestrator_dry_test")
    cfg = tmp_path / "feature_engineering.yaml"
    _write_feature_config(cfg)
    called = False

    def _mark_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(script, "build_feature_matrix", _mark_called)

    ok = script.run_feature_pipeline(2015, 3, config_path=cfg, dry_run=True)

    assert ok is True
    assert called is False
