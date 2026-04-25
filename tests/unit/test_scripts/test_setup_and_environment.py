"""Tests for setup and environment utility scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils.check_environment import (  # noqa: E402
    CheckResult,
    check_api_credentials,
    check_disk_space,
    check_model_files,
    environment_ok,
)
from scripts.utils.setup_directories import create_directories, planned_directories  # noqa: E402
from swmi.utils import config  # noqa: E402


def test_create_directories_is_idempotent(tmp_path: Path) -> None:
    first = create_directories(tmp_path)
    second = create_directories(tmp_path)

    assert first == second
    assert (tmp_path / "data" / "raw" / "goes").is_dir()
    assert (tmp_path / "data" / "processed" / "features").is_dir()
    assert (tmp_path / "data" / "sequences" / "train").is_dir()
    assert (tmp_path / "models" / "checkpoints" / "exp001").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_planned_directories_are_under_project_root(tmp_path: Path) -> None:
    directories = planned_directories(tmp_path)

    assert directories
    assert all(tmp_path in path.parents for path in directories)


def test_api_credentials_can_be_strict_or_non_blocking() -> None:
    strict_results = check_api_credentials(env={}, strict_api=True)
    relaxed_results = check_api_credentials(env={}, strict_api=False)

    assert not environment_ok(strict_results)
    assert environment_ok(relaxed_results)

    with_supermag = check_api_credentials(
        env={"SUPERMAG_USERNAME": "user"},
        strict_api=True,
    )
    assert environment_ok(with_supermag)


def test_model_file_check_requires_chaos_only_when_configured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFERENCE_FIELD", "IGRF")
    assert environment_ok(check_model_files(tmp_path))

    monkeypatch.setattr(config, "REFERENCE_FIELD", "CHAOS")
    assert not environment_ok(check_model_files(tmp_path))

    chaos_path = tmp_path / "models" / "CHAOS-8.5.mat"
    chaos_path.parent.mkdir(parents=True)
    chaos_path.write_bytes(b"placeholder")
    assert environment_ok(check_model_files(tmp_path))


def test_disk_space_threshold_reports_failure_for_impossible_threshold(tmp_path: Path) -> None:
    result = check_disk_space(tmp_path, min_free_gb=10**12)

    assert isinstance(result, CheckResult)
    assert not result.ok
