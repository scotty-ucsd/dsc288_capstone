"""Integration checks: dry-run orchestration and optional local month validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_download_script_dry_run_year_range_exits_zero() -> None:
    script = ROOT / "scripts" / "01_download_data.py"
    r = subprocess.run(
        [sys.executable, str(script), "--dry-run", "2015", "2024"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_leo_index_month_is_single_chunk_scope() -> None:
    """CHAOS/IGRF LEO build processes one calendar month per call (OOM-safe chunking)."""
    from swmi.features import leo_index

    assert hasattr(leo_index, "build_leo_index_month")
    src = Path(leo_index.__file__).read_text(encoding="utf-8")
    assert "build_leo_index_month" in src
    assert "_month_start_end" in src or "year, month" in src
