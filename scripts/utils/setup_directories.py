#!/usr/bin/env python3
"""Create the SWMI pipeline directory layout.

This utility is intentionally idempotent. It creates only directories, never
deletes or rewrites data products, and is safe to call at the start of entry
point scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_DATA_SUBDIRS = (
    "external",
    "external/station_metadata",
    "raw",
    "raw/omni",
    "raw/goes",
    "raw/goes/legacy",
    "raw/goes/modern",
    "raw/swarm",
    "raw/supermag",
    "interim",
    "interim/cleaned",
    "processed",
    "processed/aligned_1min",
    "processed/features",
    "processed/goes",
    "processed/omni",
    "processed/sequences",
    "processed/supermag",
    "processed/swarm",
    "processed/targets",
    "sequences",
    "sequences/train",
    "sequences/val",
    "sequences/test",
    "tmp",
)

_PROJECT_SUBDIRS = (
    "logs",
    "models",
    "models/artifacts",
    "models/checkpoints",
    "models/checkpoints/exp001",
    "results",
    "results/experiments",
    "results/experiments/exp001_baseline",
    "results/figures",
    "results/validation",
)


def planned_directories(project_root: str | Path = ".") -> list[Path]:
    """Return the canonical directory layout under ``project_root``."""
    root = Path(project_root)
    data_root = root / "data"
    directories = [data_root / subdir for subdir in _DATA_SUBDIRS]
    directories.extend(root / subdir for subdir in _PROJECT_SUBDIRS)
    return sorted(set(directories))


def create_directories(project_root: str | Path = ".") -> list[Path]:
    """Create all expected SWMI pipeline directories.

    Parameters
    ----------
    project_root:
        Repository root. Defaults to the current working directory.

    Returns
    -------
    list[Path]
        Paths that were ensured to exist.
    """
    directories = planned_directories(project_root)
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    return directories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create SWMI pipeline directories.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where directories should be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories without creating them.",
    )
    args = parser.parse_args(argv)

    directories = planned_directories(args.project_root)
    if args.dry_run:
        for path in directories:
            print(path)
        return 0

    create_directories(args.project_root)
    print(f"Ensured {len(directories)} SWMI directories exist under {Path(args.project_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
