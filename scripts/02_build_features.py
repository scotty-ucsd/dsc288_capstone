#!/usr/bin/env python3
"""Build monthly LEO index, SuperMAG targets, and feature matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.setup_directories import create_directories
from swmi.features.builder import build_feature_matrix
from swmi.features.leo_index import build_leo_index_month
from swmi.preprocessing.cleaners import compute_all_station_dbdt, normalize_goes_xray
from swmi.preprocessing.validation import validate_feature_matrix
from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants
from swmi.utils.logger import get_logger

log = get_logger(__name__)


def _load_feature_config(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_scientific_invariants(cfg, str(config_path))
    return cfg


def _month_paths(year: int, month: int) -> dict[str, Path]:
    month_str = f"{year:04d}{month:02d}"
    return {
        "goes_xray_raw": _PROJECT_ROOT / config.RAW_DATA_DIR / "goes" / f"goes_xray_{month_str}.parquet",
        "goes_xray_normalized": (
            _PROJECT_ROOT
            / config.PROCESSED_DIR
            / "goes"
            / f"{year:04d}"
            / f"{month:02d}"
            / f"goes_xray_normalized_{month_str}.parquet"
        ),
        "xray_validation_plot": _PROJECT_ROOT / "results" / "figures" / f"goes_xray_normalization_{month_str}.png",
        "supermag_raw": (
            _PROJECT_ROOT
            / config.RAW_DATA_DIR
            / "supermag"
            / f"{year:04d}"
            / f"{month:02d}"
            / f"supermag_{month_str}.parquet"
        ),
        "supermag_processed": (
            _PROJECT_ROOT
            / config.PROCESSED_DIR
            / "supermag"
            / f"{year:04d}"
            / f"{month:02d}"
            / f"supermag_{month_str}.parquet"
        ),
        "features": (
            _PROJECT_ROOT
            / config.FEATURES_DIR
            / f"{year:04d}"
            / f"{month:02d}"
            / f"features_{month_str}.parquet"
        ),
    }


def run_feature_pipeline(
    year: int,
    month: int,
    *,
    config_path: str | Path = _PROJECT_ROOT / "configs" / "feature_engineering.yaml",
    skip_validation: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Run P0-E3 monthly feature-build orchestration."""
    cfg = _load_feature_config(config_path)
    gap_threshold_sec = float(cfg.get("dbdt", {}).get("gap_threshold_sec", 90.0))
    paths = _month_paths(year, month)

    create_directories(_PROJECT_ROOT)
    if dry_run:
        print("Feature dry run:")
        print(f"  year={year}")
        print(f"  month={month}")
        print(f"  normalize_goes_xray={paths['goes_xray_raw']} -> {paths['goes_xray_normalized']}")
        print(f"  compute_all_station_dbdt={paths['supermag_raw']} -> {paths['supermag_processed']}")
        print(f"  build_leo_index_month={year:04d}-{month:02d}")
        print(f"  build_feature_matrix={paths['features']}")
        print(f"  validate={not skip_validation}")
        return True

    if paths["goes_xray_raw"].exists() and (force or not paths["goes_xray_normalized"].exists()):
        normalize_goes_xray(
            paths["goes_xray_raw"],
            output_path=paths["goes_xray_normalized"],
            validation_plot_path=paths["xray_validation_plot"],
            write_output=True,
        )
    elif not paths["goes_xray_raw"].exists():
        log.warning("GOES X-ray raw file missing; feature builder will use any existing normalized/raw fallback: %s", paths["goes_xray_raw"])

    if force or not paths["supermag_processed"].exists():
        compute_all_station_dbdt(
            paths["supermag_raw"],
            gap_threshold_sec=gap_threshold_sec,
            output_path=paths["supermag_processed"],
            write_output=True,
        )

    build_leo_index_month(year, month)
    build_feature_matrix(year, month)

    if skip_validation:
        return True
    return bool(validate_feature_matrix(year, month, fail_on_error=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate monthly features.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing feature matrix without running feature build.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip monthly feature-matrix validation after build.",
    )
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "feature_engineering.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate derived monthly products when possible.")
    args = parser.parse_args(argv)

    if args.validate_only:
        return 0 if validate_feature_matrix(args.year, args.month, fail_on_error=False) else 1

    ok = run_feature_pipeline(
        args.year,
        args.month,
        config_path=args.config,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
