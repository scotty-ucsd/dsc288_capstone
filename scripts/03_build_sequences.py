#!/usr/bin/env python3
"""Build train/val/test sequence NPZ files for the SWMI pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.setup_directories import create_directories
from swmi.sequences.builder import SequenceBuildResult, build_sequences
from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants
from swmi.utils.logger import get_logger

log = get_logger(__name__)


def _load_model_config(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_scientific_invariants(cfg, str(config_path))
    return cfg


def run_sequence_pipeline(
    *,
    features_dir: str | Path = config.FEATURES_DIR,
    output_dir: str | Path = config.SEQUENCES_DIR,
    scaler_dir: str | Path | None = config.ARTIFACTS_DIR,
    config_path: str | Path = _PROJECT_ROOT / "configs" / "model_baseline.yaml",
    dry_run: bool = False,
) -> SequenceBuildResult | None:
    """Run P0-E4 sequence orchestration."""
    cfg = _load_model_config(config_path)
    sequence_cfg = cfg.get("sequence", {})
    input_window_min = int(sequence_cfg.get("input_window_min", 120))
    stride_min = int(sequence_cfg.get("stride_min", 1))
    max_gap_fraction = float(sequence_cfg.get("max_gap_fraction", 0.10))
    forecast_horizon_min = int(cfg.get("forecast_horizon_min", config.FORECAST_HORIZON_MIN))

    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {features_path}")

    create_directories(_PROJECT_ROOT)
    if dry_run:
        print("Sequence dry run:")
        print(f"  features_dir={features_path}")
        print(f"  output_dir={Path(output_dir)}")
        print(f"  scaler_dir={Path(scaler_dir) if scaler_dir is not None else None}")
        print(f"  input_window_min={input_window_min}")
        print(f"  forecast_horizon_min={forecast_horizon_min}")
        print(f"  max_gap_fraction={max_gap_fraction}")
        print(f"  stride_min={stride_min}")
        return None

    result = build_sequences(
        features_dir=features_path,
        output_dir=output_dir,
        scaler_dir=scaler_dir,
        input_window_min=input_window_min,
        forecast_horizon_min=forecast_horizon_min,
        max_gap_fraction=max_gap_fraction,
        stride_min=stride_min,
    )
    log.info("Sequence build complete: split_counts=%s", result.split_counts)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SWMI sequence NPZ files.")
    parser.add_argument("--features-dir", default=config.FEATURES_DIR)
    parser.add_argument("--output-dir", default=config.SEQUENCES_DIR)
    parser.add_argument("--scaler-dir", default=config.ARTIFACTS_DIR)
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "model_baseline.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_sequence_pipeline(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        scaler_dir=args.scaler_dir,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
