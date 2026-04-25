#!/usr/bin/env python3
"""Train and evaluate P0 M0/M1/M2 multi-output baseline models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.setup_directories import create_directories
from swmi.models.baseline_lstm import MultiOutputBaseline, train_and_evaluate_baselines
from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants
from swmi.utils.logger import get_logger

log = get_logger(__name__)


def _load_model_config(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_scientific_invariants(cfg, str(config_path))
    return cfg


def _default_split_path(sequences_dir: str | Path, split: str) -> Path:
    return Path(sequences_dir) / split / f"sequences_{split}.npz"


def _json_ready(value):
    """Convert nested metrics to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return value


def run_training_pipeline(
    *,
    sequences_dir: str | Path = config.SEQUENCES_DIR,
    train_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    eval_split: str = "val",
    config_path: str | Path = _PROJECT_ROOT / "configs" / "model_baseline.yaml",
    output_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    dry_run: bool = False,
    gbm_max_iter_override: int | None = None,
) -> tuple[MultiOutputBaseline, dict[str, dict]] | None:
    """Run P0-E5 baseline training orchestration."""
    cfg = _load_model_config(config_path)
    experiment_cfg = cfg.get("experiment", {})
    m1_cfg = cfg.get("m1_loglinear", {})
    m2_cfg = cfg.get("m2_gbm", {})

    sequences_root = Path(sequences_dir)
    final_train_path = Path(train_path) if train_path is not None else _default_split_path(sequences_root, "train")
    final_eval_path = Path(eval_path) if eval_path is not None else _default_split_path(sequences_root, eval_split)
    final_output_dir = Path(output_dir or experiment_cfg.get("output_dir", "results/experiments/exp001_baseline"))
    final_checkpoint_dir = Path(checkpoint_dir or experiment_cfg.get("checkpoint_dir", "models/checkpoints/exp001"))

    if not final_train_path.exists():
        raise FileNotFoundError(f"Training sequence file not found: {final_train_path}")
    if not final_eval_path.exists():
        raise FileNotFoundError(f"Evaluation sequence file not found: {final_eval_path}")

    ridge_alpha = float(m1_cfg.get("alpha", 1.0))
    ridge_fit_intercept = bool(m1_cfg.get("fit_intercept", True))
    gbm_max_iter = int(gbm_max_iter_override or m2_cfg.get("n_estimators", 500))
    gbm_learning_rate = float(m2_cfg.get("learning_rate", 0.05))
    gbm_min_samples_leaf = int(m2_cfg.get("min_child_samples", 50))

    create_directories(_PROJECT_ROOT)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("Training dry run:")
        print(f"  train_path={final_train_path}")
        print(f"  eval_path={final_eval_path}")
        print(f"  output_dir={final_output_dir}")
        print(f"  checkpoint_dir={final_checkpoint_dir}")
        print(f"  ridge_alpha={ridge_alpha}")
        print(f"  gbm_max_iter={gbm_max_iter}")
        return None

    baseline, metrics = train_and_evaluate_baselines(
        final_train_path,
        final_eval_path,
        output_dir=final_checkpoint_dir,
        ridge_alpha=ridge_alpha,
        ridge_fit_intercept=ridge_fit_intercept,
        gbm_max_iter=gbm_max_iter,
        gbm_learning_rate=gbm_learning_rate,
        gbm_min_samples_leaf=gbm_min_samples_leaf,
    )

    metrics_path = final_output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(metrics), fh, indent=2, sort_keys=True)
    log.info("Training complete. Metrics saved to %s", metrics_path)
    return baseline, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SWMI baseline models.")
    parser.add_argument("--sequences-dir", default=config.SEQUENCES_DIR)
    parser.add_argument("--train-path")
    parser.add_argument("--eval-path")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "model_baseline.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_training_pipeline(
        sequences_dir=args.sequences_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_split=args.eval_split,
        config_path=args.config,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
