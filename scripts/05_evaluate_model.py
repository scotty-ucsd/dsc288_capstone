#!/usr/bin/env python3
"""Evaluate trained SWMI baseline models on a sequence split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.setup_directories import create_directories
from swmi.evaluation.reports import write_evaluation_report
from swmi.evaluation.visualization import plot_global_rmse, plot_prediction_scatter
from swmi.models.baseline_lstm import MultiOutputBaseline, load_sequence_npz
from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants
from swmi.utils.logger import get_logger

log = get_logger(__name__)


def _load_model_config(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_scientific_invariants(cfg, str(config_path))
    return cfg


def _default_sequence_path(sequences_dir: str | Path, split: str) -> Path:
    return Path(sequences_dir) / split / f"sequences_{split}.npz"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _load_baseline(checkpoint_path: str | Path) -> MultiOutputBaseline:
    model = joblib.load(checkpoint_path)
    if not isinstance(model, MultiOutputBaseline):
        raise TypeError(f"Expected MultiOutputBaseline checkpoint, got {type(model).__name__}")
    return model


def run_evaluation_pipeline(
    *,
    sequences_dir: str | Path = config.SEQUENCES_DIR,
    sequence_path: str | Path | None = None,
    split: str = "test",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = _PROJECT_ROOT / "configs" / "model_baseline.yaml",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict] | None:
    """Run P0-E6 evaluation orchestration."""
    cfg = _load_model_config(config_path)
    experiment_cfg = cfg.get("experiment", {})
    final_sequence_path = Path(sequence_path) if sequence_path is not None else _default_sequence_path(sequences_dir, split)
    final_checkpoint_path = Path(
        checkpoint_path
        or Path(experiment_cfg.get("checkpoint_dir", "models/checkpoints/exp001")) / "multioutput_baseline.joblib"
    )
    final_output_dir = Path(output_dir or experiment_cfg.get("output_dir", "results/experiments/exp001_baseline"))

    if not final_sequence_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {final_sequence_path}")
    if not final_checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {final_checkpoint_path}")

    create_directories(_PROJECT_ROOT)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("Evaluation dry run:")
        print(f"  split={split}")
        print(f"  sequence_path={final_sequence_path}")
        print(f"  checkpoint_path={final_checkpoint_path}")
        print(f"  output_dir={final_output_dir}")
        return None

    model = _load_baseline(final_checkpoint_path)
    dataset = load_sequence_npz(final_sequence_path)
    predictions = model.predict(dataset)
    metrics = model.evaluate(dataset)

    metrics_path = final_output_dir / f"evaluation_metrics_{split}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(_json_ready(metrics), fh, indent=2, sort_keys=True)

    predictions_path = final_output_dir / f"predictions_{split}.npz"
    np.savez_compressed(
        predictions_path,
        **{f"pred_{name}": pred for name, pred in predictions.items()},
        y_true=dataset.y,
        target_mask=dataset.target_mask,
        stations=np.asarray(dataset.stations, dtype="U"),
    )

    figure_paths = [
        plot_global_rmse(metrics, final_output_dir / f"global_rmse_{split}.png"),
    ]
    if "M1" in predictions:
        figure_paths.append(
            plot_prediction_scatter(
                dataset.y,
                predictions["M1"],
                final_output_dir / f"m1_scatter_{split}.png",
                model_name="M1",
            )
        )

    write_evaluation_report(
        metrics,
        final_output_dir / f"evaluation_report_{split}.md",
        split=split,
        checkpoint_path=final_checkpoint_path,
        sequence_path=final_sequence_path,
        figure_paths=figure_paths,
    )
    log.info("Evaluation complete. Metrics saved to %s", metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SWMI baseline model outputs.")
    parser.add_argument("--sequences-dir", default=config.SEQUENCES_DIR)
    parser.add_argument("--sequence-path")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "model_baseline.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_evaluation_pipeline(
        sequences_dir=args.sequences_dir,
        sequence_path=args.sequence_path,
        split=args.split,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
