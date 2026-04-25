"""Tests for the P0-E6 evaluation orchestrator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swmi.models.baseline_lstm import MultiOutputBaseline, load_sequence_npz  # noqa: E402


def _load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model_config(path: Path, checkpoint_dir: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
forecast_horizon_min: 60
split_buffer_days: 7
sequence:
  input_window_min: 120
  stride_min: 1
  max_gap_fraction: 0.10
m1_loglinear:
  alpha: 1.0
  fit_intercept: true
m2_gbm:
  n_estimators: 5
  learning_rate: 0.05
  min_child_samples: 2
experiment:
  output_dir: "{output_dir}"
  checkpoint_dir: "{checkpoint_dir}"
""".strip(),
        encoding="utf-8",
    )


def _write_npz(path: Path, *, n_samples: int = 18) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n_samples, 4, 3)).astype(np.float32)
    y = np.column_stack([X[:, -1, 0] + 0.5, X[:, -1, 1] - 0.5]).astype(np.float32)
    y[1, 1] = np.nan
    target_mask = ~np.isnan(y)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        target_mask=target_mask,
        current_y=y - 0.1,
        current_target_mask=target_mask,
        stations=np.asarray(["ABK", "TRO"], dtype="U"),
        feature_columns=np.asarray(["f0", "f1", "f2"], dtype="U"),
        target_columns=np.asarray(
            ["dbdt_horizontal_magnitude_ABK", "dbdt_horizontal_magnitude_TRO"],
            dtype="U",
        ),
    )


def _write_checkpoint(train_path: Path, checkpoint_dir: Path) -> Path:
    dataset = load_sequence_npz(train_path)
    model = MultiOutputBaseline(gbm_max_iter=5, gbm_min_samples_leaf=2).fit(dataset)
    checkpoint_dir.mkdir(parents=True)
    model.save(checkpoint_dir)
    return checkpoint_dir / "multioutput_baseline.joblib"


def test_evaluation_orchestrator_writes_metrics_predictions_figures_and_report(tmp_path: Path) -> None:
    module = _load_script("scripts/05_evaluate_model.py", "evaluate_model_script")
    sequence_path = tmp_path / "sequences_test.npz"
    train_path = tmp_path / "sequences_train.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "model_baseline.yaml"
    _write_npz(sequence_path, n_samples=10)
    _write_npz(train_path, n_samples=20)
    checkpoint_path = _write_checkpoint(train_path, checkpoint_dir)
    _write_model_config(config_path, checkpoint_dir, output_dir)

    metrics = module.run_evaluation_pipeline(
        sequence_path=sequence_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
        split="test",
    )

    assert metrics is not None
    assert set(metrics) == {"M0", "M1", "M2"}
    assert (output_dir / "evaluation_metrics_test.json").exists()
    assert (output_dir / "predictions_test.npz").exists()
    assert (output_dir / "global_rmse_test.png").exists()
    assert (output_dir / "m1_scatter_test.png").exists()
    assert (output_dir / "evaluation_report_test.md").exists()


def test_evaluation_orchestrator_dry_run_does_not_write_outputs(tmp_path: Path) -> None:
    module = _load_script("scripts/05_evaluate_model.py", "evaluate_model_script_dry")
    sequence_path = tmp_path / "sequences_test.npz"
    train_path = tmp_path / "sequences_train.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "model_baseline.yaml"
    _write_npz(sequence_path, n_samples=10)
    _write_npz(train_path, n_samples=20)
    checkpoint_path = _write_checkpoint(train_path, checkpoint_dir)
    _write_model_config(config_path, checkpoint_dir, output_dir)

    result = module.run_evaluation_pipeline(
        sequence_path=sequence_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
        split="test",
        dry_run=True,
    )

    assert result is None
    assert not (output_dir / "evaluation_metrics_test.json").exists()
