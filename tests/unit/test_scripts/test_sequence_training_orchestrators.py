"""Tests for sequence and training orchestrator scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def _write_model_config(path: Path) -> None:
    path.write_text(
        """
forecast_horizon_min: 2
split_buffer_days: 7
sequence:
  input_window_min: 5
  stride_min: 10
  max_gap_fraction: 0.10
m1_loglinear:
  alpha: 1.0
  fit_intercept: true
m2_gbm:
  n_estimators: 5
  learning_rate: 0.05
  min_child_samples: 2
experiment:
  output_dir: results/experiments/exp001_baseline
  checkpoint_dir: models/checkpoints/exp001
""".strip().replace("forecast_horizon_min: 2", "forecast_horizon_min: 60"),
        encoding="utf-8",
    )


def _feature_df(periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "omni_bz_gsm": np.linspace(-5.0, 5.0, periods),
            "newell_phi": np.linspace(1.0, 2.0, periods),
            "ut_sin": np.sin(np.arange(periods)),
            "ut_cos": np.cos(np.arange(periods)),
            "dbdt_horizontal_magnitude_ABK": np.linspace(0.0, 1.0, periods),
            "dbdt_horizontal_magnitude_TRO": np.linspace(1.0, 2.0, periods),
            "dbdt_missing_flag_ABK": [0] * periods,
            "dbdt_missing_flag_TRO": [0] * periods,
            "mlt_ABK": [12.0] * periods,
            "mlt_TRO": [13.0] * periods,
        }
    )


def _write_npz(path: Path, *, n_samples: int = 16) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, 4, 3)).astype(np.float32)
    y = np.column_stack([X[:, -1, 0], X[:, -1, 1]]).astype(np.float32)
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


def test_sequence_orchestrator_builds_npz_outputs(tmp_path: Path) -> None:
    module = _load_script("scripts/03_build_sequences.py", "build_sequences_script")
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "sequences"
    scaler_dir = tmp_path / "artifacts"
    config_path = tmp_path / "model_baseline.yaml"
    features_dir.mkdir()
    _write_model_config(config_path)
    _feature_df().to_parquet(features_dir / "features_202001.parquet", index=False)

    result = module.run_sequence_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        scaler_dir=scaler_dir,
        config_path=config_path,
    )

    assert result is not None
    assert (output_dir / "train" / "sequences_train.npz").exists()
    assert (scaler_dir / "scaler_v1.pkl").exists()


def test_sequence_orchestrator_dry_run_does_not_write(tmp_path: Path) -> None:
    module = _load_script("scripts/03_build_sequences.py", "build_sequences_script_dry")
    features_dir = tmp_path / "features"
    config_path = tmp_path / "model_baseline.yaml"
    features_dir.mkdir()
    _write_model_config(config_path)

    result = module.run_sequence_pipeline(
        features_dir=features_dir,
        output_dir=tmp_path / "sequences",
        scaler_dir=tmp_path / "artifacts",
        config_path=config_path,
        dry_run=True,
    )

    assert result is None
    assert not (tmp_path / "sequences").exists()


def test_training_orchestrator_trains_and_writes_metrics(tmp_path: Path) -> None:
    module = _load_script("scripts/04_train_baseline.py", "train_baseline_script")
    sequences_dir = tmp_path / "sequences"
    train_dir = sequences_dir / "train"
    val_dir = sequences_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    _write_npz(train_dir / "sequences_train.npz", n_samples=20)
    _write_npz(val_dir / "sequences_val.npz", n_samples=10)
    config_path = tmp_path / "model_baseline.yaml"
    _write_model_config(config_path)

    result = module.run_training_pipeline(
        sequences_dir=sequences_dir,
        config_path=config_path,
        output_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result is not None
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "checkpoints" / "multioutput_baseline.joblib").exists()


def test_training_orchestrator_dry_run_does_not_train(tmp_path: Path) -> None:
    module = _load_script("scripts/04_train_baseline.py", "train_baseline_script_dry")
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "val.npz"
    _write_npz(train_path)
    _write_npz(eval_path)
    config_path = tmp_path / "model_baseline.yaml"
    _write_model_config(config_path)

    result = module.run_training_pipeline(
        train_path=train_path,
        eval_path=eval_path,
        config_path=config_path,
        output_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        dry_run=True,
    )

    assert result is None
    assert not (tmp_path / "checkpoints" / "multioutput_baseline.joblib").exists()
