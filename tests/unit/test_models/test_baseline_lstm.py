"""Tests for P0 multi-output baseline models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.evaluation.metrics import multioutput_regression_metrics  # noqa: E402
from swmi.models.baseline_lstm import (  # noqa: E402
    MultiOutputBaseline,
    PersistenceBaseline,
    load_sequence_npz,
    train_and_evaluate_baselines,
)
from swmi.training.losses import nan_masked_mse_numpy  # noqa: E402


def _write_npz(path: Path, *, n_samples: int = 16) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 4, 3)).astype(np.float32)
    y = np.column_stack(
        [
            X[:, -1, 0] + 1.0,
            X[:, -1, 1] - 1.0,
        ]
    ).astype(np.float32)
    y[1, 1] = np.nan
    target_mask = ~np.isnan(y)
    current_y = y - 0.25
    current_y[2, 0] = np.nan
    current_target_mask = ~np.isnan(current_y)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        target_mask=target_mask,
        current_y=current_y,
        current_target_mask=current_target_mask,
        stations=np.asarray(["ABK", "TRO"], dtype="U"),
        feature_columns=np.asarray(["f0", "f1", "f2"], dtype="U"),
        target_columns=np.asarray(
            ["dbdt_horizontal_magnitude_ABK", "dbdt_horizontal_magnitude_TRO"],
            dtype="U",
        ),
    )


def test_load_sequence_npz_validates_multi_output_contract(tmp_path: Path) -> None:
    path = tmp_path / "sequences_train.npz"
    _write_npz(path)

    dataset = load_sequence_npz(path)

    assert dataset.X.shape == (16, 4, 3)
    assert dataset.y.shape == (16, 2)
    assert dataset.target_mask.shape == dataset.y.shape
    assert dataset.current_y is not None
    assert dataset.stations == ["ABK", "TRO"]


def test_nan_masked_mse_ignores_missing_targets() -> None:
    y_true = np.array([[1.0, np.nan], [3.0, 5.0]])
    y_pred = np.array([[2.0, 100.0], [1.0, 1.0]])

    mse = nan_masked_mse_numpy(y_pred, y_true)

    assert mse == pytest.approx((1.0 + 4.0 + 16.0) / 3.0)


def test_persistence_uses_current_targets_without_future_y_leakage(tmp_path: Path) -> None:
    path = tmp_path / "sequences_train.npz"
    _write_npz(path)
    dataset = load_sequence_npz(path)

    model = PersistenceBaseline().fit(dataset)
    pred = model.predict(dataset)

    assert pred[0, 0] == pytest.approx(dataset.current_y[0, 0])
    assert pred[2, 0] != pytest.approx(dataset.y[2, 0])


def test_multioutput_baseline_trains_and_evaluates(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "val.npz"
    output_dir = tmp_path / "checkpoint"
    _write_npz(train_path, n_samples=20)
    _write_npz(eval_path, n_samples=10)

    baseline, metrics = train_and_evaluate_baselines(
        train_path,
        eval_path,
        output_dir=output_dir,
        gbm_max_iter=5,
        gbm_min_samples_leaf=2,
    )

    assert isinstance(baseline, MultiOutputBaseline)
    assert set(metrics) == {"M0", "M1", "M2"}
    assert metrics["M1"]["global"]["n_valid"] > 0
    assert (output_dir / "multioutput_baseline.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_multioutput_metrics_include_per_station_counts() -> None:
    y_true = np.array([[1.0, np.nan], [2.0, 4.0]])
    y_pred = np.array([[1.5, 0.0], [1.0, 5.0]])

    metrics = multioutput_regression_metrics(y_true, y_pred, stations=["ABK", "TRO"])

    assert metrics["global"]["n_valid"] == 3
    assert metrics["per_station"]["ABK"]["n_valid"] == 2
    assert metrics["per_station"]["TRO"]["n_valid"] == 1
