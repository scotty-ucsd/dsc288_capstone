"""P0 multi-output baseline models for SWMI dB/dt forecasting.

Despite the historical file name, P0-S7 defines non-neural baselines:
- M0 persistence
- M1 ridge/log-linear multi-output regression
- M2 per-station gradient boosting

All models consume sequence NPZ files produced by ``swmi.sequences.builder`` and
respect station-level NaN masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from swmi.evaluation.metrics import multioutput_regression_metrics
from swmi.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SequenceDataset:
    """Loaded sequence arrays for one split."""

    X: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray
    current_y: np.ndarray | None
    current_target_mask: np.ndarray | None
    stations: list[str]
    feature_columns: list[str]
    target_columns: list[str]


def load_sequence_npz(path: str | Path) -> SequenceDataset:
    """Load a P0-S6 sequence NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        target_mask = data["target_mask"].astype(bool) if "target_mask" in data else ~np.isnan(y)
        current_y = data["current_y"].astype(np.float32) if "current_y" in data else None
        current_target_mask = data["current_target_mask"].astype(bool) if "current_target_mask" in data else None
        stations = [str(station) for station in data["stations"]]
        feature_columns = [str(col) for col in data["feature_columns"]]
        target_columns = [str(col) for col in data["target_columns"]]

    if X.ndim != 3:
        raise ValueError(f"X must have shape (samples, timesteps, features), got {X.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must have shape (samples, stations), got {y.shape}")
    if target_mask.shape != y.shape:
        raise ValueError(f"target_mask shape {target_mask.shape} does not match y shape {y.shape}")
    if current_y is not None and current_y.shape != y.shape:
        raise ValueError(f"current_y shape {current_y.shape} does not match y shape {y.shape}")
    if current_target_mask is not None and current_target_mask.shape != y.shape:
        raise ValueError(
            f"current_target_mask shape {current_target_mask.shape} does not match y shape {y.shape}"
        )
    if y.shape[1] != len(stations):
        raise ValueError("Station metadata length does not match y width.")
    return SequenceDataset(X, y, target_mask, current_y, current_target_mask, stations, feature_columns, target_columns)


def _last_timestep_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"X must be 3D, got {X.shape}")
    return X[:, -1, :].astype(np.float32)


def _signed_log1p_features(X: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed drivers while preserving sign."""
    return np.sign(X) * np.log1p(np.abs(X))


def _fill_targets_for_multioutput(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = np.asarray(y, dtype=np.float32).copy()
    mask_arr = np.asarray(mask, dtype=bool)
    for station_idx in range(filled.shape[1]):
        valid = mask_arr[:, station_idx] & ~np.isnan(filled[:, station_idx])
        if not valid.any():
            filled[:, station_idx] = 0.0
        else:
            fill_value = float(np.nanmean(filled[valid, station_idx]))
            filled[~valid, station_idx] = fill_value
    return filled


class PersistenceBaseline:
    """M0 persistence baseline using current observed station target if present."""

    name = "M0_persistence"

    def fit(self, train: SequenceDataset) -> "PersistenceBaseline":
        self.n_outputs_ = train.y.shape[1]
        self.fallback_values_ = np.nanmean(train.y, axis=0)
        self.fallback_values_ = np.where(np.isnan(self.fallback_values_), 0.0, self.fallback_values_).astype(np.float32)
        return self

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        if not hasattr(self, "fallback_values_"):
            raise RuntimeError("Model has not been fit.")
        pred = np.broadcast_to(self.fallback_values_, dataset.y.shape).astype(np.float32).copy()
        if dataset.current_y is not None:
            current_mask = ~np.isnan(dataset.current_y)
            if dataset.current_target_mask is not None:
                current_mask &= dataset.current_target_mask
            pred[current_mask] = dataset.current_y[current_mask]
        return pred


class RidgeMultiOutputBaseline:
    """M1 ridge multi-output regression on last-timestep global features."""

    name = "M1_ridge_multioutput"

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    def fit(self, train: SequenceDataset) -> "RidgeMultiOutputBaseline":
        X_train = _signed_log1p_features(_last_timestep_features(train.X))
        y_train = _fill_targets_for_multioutput(train.y, train.target_mask)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        return self.model.predict(_signed_log1p_features(_last_timestep_features(dataset.X))).astype(np.float32)


class PerStationGradientBoostingBaseline:
    """M2 independent gradient boosting models, one per station."""

    name = "M2_per_station_gradient_boosting"

    def __init__(
        self,
        *,
        max_iter: int = 500,
        max_leaf_nodes: int | None = 31,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 50,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.models_: list[HistGradientBoostingRegressor | None] = []
        self.fallback_values_: np.ndarray | None = None

    def fit(self, train: SequenceDataset) -> "PerStationGradientBoostingBaseline":
        X_train = _last_timestep_features(train.X)
        self.models_ = []
        fallback_values = np.zeros(train.y.shape[1], dtype=np.float32)
        for station_idx in range(train.y.shape[1]):
            valid = train.target_mask[:, station_idx] & ~np.isnan(train.y[:, station_idx])
            if not valid.any():
                self.models_.append(None)
                continue
            fallback_values[station_idx] = float(np.nanmean(train.y[valid, station_idx]))
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            model.fit(X_train[valid], train.y[valid, station_idx])
            self.models_.append(model)
        self.fallback_values_ = fallback_values
        return self

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        if self.fallback_values_ is None or not self.models_:
            raise RuntimeError("Model has not been fit.")
        X_eval = _last_timestep_features(dataset.X)
        pred = np.empty((X_eval.shape[0], len(self.models_)), dtype=np.float32)
        for station_idx, model in enumerate(self.models_):
            if model is None:
                pred[:, station_idx] = self.fallback_values_[station_idx]
            else:
                pred[:, station_idx] = model.predict(X_eval).astype(np.float32)
        return pred


class MultiOutputBaseline:
    """Convenience wrapper that trains and evaluates M0/M1/M2 baselines."""

    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        ridge_fit_intercept: bool = True,
        gbm_max_iter: int = 500,
        gbm_learning_rate: float = 0.05,
        gbm_min_samples_leaf: int = 50,
    ) -> None:
        self.models = {
            "M0": PersistenceBaseline(),
            "M1": RidgeMultiOutputBaseline(alpha=ridge_alpha, fit_intercept=ridge_fit_intercept),
            "M2": PerStationGradientBoostingBaseline(
                max_iter=gbm_max_iter,
                learning_rate=gbm_learning_rate,
                min_samples_leaf=gbm_min_samples_leaf,
            ),
        }

    def fit(self, train: SequenceDataset) -> "MultiOutputBaseline":
        for key, model in self.models.items():
            log.info("Fitting %s baseline...", key)
            model.fit(train)
        self.stations_ = train.stations
        self.feature_columns_ = train.feature_columns
        return self

    def predict(self, dataset: SequenceDataset) -> dict[str, np.ndarray]:
        return {key: model.predict(dataset) for key, model in self.models.items()}

    def evaluate(self, dataset: SequenceDataset) -> dict[str, dict]:
        return {
            key: multioutput_regression_metrics(
                dataset.y,
                pred,
                stations=dataset.stations,
                mask=dataset.target_mask,
            )
            for key, pred in self.predict(dataset).items()
        }

    def save(self, output_dir: str | Path, metrics: dict | None = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path / "multioutput_baseline.joblib")
        if metrics is not None:
            with open(output_path / "metrics.json", "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, sort_keys=True)


def train_and_evaluate_baselines(
    train_path: str | Path,
    eval_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    ridge_alpha: float = 1.0,
    ridge_fit_intercept: bool = True,
    gbm_max_iter: int = 500,
    gbm_learning_rate: float = 0.05,
    gbm_min_samples_leaf: int = 50,
) -> tuple[MultiOutputBaseline, dict[str, dict]]:
    """Train M0/M1/M2 on train NPZ and evaluate on another split."""
    train = load_sequence_npz(train_path)
    evaluation = load_sequence_npz(eval_path)
    baseline = MultiOutputBaseline(
        ridge_alpha=ridge_alpha,
        ridge_fit_intercept=ridge_fit_intercept,
        gbm_max_iter=gbm_max_iter,
        gbm_learning_rate=gbm_learning_rate,
        gbm_min_samples_leaf=gbm_min_samples_leaf,
    ).fit(train)
    metrics = baseline.evaluate(evaluation)
    if output_dir is not None:
        baseline.save(output_dir, metrics)
    return baseline, metrics
