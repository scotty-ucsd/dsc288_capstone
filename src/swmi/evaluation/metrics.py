"""Evaluation metrics for multi-station dB/dt forecasts."""

from __future__ import annotations

import numpy as np

from swmi.training.losses import nan_masked_mae_numpy, nan_masked_mse_numpy


def _valid_mask(y_true: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    true = np.asarray(y_true, dtype=np.float64)
    valid = ~np.isnan(true)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != true.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} does not match target shape {true.shape}")
        valid &= mask_arr
    return valid


def _station_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray,
    station_idx: int,
) -> dict[str, float | int]:
    station_valid = valid[:, station_idx]
    n_valid = int(station_valid.sum())
    if n_valid == 0:
        return {"n_valid": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    errors = y_pred[station_valid, station_idx] - y_true[station_valid, station_idx]
    mse = float(np.mean(errors**2))
    return {
        "n_valid": n_valid,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
    }


def multioutput_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    stations: list[str] | np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute global and per-station metrics using NaN-aware masks."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.ndim != 2:
        raise ValueError(f"Expected 2D multi-output arrays, got shape {true.shape}")

    valid = _valid_mask(true, mask)
    station_labels = [str(station) for station in (stations if stations is not None else range(true.shape[1]))]
    if len(station_labels) != true.shape[1]:
        raise ValueError("Number of station labels does not match target width.")

    global_mse = nan_masked_mse_numpy(pred, true, mask=valid)
    global_mae = nan_masked_mae_numpy(pred, true, mask=valid)
    per_station = {
        station: _station_metric(true, pred, valid, idx)
        for idx, station in enumerate(station_labels)
    }
    return {
        "global": {
            "n_valid": int(valid.sum()),
            "mse": global_mse,
            "rmse": float(np.sqrt(global_mse)) if not np.isnan(global_mse) else float("nan"),
            "mae": global_mae,
        },
        "per_station": per_station,
    }
