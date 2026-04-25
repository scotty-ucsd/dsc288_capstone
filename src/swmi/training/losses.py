"""NaN-aware losses for multi-station dB/dt targets."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency
    torch = None  # type: ignore[assignment]


def nan_masked_mse_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean squared error over valid station targets only."""
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred={pred.shape}, y_true={true.shape}")

    valid = ~np.isnan(true)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != true.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} does not match target shape {true.shape}")
        valid &= mask_arr
    if not valid.any():
        return float("nan")
    return float(np.mean((pred[valid] - true[valid]) ** 2))


def nan_masked_mae_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute error over valid station targets only."""
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred={pred.shape}, y_true={true.shape}")

    valid = ~np.isnan(true)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != true.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} does not match target shape {true.shape}")
        valid &= mask_arr
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - true[valid])))


def nan_masked_mse_torch(y_pred, y_true, mask=None):
    """Torch MSE over valid targets, preserving differentiability."""
    if torch is None:
        raise ImportError("torch is required for nan_masked_mse_torch")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}")

    valid = ~torch.isnan(y_true)
    if mask is not None:
        if mask.shape != y_true.shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match target shape {tuple(y_true.shape)}")
        valid = valid & mask.bool()
    if not torch.any(valid):
        return torch.tensor(float("nan"), dtype=y_pred.dtype, device=y_pred.device)
    return torch.mean((y_pred[valid] - y_true[valid]) ** 2)
