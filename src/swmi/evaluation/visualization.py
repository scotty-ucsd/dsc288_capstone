"""Plotting utilities for SWMI evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_global_rmse(metrics: dict, output_path: str | Path) -> Path:
    """Create a bar chart of global RMSE by model."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = sorted(metrics)
    rmse = [float(metrics[name]["global"].get("rmse", np.nan)) for name in model_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(model_names, rmse)
    ax.set_ylabel("RMSE (nT/min)")
    ax.set_title("Global dB/dt Forecast RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    *,
    model_name: str,
    max_points: int = 5000,
) -> Path:
    """Create observed-vs-predicted scatter for all valid station targets."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = ~np.isnan(y_true)
    observed = y_true[valid]
    predicted = y_pred[valid]
    if len(observed) > max_points:
        idx = np.linspace(0, len(observed) - 1, max_points).astype(int)
        observed = observed[idx]
        predicted = predicted[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(observed, predicted, s=8, alpha=0.4)
    if len(observed) > 0:
        vmin = float(np.nanmin([observed.min(), predicted.min()]))
        vmax = float(np.nanmax([observed.max(), predicted.max()]))
        ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1)
    ax.set_xlabel("Observed dB/dt (nT/min)")
    ax.set_ylabel("Predicted dB/dt (nT/min)")
    ax.set_title(f"{model_name} Observed vs Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
