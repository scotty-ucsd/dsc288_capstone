"""Report generation for SWMI baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path


def write_evaluation_report(
    metrics: dict,
    output_path: str | Path,
    *,
    split: str,
    checkpoint_path: str | Path,
    sequence_path: str | Path,
    figure_paths: list[str | Path] | None = None,
) -> Path:
    """Write a compact Markdown evaluation report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = [Path(fig) for fig in (figure_paths or [])]

    lines = [
        "# SWMI Baseline Evaluation",
        "",
        f"- Split: `{split}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Sequence file: `{sequence_path}`",
        "",
        "## Global Metrics",
        "",
        "| Model | Valid Targets | RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for model_name, model_metrics in sorted(metrics.items()):
        global_metrics = model_metrics.get("global", {})
        lines.append(
            "| {model} | {n_valid} | {rmse:.6g} | {mae:.6g} |".format(
                model=model_name,
                n_valid=int(global_metrics.get("n_valid", 0)),
                rmse=float(global_metrics.get("rmse", float("nan"))),
                mae=float(global_metrics.get("mae", float("nan"))),
            )
        )

    if figures:
        lines.extend(["", "## Figures", ""])
        for fig in figures:
            lines.append(f"- `{fig}`")

    lines.extend(["", "## Metrics JSON", "", "```json", json.dumps(metrics, indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
