"""Validate LEO index physics against multi-station SuperMAG dB/dt targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_PREFIX = "dbdt_horizontal_magnitude_"
LEO_PREFIXES = ("leo_high_lat", "leo_mid_lat", "leo_dayside", "leo_nightside", "leo_index_global")
DMSP_DEFER_R2_THRESHOLD = 0.10


@dataclass(frozen=True)
class LeoValidationResult:
    """Summary of the LEO-index validation decision."""

    best_leo_feature: str
    best_lag_min: int
    best_correlation: float
    variance_explained: float
    dmsp_recommendation: str
    report_path: str


def _target_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(col for col in df.columns if col.startswith(TARGET_PREFIX))
    if not cols:
        raise ValueError(f"No target columns found with prefix {TARGET_PREFIX!r}.")
    return cols


def _leo_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        col
        for col in df.columns
        if any(col == prefix or col.startswith(f"{prefix}_") for prefix in LEO_PREFIXES)
        and not col.endswith(("_count", "_decay_age", "_is_fresh"))
    ]
    if not cols:
        raise ValueError("No LEO index columns found.")
    return sorted(cols)


def load_validation_features(features_dir: str | Path) -> pd.DataFrame:
    """Load feature matrices for validation from a directory or single Parquet."""
    path = Path(features_dir)
    paths = [path] if path.is_file() else sorted(path.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature parquet files found under {path}")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        raise KeyError("Feature matrix is missing required timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Feature matrix contains null or unparseable timestamps.")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def add_global_supermag_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a global mean SuperMAG target across available station targets."""
    out = df.copy()
    targets = _target_columns(out)
    out["supermag_global_dbdt"] = out[targets].mean(axis=1, skipna=True)
    out["supermag_station_valid_count"] = out[targets].notna().sum(axis=1)
    return out


def lag_correlation_table(
    df: pd.DataFrame,
    *,
    lags_min: Iterable[int] = range(-180, 181, 10),
    target_col: str = "supermag_global_dbdt",
) -> pd.DataFrame:
    """Compute LEO-index correlation against a target over minute lags.

    Positive lag means the LEO feature is evaluated earlier than the target
    by that many one-minute rows.
    """
    leo_cols = _leo_columns(df)
    rows = []
    target = pd.to_numeric(df[target_col], errors="coerce")
    for leo_col in leo_cols:
        leo = pd.to_numeric(df[leo_col], errors="coerce")
        for lag in lags_min:
            shifted = leo.shift(int(lag))
            valid = shifted.notna() & target.notna()
            if int(valid.sum()) < 3:
                corr = float("nan")
            else:
                corr = float(shifted[valid].corr(target[valid]))
            rows.append(
                {
                    "leo_feature": leo_col,
                    "lag_min": int(lag),
                    "correlation": corr,
                    "abs_correlation": abs(corr) if pd.notna(corr) else float("nan"),
                    "variance_explained": corr * corr if pd.notna(corr) else float("nan"),
                    "n_valid": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["abs_correlation", "n_valid"], ascending=[False, False])


def station_correlation_table(
    df: pd.DataFrame,
    best_leo_feature: str,
    best_lag_min: int,
) -> pd.DataFrame:
    """Compute per-station correlation for the selected LEO feature and lag."""
    shifted = pd.to_numeric(df[best_leo_feature], errors="coerce").shift(int(best_lag_min))
    rows = []
    for target_col in _target_columns(df):
        station = target_col.removeprefix(TARGET_PREFIX)
        target = pd.to_numeric(df[target_col], errors="coerce")
        valid = shifted.notna() & target.notna()
        corr = float(shifted[valid].corr(target[valid])) if int(valid.sum()) >= 3 else float("nan")
        rows.append(
            {
                "station": station,
                "correlation": corr,
                "abs_correlation": abs(corr) if pd.notna(corr) else float("nan"),
                "n_valid": int(valid.sum()),
                "glat": _station_context_value(df, "glat", station),
                "glon": _station_context_value(df, "glon", station),
                "mlat": _station_context_value(df, "mlat", station),
                "mlon": _station_context_value(df, "mlon", station),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def _station_context_value(df: pd.DataFrame, field: str, station: str) -> float:
    col = f"{field}_{station}"
    if col not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def plot_lag_scan(lag_table: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature, group in lag_table.groupby("leo_feature"):
        ordered = group.sort_values("lag_min")
        ax.plot(ordered["lag_min"], ordered["variance_explained"], marker="o", label=feature)
    ax.axhline(DMSP_DEFER_R2_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (min; positive = LEO leads target)")
    ax.set_ylabel("Variance explained (R^2)")
    ax.set_title("LEO Index Lag Scan Against Global SuperMAG dB/dt")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_station_spatial_correlations(station_table: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = station_table.dropna(subset=["glat", "glon", "correlation"])
    if valid.empty:
        ax.text(0.5, 0.5, "No station coordinates available", ha="center", va="center")
        ax.set_axis_off()
    else:
        scatter = ax.scatter(valid["glon"], valid["glat"], c=valid["correlation"], cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xlabel("Geographic longitude")
        ax.set_ylabel("Geographic latitude")
        ax.set_title("Station Correlation With Selected LEO Index")
        fig.colorbar(scatter, ax=ax, label="Correlation")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_validation_report(
    result: LeoValidationResult,
    *,
    output_path: str | Path,
    lag_plot_path: str | Path,
    spatial_plot_path: str | Path,
) -> None:
    text = f"""# LEO Index Validation

Best LEO feature: `{result.best_leo_feature}`

Best lag: `{result.best_lag_min}` minutes, where positive means LEO leads the SuperMAG target.

Best correlation: `{result.best_correlation:.4f}`

Variance explained: `{result.variance_explained:.4f}`

DMSP recommendation: **{result.dmsp_recommendation}**

Decision threshold: defer DMSP if LEO index explains at least `{DMSP_DEFER_R2_THRESHOLD:.2f}` of global SuperMAG dB/dt variance.

Figures:

- Lag scan: `{Path(lag_plot_path).name}`
- Station spatial correlations: `{Path(spatial_plot_path).name}`
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")


def run_leo_index_validation(
    *,
    features_dir: str | Path,
    output_dir: str | Path = "results/validation/leo_index",
    lags_min: Iterable[int] = range(-180, 181, 10),
) -> LeoValidationResult:
    """Run the P0-C1 validation workflow and write tables, figures, and report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df = add_global_supermag_target(load_validation_features(features_dir))

    lag_table = lag_correlation_table(df, lags_min=lags_min)
    if lag_table.empty or lag_table["variance_explained"].dropna().empty:
        raise ValueError("LEO validation produced no valid lag correlations.")
    best = lag_table.iloc[0]
    station_table = station_correlation_table(
        df,
        best_leo_feature=str(best["leo_feature"]),
        best_lag_min=int(best["lag_min"]),
    )

    lag_csv = output_path / "leo_lag_correlations.csv"
    station_csv = output_path / "leo_station_correlations.csv"
    lag_table.to_csv(lag_csv, index=False)
    station_table.to_csv(station_csv, index=False)

    lag_plot = output_path / "leo_lag_scan.png"
    spatial_plot = output_path / "leo_station_spatial_correlations.png"
    plot_lag_scan(lag_table, lag_plot)
    plot_station_spatial_correlations(station_table, spatial_plot)

    variance_explained = float(best["variance_explained"])
    recommendation = "defer DMSP" if variance_explained >= DMSP_DEFER_R2_THRESHOLD else "escalate DMSP review"
    report = output_path / "leo_index_validation_report.md"
    result = LeoValidationResult(
        best_leo_feature=str(best["leo_feature"]),
        best_lag_min=int(best["lag_min"]),
        best_correlation=float(best["correlation"]),
        variance_explained=variance_explained,
        dmsp_recommendation=recommendation,
        report_path=str(report),
    )
    write_validation_report(result, output_path=report, lag_plot_path=lag_plot, spatial_plot_path=spatial_plot)
    (output_path / "leo_index_validation_summary.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
