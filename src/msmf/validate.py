"""Data validation utilities for space weather datasets.

Every processing script MUST use validate_output() to catch NaN issues
in space-weather data (per project rules).

Use validate_phase_shift() after building supervised tables to catch accidental
forward shifts of targets or raw fields into feature rows (positional causality).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_output(
    data: np.ndarray | pd.DataFrame,
    name: str = "data",
    max_nan_fraction: float = 0.05,
    raise_on_all_nan: bool = True,
) -> dict:
    """Validate space weather data for NaN issues and quality.

    Args:
        data: Input array or DataFrame to validate.
        name: Descriptive name for error messages.
        max_nan_fraction: Maximum allowed fraction of NaN values (default 5%).
        raise_on_all_nan: If True, raise error when all values are NaN.

    Returns:
        Dictionary with validation statistics:
            - total_values: Total number of values
            - nan_count: Number of NaN values
            - nan_fraction: Fraction of NaN values
            - valid: Whether data passes validation

    Raises:
        ValueError: If all values are NaN and raise_on_all_nan is True.
        ValueError: If NaN fraction exceeds max_nan_fraction.
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    total_values = arr.size
    nan_count = np.count_nonzero(np.isnan(arr.astype(float)))
    nan_fraction = nan_count / total_values if total_values > 0 else 0.0

    stats = {
        "total_values": total_values,
        "nan_count": nan_count,
        "nan_fraction": nan_fraction,
        "valid": True,
    }

    if total_values == 0:
        stats["valid"] = False
        raise ValueError(f"[{name}] Empty data array")

    if nan_count == total_values and raise_on_all_nan:
        stats["valid"] = False
        raise ValueError(f"[{name}] All values are NaN")

    if nan_fraction > max_nan_fraction:
        stats["valid"] = False
        raise ValueError(
            f"[{name}] NaN fraction {nan_fraction:.2%} exceeds "
            f"threshold {max_nan_fraction:.2%}"
        )

    return stats


def validate_time_series(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    expected_cadence_minutes: int = 1,
    max_gap_minutes: int = 10,
) -> dict:
    """Validate time series for gaps and cadence issues.

    Args:
        df: DataFrame with time series data.
        time_col: Name of timestamp column.
        expected_cadence_minutes: Expected time between samples.
        max_gap_minutes: Maximum allowed gap before flagging.

    Returns:
        Dictionary with gap statistics and validation status.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    times = pd.to_datetime(df[time_col])
    diffs = times.diff().dt.total_seconds() / 60  # Minutes

    gaps = diffs[diffs > max_gap_minutes]

    return {
        "total_records": len(df),
        "time_range_start": times.min(),
        "time_range_end": times.max(),
        "expected_cadence_min": expected_cadence_minutes,
        "median_cadence_min": diffs.median(),
        "gap_count": len(gaps),
        "max_gap_min": diffs.max() if len(diffs) > 0 else 0,
        "valid": len(gaps) == 0,
    }


def validate_phase_shift(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    prediction_horizon_steps: int,
    time_col: str | None = None,
    require_increasing_time: bool = True,
    max_future_row_lag: int | None = None,
    auxiliary_raw_cols: Sequence[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
) -> dict:
    """Detect forward-looking (phase-shift) leakage in time-ordered tabular features.

    **Positional causality convention (read before interpreting results).**

    This function does **not** assume uniform sampling in clock time. It uses **row
    order** as the causal order: row ``i`` is understood to occur strictly before
    row ``i+1`` in the modeling workflow. Timestamps, if provided, are used only
    to verify monotonicity when ``require_increasing_time`` is True.

    **Leakage definition implemented here.**

    Let ``T[j]`` denote the supervised target value stored in ``target_col`` at
    dataframe row index ``j`` (after any sorting applied by the caller). For a
    candidate feature column ``f`` and integer ``k > 0``:

    - ``target_col.shift(-k)`` at row ``i`` equals ``T[i+k]`` (pandas convention:
      values move "up", so row ``i`` receives the target from **later** row
      ``i+k``).

    **Leakage (target echo):** For any ``k >= 1`` up to the configured maximum,
    if the feature series ``f`` is element-wise equal (within ``rtol``/``atol``) to
    ``target_col.shift(-k)`` over a non-trivial overlap, we treat ``f`` as encoding
    information from **future rows' targets** relative to row ``i``. That violates
    strict causal construction of inputs at the prediction origin row.

    **Leakage (raw echo):** If ``auxiliary_raw_cols`` is provided (e.g. contemporaneous
    ground ``dB/dt`` or any series that must not be forward-copied into past rows),
    the same test is applied with ``target_col`` replaced by each raw column. This
    catches accidental ``.shift(-k)`` on drivers or merged fields that pulls **future
    raw observations** into the feature row for time ``i``.

    **What this does *not* check (callers must handle separately).**

    - **Valid lagged (backward) features:** ``raw.shift(+L)`` with ``L > 0`` uses
      only **past** rows relative to ``i``; it will not match ``target.shift(-k)``
      except by pathological coincidence. Such features are *intended* and are not
      flagged by this test.
    - **Rolling windows:** Centered or forward-looking windows are not inferred from
      static columns. Only backward-looking windows (or shifted summaries that the
      pipeline documents) are safe; validate window definitions in the engineering
      step.
    - **Merged external data:** If a merge realigns timestamps incorrectly
      (off-by-one, ``merge_asof`` with wrong direction), row ``i`` may leak future
      information without matching any ``shift(-k)`` pattern tested here. Re-run
      after merge with known-good alignment, or extend checks with join keys.

    Args:
        df: Time-ordered samples (one row per prediction origin). Caller must sort.
        feature_cols: Columns used as model inputs at each row.
        target_col: Supervised target column (e.g. ``dB/dt`` at ``t + H``).
        prediction_horizon_steps: Declared forecast horizon **in rows** (e.g. 60 for
            60 one-minute steps). Used as default span for ``max_future_row_lag`` and
            for documentation consistency with project LSTM framing.
        time_col: Optional timestamp column; if set and ``require_increasing_time``,
            must be strictly increasing.
        require_increasing_time: If True and ``time_col`` is set, enforce strict
            increase of timestamps.
        max_future_row_lag: Largest ``k`` tested for ``shift(-k)`` echoes. Defaults to
            ``prediction_horizon_steps``.
        auxiliary_raw_cols: Optional additional series to test for forward-shift
            duplication into features (see **raw echo** above).
        rtol, atol, equal_nan: forwarded to ``numpy.allclose`` on aligned finite pairs.

    Returns:
        Dict with ``valid`` (bool), ``violations`` (list of str), ``checks_run`` (int),
        ``time_order_ok`` (bool | None), and ``target_in_features`` (bool).

    Raises:
        ValueError: Missing columns, empty frame, or non-increasing time when required.
    """
    violations: list[str] = []
    checks_run = 0
    time_order_ok: bool | None = None

    if df.empty:
        raise ValueError("[validate_phase_shift] DataFrame is empty")

    if target_col not in df.columns:
        raise ValueError(f"[validate_phase_shift] target_col '{target_col}' missing")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[validate_phase_shift] feature_cols not in df: {missing}")

    if target_col in feature_cols:
        violations.append(
            f"target_col '{target_col}' must not appear in feature_cols (label leakage)."
        )

    feat_for_echo = [f for f in feature_cols if f != target_col]

    aux_list = list(auxiliary_raw_cols) if auxiliary_raw_cols is not None else []
    for c in aux_list:
        if c not in df.columns:
            raise ValueError(f"[validate_phase_shift] auxiliary_raw_cols: '{c}' missing")

    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"[validate_phase_shift] time_col '{time_col}' missing")
        times = pd.to_datetime(df[time_col])
        if require_increasing_time:
            if not times.is_monotonic_increasing or times.duplicated().any():
                raise ValueError(
                    f"[validate_phase_shift] '{time_col}' must be strictly increasing "
                    "when require_increasing_time=True"
                )
            time_order_ok = True
        else:
            time_order_ok = bool(times.is_monotonic_increasing)
    else:
        time_order_ok = None

    k_max = max_future_row_lag
    if k_max is None:
        k_max = max(1, int(prediction_horizon_steps))
    k_max = int(k_max)
    if k_max < 1:
        raise ValueError("[validate_phase_shift] max_future_row_lag must be >= 1")

    def _allclose_series(a: pd.Series, b: pd.Series) -> bool:
        aligned = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan)
        mask = aligned["a"].notna() | aligned["b"].notna()
        if not mask.any():
            return False
        x = aligned.loc[mask, "a"].to_numpy(dtype=float, copy=False)
        y = aligned.loc[mask, "b"].to_numpy(dtype=float, copy=False)
        return bool(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan))

    # Target echo: feature ~ target.shift(-k), k >= 1
    target_series = df[target_col]
    for k in range(1, k_max + 1):
        shifted_target = target_series.shift(-k)
        for f in feat_for_echo:
            checks_run += 1
            if _allclose_series(df[f], shifted_target):
                violations.append(
                    f"feature '{f}' matches target_col shifted forward by {k} row(s) "
                    f"(~ target at index i+{k}); suspected target leakage."
                )

    # Raw echo: feature ~ raw.shift(-k)
    for raw_col in aux_list:
        raw_series = df[raw_col]
        for k in range(1, k_max + 1):
            shifted_raw = raw_series.shift(-k)
            for f in feat_for_echo:
                if f == raw_col:
                    continue
                checks_run += 1
                if _allclose_series(df[f], shifted_raw):
                    violations.append(
                        f"feature '{f}' matches '{raw_col}' shifted forward by {k} "
                        f"row(s); suspected future raw-field leakage."
                    )

    # Contemporaneous duplicate of target (same row label abuse)
    for f in feat_for_echo:
        checks_run += 1
        if _allclose_series(df[f], target_series):
            violations.append(
                f"feature '{f}' matches target_col on the same row (contemporaneous "
                "target leakage)."
            )

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
        "time_order_ok": time_order_ok,
        "target_in_features": target_col in feature_cols,
        "max_future_row_lag": k_max,
        "prediction_horizon_steps": int(prediction_horizon_steps),
    }
