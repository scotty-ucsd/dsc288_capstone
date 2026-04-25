"""
schema.py
Schema validation for all Parquet outputs in the dB/dt forecasting pipeline.

Call ``validate_output_schema(df, source_name)`` immediately before writing
any Parquet file to catch structural violations before they corrupt downstream
temporal joins or training sequences.

Conventions guaranteed by this module:
- Every output DataFrame has a column named exactly ``timestamp``.
- ``df["timestamp"]`` is ``datetime64[ns, UTC]`` (timezone-aware UTC).
- No duplicate timestamps within a single-source monthly file.
- Columns that are entirely NaN are logged at WARNING level.

TODO: Add physical range checks; fix allow_duplicates bug
"""

import pandas as pd

from logger import get_logger

log = get_logger(__name__)


def validate_output_schema(df: pd.DataFrame, source_name: str, unique_subset: list[str] | None = None) -> None:
    """Validate structural invariants for a pipeline output DataFrame.

    This function is a guard rail, not a transformer. It raises on hard
    violations and logs warnings on soft violations. The caller is
    responsible for fixing the data before calling this function.

    Parameters
    ----------
    df:
        The DataFrame to validate. Must satisfy all invariants listed below.
    source_name:
        Human-readable label for the source (e.g. ``"OMNI"``, ``"GOES-16"``).
        Used in log messages and exception text.

    Raises
    ------
    KeyError
        If the ``timestamp`` column is absent.
    TypeError
        If ``df["timestamp"]`` is not ``datetime64[ns, UTC]``.
    ValueError
        If duplicate timestamps are detected.

    Warnings
    --------
    Logs a WARNING for any column whose NaN fraction exceeds 50 %.
    Logs a WARNING for any column that is entirely NaN.

    Notes
    -----
    - The NaN fraction check is informational. It does NOT raise.
    - Duplicate detection uses ``pd.Series.duplicated(keep=False)``, which
      flags ALL rows involved in a duplicate pair (not just the second).

    Examples
    --------
    >>> import pandas as pd
    >>> from schema import validate_output_schema
    >>> df = pd.DataFrame({
    ...     "timestamp": pd.date_range("2015-03-01", periods=3, freq="1min", tz="UTC"),
    ...     "goes_bz_gsm": [-5.0, -7.0, -9.0],
    ... })
    >>> validate_output_schema(df, "GOES-15")   # passes silently
    """
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Invariant 1: 'timestamp' column must exist with this exact name.
    # ------------------------------------------------------------------
    if "timestamp" not in df.columns:
        present = list(df.columns)
        raise KeyError(
            f"[{source_name}] Missing required column 'timestamp'. "
            f"Present columns: {present}. "
            "Rename the time column to 'timestamp' before writing."
        )

    # ------------------------------------------------------------------
    # Invariant 2: timestamp must be timezone-aware UTC.
    # ------------------------------------------------------------------
    ts_dtype = df["timestamp"].dtype
    is_utc = (
        hasattr(ts_dtype, "tz")
        and ts_dtype.tz is not None
        and str(ts_dtype.tz) in ("UTC", "utc")
    )
    if not is_utc:
        raise TypeError(
            f"[{source_name}] df['timestamp'] must be datetime64[ns, UTC]. "
            f"Found dtype: {ts_dtype!r}. "
            "Use pd.to_datetime(col, utc=True) to convert."
        )

    # ------------------------------------------------------------------
    # Invariant 3: no duplicate rows across uniqueness subset
    # ------------------------------------------------------------------
    subset = unique_subset if unique_subset is not None else ["timestamp"]
    
    missing_cols = [c for c in subset if c not in df.columns]
    if missing_cols:
        raise KeyError(f"[{source_name}] Unique subset columns missing: {missing_cols}")

    dups = df.duplicated(subset=subset, keep=False)
    if dups.any():
        n_dup = int(dups.sum())
        dup_examples = df.loc[dups, subset].head(5).to_dict(orient="records")
        raise ValueError(
            f"[{source_name}] {n_dup} duplicated rows found for keys {subset}. "
            f"Examples: {dup_examples}. "
            "Remove duplicates before writing (e.g. drop_duplicates or resample)."
        )

    # ------------------------------------------------------------------
    # Soft check: per-column NaN fraction.
    # ------------------------------------------------------------------
    for col in df.columns:
        if col == "timestamp":
            continue
        null_count = int(df[col].isna().sum())
        if n_rows == 0:
            continue
        frac = null_count / n_rows
        if frac == 1.0:
            log.warning(
                "[%s] Column '%s' is entirely NaN (%d/%d rows). "
                "Possible failed join or empty retrieval.",
                source_name, col, null_count, n_rows,
            )
        elif frac > 0.5:
            log.warning(
                "[%s] Column '%s' has %.1f%% NaN (%d/%d rows).",
                source_name, col, frac * 100, null_count, n_rows,
            )

    # ------------------------------------------------------------------
    # Summary log.
    # ------------------------------------------------------------------
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    nan_fracs = {
        c: f"{df[c].isna().mean():.2%}"
        for c in df.columns
        if c != "timestamp" and df[c].isna().any()
    }
    log.info(
        "[%s] Schema OK | rows=%d | timestamp=[%s, %s] | nan_cols=%s",
        source_name,
        n_rows,
        ts_min,
        ts_max,
        nan_fracs if nan_fracs else "none",
    )
