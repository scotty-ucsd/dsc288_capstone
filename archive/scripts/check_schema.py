"""
check_schema.py
CLI inspection tool for pipeline Parquet files.
Delegates structural validation to schema.validate_output_schema.

Usage:
    uv run python scripts/check_schema.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger
from schema import validate_output_schema

log = get_logger(__name__)

_YEAR, _MONTH = 2015, 3
_MS = f"{_YEAR:04d}{_MONTH:02d}"

FILES = {
    "omni_l1":    os.path.join(config.PROCESSED_DIR, "omni",    "2015", "03", f"omni_{_MS}.parquet"),
    "goes_gsm":   os.path.join(config.PROCESSED_DIR, "goes",    "2015", "03", f"goes15_{_MS}.parquet"),
    "leo_index":  os.path.join(config.PROCESSED_DIR, "swarm",   "2015", "03", f"swarm_leo_index_{_MS}.parquet"),
    "supermag":   os.path.join(config.PROCESSED_DIR, "supermag","2015", "03", f"supermag_{_MS}.parquet"),
    "swarm_a_raw":os.path.join(config.RAW_DATA_DIR,  "swarm",   "2015", "03", f"swarmA_LR1B_{_MS}.parquet"),
    "swarm_b_raw":os.path.join(config.RAW_DATA_DIR,  "swarm",   "2015", "03", f"swarmB_LR1B_{_MS}.parquet"),
    "swarm_c_raw":os.path.join(config.RAW_DATA_DIR,  "swarm",   "2015", "03", f"swarmC_LR1B_{_MS}.parquet"),
}


def _preview_value(val, max_len=120):
    try:
        if isinstance(val, np.ndarray):
            arr = np.asarray(val).reshape(-1)
            return f"ndarray shape={val.shape} preview={np.array2string(arr[:6], precision=3)}"
        text = repr(val)
        return text[:max_len] + "..." if len(text) > max_len else text
    except Exception as e:
        return f"<preview_error: {e}>"


def inspect_parquet(label: str, path: str, object_preview_rows: int = 3) -> None:
    """Print a detailed schema report and run validate_output_schema."""
    print("=" * 88)
    print(f"{label}")
    print(f"path: {path}")

    if not os.path.exists(path):
        print("status: MISSING")
        log.warning("File not found during schema check: %s", path)
        return

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"status: READ_ERROR — {exc}")
        log.error("Could not read %s: %s", path, exc)
        return

    nrows = len(df)
    print(f"status: OK | rows={nrows} | cols={len(df.columns)}")

    # Timestamp quick-look
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        log.debug("Timestamp range for %s: %s → %s", label, ts.min(), ts.max())
        print(f"timestamp: min={ts.min()} | max={ts.max()} | "
              f"null={int(ts.isna().sum())} | dup_rows={int(ts.duplicated(keep=False).sum())}")
    else:
        print("timestamp: <missing>")

    # Column dtypes
    print("\ndtypes:")
    for c in df.columns:
        print(f"  {c:<35} {str(df[c].dtype)}")

    # Null summary
    print("\nnull summary:")
    null_counts = df.isna().sum().sort_values(ascending=False)
    for col, cnt in null_counts.items():
        if cnt > 0:
            print(f"  {col:<35} nulls={int(cnt):<10} frac={100*cnt/max(nrows,1):.2f}%")
    if int(null_counts.sum()) == 0:
        print("  <no nulls>")

    # Object column previews
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if object_cols:
        print("\nobject-like column previews:")
        for col in object_cols:
            nonnull = df[col].dropna()
            print(f"  {col}:")
            for i, val in enumerate(nonnull.head(object_preview_rows)):
                print(f"    [{i}] type={type(val).__name__} value={_preview_value(val)}")

    # Delegate structural validation to schema.py
    try:
        validate_output_schema(df, label)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"\nSCHEMA VIOLATION: {exc}")
        log.error("Schema violation for %s: %s", label, exc)

    print()


def main() -> None:
    for label, path in FILES.items():
        inspect_parquet(label, path)


if __name__ == "__main__":
    main()
