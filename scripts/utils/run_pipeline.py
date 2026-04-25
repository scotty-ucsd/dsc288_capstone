#!/usr/bin/env python3
"""
run_pipeline.py
Master orchestration script for the dB/dt geomagnetic forecasting pipeline.

Drives all retrieval and processing steps month by month to prevent API
timeouts and provide a single entry point for multi-year production runs.

Usage
-----
# Dry run (print operations, no execution):
  uv run python run_pipeline.py --start 2015-03 --end 2015-03 --dry-run

# Real run:
  uv run python run_pipeline.py --start 2015-01 --end 2021-12

Orchestration order (per month)
--------------------------------
1. retrieve_omni
2. retrieve_goes  (all configured satellites)
3. retrieve_swarm (all three satellites)
4. retrieve_supermag (all configured stations)
5. compute_leo_index
6. validate_sources
7. build_feature_matrix
8. validate_feature_matrix
"""

import argparse
import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import config
from logger import get_logger

log = get_logger(__name__)

FAILED_LOG_PATH = os.path.join(config.LOGS_DIR, "failed_months.log")


# ---------------------------------------------------------------------------
# Month iteration
# ---------------------------------------------------------------------------

def _month_range(start_ym: str, end_ym: str):
    """Yield (year, month) pairs from start_ym to end_ym inclusive.

    Parameters
    ----------
    start_ym, end_ym:
        Strings in 'YYYY-MM' format.

    Yields
    ------
    tuple[int, int]
        (year, month)
    """
    y_s, m_s = [int(x) for x in start_ym.split("-")]
    y_e, m_e = [int(x) for x in end_ym.split("-")]
    current = datetime.date(y_s, m_s, 1)
    end = datetime.date(y_e, m_e, 1)

    while current <= end:
        yield current.year, current.month
        if current.month == 12:
            current = datetime.date(current.year + 1, 1, 1)
        else:
            current = datetime.date(current.year, current.month + 1, 1)


# ---------------------------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------------------------

def _run_step(step_name: str, year: int, month: int, fn, dry_run: bool) -> bool:
    """Execute one pipeline step, catching all errors.

    Returns True on success, False on failure.
    """
    tag = f"{year:04d}-{month:02d} / {step_name}"

    if dry_run:
        print(f"  [DRY-RUN] {tag}")
        return True

    log.info("START  %s", tag)
    try:
        fn()
        log.info("DONE   %s", tag)
        return True
    except Exception:
        log.error("FAILED %s\n%s", tag, traceback.format_exc())
        _record_failure(year, month, step_name)
        return False


def _record_failure(year: int, month: int, step: str) -> None:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{year:04d}-{month:02d}\t{step}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-year dB/dt pipeline orchestrator"
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM",
        help="First month to process (inclusive).",
    )
    parser.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM",
        help="Last month to process (inclusive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operation sequence without executing anything.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY-RUN MODE -- no data will be downloaded or written ===")

    # Lazy imports (keep orchestrator fast to import for --dry-run)
    from scripts.retrieve_omni import retrieve_omni
    from scripts.retrieve_goes import retrieve_goes
    from scripts.retrieve_swarm import retrieve_swarm_month
    from scripts.retrieve_supermag import retrieve_supermag_month
    from scripts.compute_leo_index import build_leo_index_month
    from scripts.validate_sources import validate_datasets
    from scripts.build_feature_matrix import build_feature_matrix
    from scripts.validate_feature_matrix import validate_feature_matrix

    months = list(_month_range(args.start, args.end))
    total = len(months)
    succeeded = 0
    failed = 0

    log.info("Pipeline starting: %s -> %s (%d months)", args.start, args.end, total)
    if args.dry_run:
        print(f"Pipeline: {args.start} -> {args.end} ({total} months)")

    for year, month in months:
        month_start = datetime.datetime.now(datetime.UTC)
        tag = f"{year:04d}-{month:02d}"
        log.info("=== Month %s ===", tag)

        if args.dry_run:
            print(f"\n--- {tag} ---")

        month_ok = True

        steps = [
            ("retrieve_omni", lambda y=year, m=month: retrieve_omni(y, m)),
            ("retrieve_goes", lambda y=year, m=month: retrieve_goes(y, m)),
            ("retrieve_swarm_A", lambda y=year, m=month: retrieve_swarm_month(y, m, "A")),
            ("retrieve_swarm_B", lambda y=year, m=month: retrieve_swarm_month(y, m, "B")),
            ("retrieve_swarm_C", lambda y=year, m=month: retrieve_swarm_month(y, m, "C")),
            ("retrieve_supermag", lambda y=year, m=month: retrieve_supermag_month(y, m)),
            ("compute_leo_index", lambda y=year, m=month: build_leo_index_month(y, m)),
            ("validate_sources", lambda y=year, m=month: validate_datasets(y, m, fail_on_error=True)),
            ("build_feature_matrix", lambda y=year, m=month: build_feature_matrix(y, m)),
            ("validate_feature_matrix", lambda y=year, m=month: validate_feature_matrix(y, m, fail_on_error=True)),
        ]

        for step_name, fn in steps:
            ok = _run_step(step_name, year, month, fn, dry_run=args.dry_run)
            if not ok:
                month_ok = False
                break

        if month_ok:
            succeeded += 1
        else:
            failed += 1

        elapsed = (datetime.datetime.now(datetime.UTC) - month_start).total_seconds()
        log.info("Month %s finished in %.1f s (ok=%s)", tag, elapsed, month_ok)

    print("\n" + "=" * 60)
    print(f"Pipeline complete: {total} months attempted")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    if failed > 0:
        print(f"  Failed entries logged to: {FAILED_LOG_PATH}")
        with open(FAILED_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                print(f"    {line.rstrip()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
