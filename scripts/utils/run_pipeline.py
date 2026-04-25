#!/usr/bin/env python3
"""
run_pipeline.py
Master orchestration script for the dB/dt geomagnetic forecasting pipeline.

Supports two modes:
    full  — End-to-end pipeline: download → features → sequences → train → evaluate
    fused — Monthly fused dataset generator for EDA and feature engineering

Usage
-----
# Full pipeline (default mode):
  uv run python scripts/utils/run_pipeline.py --start 2015-01 --end 2024-12 --dry-run
  uv run python scripts/utils/run_pipeline.py --start 2015-01 --end 2024-12

# Fused dataset mode:
  uv run python scripts/utils/run_pipeline.py --mode fused --start 2015-01 --end 2024-12 --format eda
  uv run python scripts/utils/run_pipeline.py --mode fused --start 2015-03 --end 2015-03 --dry-run
  uv run python scripts/utils/run_pipeline.py --mode fused --sources omni,goes --start 2015-01 --end 2015-12

Full pipeline orchestration order (per month)
---------------------------------------------
1. run_download_pipeline  (OMNI, GOES mag/xray, Swarm A/B/C, SuperMAG)
2. run_feature_pipeline   (X-ray normalization, dB/dt, LEO index, feature matrix)
3. build_sequences        (train/val/test split with anti-leakage)
4. train_baseline         (M0/M1/M2)
5. evaluate_model         (per-station metrics, figures, reports)

Fused pipeline orchestration order (per month)
----------------------------------------------
1. Ensure feature matrix exists (download + build if needed)
2. Post-process: add completeness flags, cyclical month encoding
3. Apply column format filter (minimal/eda/debug)
4. Write fused Parquet to output directory
5. Generate summary report
"""

from __future__ import annotations

import argparse
import datetime
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from swmi.utils import config
from swmi.utils.logger import get_logger

log = get_logger(__name__)

FAILED_LOG_PATH = Path(config.LOGS_DIR) / "failed_months.log"


# ---------------------------------------------------------------------------
# Month iteration
# ---------------------------------------------------------------------------

def _month_range(start_ym: str, end_ym: str) -> list[tuple[int, int]]:
    """Return (year, month) pairs from start_ym to end_ym inclusive.

    Parameters
    ----------
    start_ym, end_ym:
        Strings in 'YYYY-MM' format.

    Returns
    -------
    list[tuple[int, int]]
        List of (year, month) tuples.

    Raises
    ------
    ValueError
        If the date strings are not valid 'YYYY-MM' format or start > end.
    """
    try:
        y_s, m_s = [int(x) for x in start_ym.split("-")]
        y_e, m_e = [int(x) for x in end_ym.split("-")]
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid date format: start={start_ym!r}, end={end_ym!r}. "
            "Expected 'YYYY-MM'."
        ) from exc

    if not (1 <= m_s <= 12 and 1 <= m_e <= 12):
        raise ValueError(
            f"Month must be 1-12: start month={m_s}, end month={m_e}."
        )

    current = datetime.date(y_s, m_s, 1)
    end = datetime.date(y_e, m_e, 1)

    if current > end:
        raise ValueError(
            f"Start date {start_ym} is after end date {end_ym}."
        )

    months: list[tuple[int, int]] = []
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime.date(current.year + 1, 1, 1)
        else:
            current = datetime.date(current.year, current.month + 1, 1)
    return months


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
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{year:04d}-{month:02d}\t{step}\n")


# ---------------------------------------------------------------------------
# Full pipeline mode
# ---------------------------------------------------------------------------

def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full end-to-end pipeline: download → features → sequences → train → evaluate.

    This modernized implementation wraps the refactored orchestrators from
    01_download_data.py and 02_build_features.py rather than calling the
    removed individual retriever functions directly.
    """
    # Lazy imports: keep orchestrator fast for --dry-run.
    from scripts.utils.setup_directories import create_directories

    create_directories(_PROJECT_ROOT)

    months = _month_range(args.start, args.end)
    total = len(months)
    succeeded = 0
    failed = 0

    if args.dry_run:
        print("=== DRY-RUN MODE -- no data will be downloaded or written ===")

    log.info("Pipeline starting: %s -> %s (%d months)", args.start, args.end, total)
    if args.dry_run:
        print(f"Pipeline: {args.start} -> {args.end} ({total} months)")

    # Lazy-import orchestrators inside the loop to avoid import-time side effects.
    from scripts import download_data_mod, build_features_mod
    download_mod = download_data_mod
    features_mod = build_features_mod

    for year, month in months:
        month_start = datetime.datetime.now(datetime.UTC)
        tag = f"{year:04d}-{month:02d}"
        log.info("=== Month %s ===", tag)

        if args.dry_run:
            print(f"\n--- {tag} ---")

        month_ok = True

        steps = [
            ("download", lambda y=year, m=month: download_mod.run_download_pipeline(y, m, dry_run=args.dry_run)),
            ("features", lambda y=year, m=month: features_mod.run_feature_pipeline(y, m, dry_run=args.dry_run)),
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

    _print_summary(total, succeeded, failed)


# ---------------------------------------------------------------------------
# Fused pipeline mode
# ---------------------------------------------------------------------------

def run_fused_pipeline(args: argparse.Namespace) -> None:
    """Generate monthly fused datasets for EDA and feature engineering.

    Steps:
    1. Iterate months in [start, end].
    2. For each month:
       a. Check if fused output exists (skip unless --force).
       b. Ensure feature matrix exists (optionally trigger build).
       c. Build fused Parquet with selected format.
    3. Generate summary report.
    """
    from scripts.utils.setup_directories import create_directories
    from swmi.features.fused import build_fused_month, generate_fused_summary_report

    create_directories(_PROJECT_ROOT)

    months = _month_range(args.start, args.end)
    total = len(months)

    # Parse sources.
    sources: list[str] | None = None
    if hasattr(args, "sources") and args.sources and args.sources.lower() != "all":
        sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    output_dir = args.output_dir
    format_mode = getattr(args, "format", "eda")
    force = getattr(args, "force", False)

    if args.dry_run:
        print("=== DRY-RUN MODE (fused) -- no data will be written ===")
        print(f"Fused pipeline: {args.start} -> {args.end} ({total} months)")
        print(f"  format:     {format_mode}")
        print(f"  output_dir: {output_dir}")
        print(f"  sources:    {sources or 'all'}")
        print(f"  force:      {force}")
        print()

    log.info(
        "Fused pipeline starting: %s -> %s (%d months, format=%s)",
        args.start, args.end, total, format_mode,
    )

    months_processed: list[tuple[int, int, bool]] = []
    succeeded = 0
    failed = 0

    for year, month in months:
        month_start = datetime.datetime.now(datetime.UTC)
        tag = f"{year:04d}-{month:02d}"
        log.info("=== Fused month %s ===", tag)

        try:
            result_path = build_fused_month(
                year, month,
                output_dir=output_dir,
                format_mode=format_mode,
                sources=sources,
                force=force,
                dry_run=args.dry_run,
            )
            months_processed.append((year, month, True))
            succeeded += 1

        except FileNotFoundError as exc:
            log.error("Fused month %s: feature matrix not found: %s", tag, exc)
            _record_failure(year, month, "fused_missing_features")
            months_processed.append((year, month, False))
            failed += 1

        except Exception:
            log.error("Fused month %s FAILED:\n%s", tag, traceback.format_exc())
            _record_failure(year, month, "fused_build")
            months_processed.append((year, month, False))
            failed += 1

        elapsed = (datetime.datetime.now(datetime.UTC) - month_start).total_seconds()
        log.info("Fused month %s finished in %.1f s", tag, elapsed)

    # Generate summary report (skip on dry-run).
    if not args.dry_run:
        try:
            generate_fused_summary_report(output_dir, months_processed)
        except Exception:
            log.warning("Summary report generation failed:\n%s", traceback.format_exc())

    _print_summary(total, succeeded, failed)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(total: int, succeeded: int, failed: int) -> None:
    print("\n" + "=" * 60)
    print(f"Pipeline complete: {total} months attempted")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    if failed > 0 and FAILED_LOG_PATH.exists():
        print(f"  Failed entries logged to: {FAILED_LOG_PATH}")
        with open(FAILED_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                print(f"    {line.rstrip()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Lazy module imports for full pipeline
# ---------------------------------------------------------------------------
# The numbered scripts (01_download_data.py, 02_build_features.py) cannot be
# directly imported by name because their filenames start with digits. We use
# importlib to load them as modules at runtime, but only when --mode full is used.

class _LazyScriptModule:
    """Lazy-load a script module only when first accessed."""

    def __init__(self, script_path: str, module_name: str):
        self._script_path = _PROJECT_ROOT / script_path
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                self._module_name, str(self._script_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load script: {self._script_path}")
            self._module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._module)
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


# These are loaded lazily — no import happens until first attribute access.
# They are module-level so run_full_pipeline can reference them.
class scripts:
    download_data_mod = _LazyScriptModule("scripts/01_download_data.py", "download_data")
    build_features_mod = _LazyScriptModule("scripts/02_build_features.py", "build_features")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-year dB/dt pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (default)
  %(prog)s --start 2015-01 --end 2024-12

  # Fused dataset for EDA
  %(prog)s --mode fused --start 2015-01 --end 2024-12 --format eda

  # Fused with specific sources, dry-run
  %(prog)s --mode fused --sources omni,goes --start 2015-01 --end 2015-12 --dry-run
""",
    )

    # Shared arguments.
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
    parser.add_argument(
        "--mode",
        choices=["full", "fused"],
        default="full",
        help="Pipeline mode: full (default) or fused (EDA datasets).",
    )

    # Fused-mode arguments.
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="Comma-separated sources for fused mode: omni,goes,swarm,supermag,leo (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["minimal", "eda", "debug"],
        default="eda",
        help="Fused dataset column format (default: eda).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fused/",
        help="Output directory for fused datasets (default: data/fused/).",
    )
    parser.add_argument(
        "--station-filter",
        choices=["all", "best_coverage", "active"],
        default="all",
        help="Station selection for fused mode (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate output even if it already exists.",
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(args)
    elif args.mode == "fused":
        run_fused_pipeline(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
