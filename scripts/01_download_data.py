#!/usr/bin/env python3
"""Download monthly source data for the SWMI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.setup_directories import create_directories
from swmi.api.goes import retrieve_goes_mag, retrieve_goes_xray
from swmi.api.omni import retrieve_omni
from swmi.api.supermag import retrieve_supermag_month
from swmi.api.swarm import retrieve_swarm_month
from swmi.preprocessing.validation import validate_sources
from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants
from swmi.utils.logger import get_logger

log = get_logger(__name__)


class DownloadStepResult(NamedTuple):
    """Status for one source retrieval step."""

    name: str
    ok: bool
    error: str | None = None


def _load_data_config(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    validate_scientific_invariants(cfg, str(config_path))
    if str(cfg.get("master_cadence")) != config.MASTER_CADENCE:
        raise ValueError(
            f"master_cadence={cfg.get('master_cadence')!r} does not match "
            f"MASTER_CADENCE={config.MASTER_CADENCE!r}"
        )
    return cfg


def _parse_two_year_ints(tokens: list[str]) -> tuple[int, int] | None:
    """If ``tokens == ['2015', '2024']``, return (2015, 2024)."""
    if len(tokens) < 2:
        return None
    if not str(tokens[0]).isdigit() or not str(tokens[1]).isdigit():
        return None
    return int(tokens[0]), int(tokens[1])


def _year_range_after_dry_run(argv: list[str]) -> tuple[int, int] | None:
    """If argv looks like ``... --dry-run 2015 2024 ...``, return the two years."""
    if "--dry-run" not in argv:
        return None
    i = argv.index("--dry-run")
    tail = [a for a in argv[i + 1 :] if not str(a).startswith("-")]
    return _parse_two_year_ints(tail)


def _run_step(name: str, func, *args, continue_on_error: bool) -> DownloadStepResult:
    try:
        func(*args)
    except Exception as exc:
        log.exception("%s retrieval failed.", name)
        if not continue_on_error:
            raise
        return DownloadStepResult(name=name, ok=False, error=str(exc))
    return DownloadStepResult(name=name, ok=True)


def run_download_pipeline(
    year: int,
    month: int,
    *,
    config_path: str | Path = _PROJECT_ROOT / "configs" / "data_retrieval.yaml",
    skip_validation: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = True,
    force_inventory_refresh: bool = False,
    supermag_request_delay_sec: float = 0.0,
) -> list[DownloadStepResult]:
    """Run P0-E2 monthly source retrieval orchestration."""
    cfg = _load_data_config(config_path)
    swarm_sats = [str(sat).upper() for sat in cfg.get("swarm", {}).get("satellites", config.SWARM_SATELLITES)]

    create_directories(_PROJECT_ROOT)
    steps: list[tuple[str, object, tuple]] = [
        ("omni", retrieve_omni, (year, month)),
        ("goes_mag", retrieve_goes_mag, (year, month)),
        ("goes_xray", retrieve_goes_xray, (year, month)),
    ]
    steps.extend(
        (f"swarm_{sat}", retrieve_swarm_month, (year, month, sat))
        for sat in swarm_sats
    )
    steps.append(
        (
            "supermag",
            retrieve_supermag_month,
            (year, month, None, True, force_inventory_refresh, supermag_request_delay_sec),
        )
    )

    if dry_run:
        print("Download dry run:")
        print(f"  year={year}")
        print(f"  month={month}")
        print(f"  sources={[name for name, _, _ in steps]}")
        print(f"  validate={not skip_validation}")
        return []

    results = [
        _run_step(name, func, *args, continue_on_error=continue_on_error)
        for name, func, args in steps
    ]

    if not skip_validation:
        valid = validate_sources(year, month, fail_on_error=False)
        results.append(DownloadStepResult(name="source_validation", ok=bool(valid)))

    failures = [result for result in results if not result.ok]
    if failures:
        log.warning("Download pipeline completed with failed steps: %s", [f.name for f in failures])
    else:
        log.info("Download pipeline completed successfully for %04d-%02d.", year, month)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and validate monthly source data.")
    parser.add_argument(
        "--year",
        type=int,
        required=False,
        help="Calendar year (required unless --dry-run is used with start/end year positionals).",
    )
    parser.add_argument(
        "--month",
        type=int,
        required=False,
        help="Calendar month 1-12 (required with --year when not using dry-run range).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing source products without running retrieval.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip monthly source validation after retrieval.",
    )
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "configs" / "data_retrieval.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise immediately on the first source retrieval failure.",
    )
    parser.add_argument(
        "--force-inventory-refresh",
        action="store_true",
        help="Bypass cached SuperMAG station inventory for this month.",
    )
    parser.add_argument(
        "--supermag-request-delay-sec",
        type=float,
        default=0.0,
        help="Optional delay between SuperMAG station requests.",
    )
    argv_list = argv if argv is not None else sys.argv[1:]
    args, unknown = parser.parse_known_args(argv_list)

    create_directories(_PROJECT_ROOT)

    if args.validate_only:
        if args.year is None or args.month is None:
            log.error("--validate-only requires --year and --month.")
            return 1
        return 0 if validate_sources(args.year, args.month, fail_on_error=False) else 1

    if not args.dry_run and not args.validate_only and unknown:
        log.error("Unrecognized arguments: %s", " ".join(unknown))
        return 2

    if args.dry_run:
        yr = _parse_two_year_ints(unknown) or _year_range_after_dry_run(
            [a for a in (argv if argv is not None else sys.argv)]
        )
        if yr is not None:
            y0, y1 = yr
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for m in range(1, 13):
                    run_download_pipeline(
                        y,
                        m,
                        config_path=args.config,
                        skip_validation=True,
                        dry_run=True,
                    )
            return 0
        if args.year is not None and args.month is not None:
            run_download_pipeline(
                args.year,
                args.month,
                config_path=args.config,
                skip_validation=args.skip_validation,
                dry_run=True,
            )
            return 0
        log.error("Dry run requires --year and --month, or: --dry-run <start_year> <end_year>")
        return 1

    if args.year is None or args.month is None:
        log.error("Missing --year / --month (or use --dry-run with a year range).")
        return 1

    results = run_download_pipeline(
        args.year,
        args.month,
        config_path=args.config,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        continue_on_error=not args.fail_fast,
        force_inventory_refresh=args.force_inventory_refresh,
        supermag_request_delay_sec=args.supermag_request_delay_sec,
    )
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
