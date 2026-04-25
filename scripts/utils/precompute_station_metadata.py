#!/usr/bin/env python3
"""Precompute SuperMAG station coordinate metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from swmi.api.supermag import precompute_station_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute SuperMAG station metadata.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "data" / "external" / "station_metadata" / "supermag_station_coords.parquet"),
    )
    parser.add_argument("--station", action="append", dest="stations", help="Restrict to a station; repeatable.")
    parser.add_argument("--force-inventory-refresh", action="store_true")
    parser.add_argument(
        "--no-cached-raw",
        action="store_true",
        help="Do not use existing raw SuperMAG files as metadata sources.",
    )
    parser.add_argument(
        "--no-fetch-missing",
        action="store_true",
        help="Do not fetch station metadata when cached raw files are unavailable.",
    )
    parser.add_argument("--request-delay-sec", type=float, default=0.0)
    args = parser.parse_args(argv)

    precompute_station_metadata(
        args.start_year,
        args.start_month,
        args.end_year,
        args.end_month,
        stations=args.stations,
        output_path=args.output,
        force_inventory_refresh=args.force_inventory_refresh,
        use_cached_raw=not args.no_cached_raw,
        fetch_missing_metadata=not args.no_fetch_missing,
        request_delay_sec=args.request_delay_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
