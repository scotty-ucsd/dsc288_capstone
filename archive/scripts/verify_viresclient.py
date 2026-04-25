#!/usr/bin/env python3
"""Verify ESA Swarm (VirES) authentication and data access.

Run this script to confirm your VirES token is configured correctly:
    uv run python scripts/verify_viresclient.py

If not configured, visit https://vires.services/ to create an account
and token, then run:
    viresclient set_token https://vires.services/ows
"""

from datetime import datetime, timedelta

from viresclient import SwarmRequest


def main() -> None:
    """Test VirES connection by requesting a small data sample."""
    print("=" * 60)
    print("VirES for Swarm Authentication Test")
    print("=" * 60)

    try:
        request = SwarmRequest()
        print("✓ SwarmRequest initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SwarmRequest: {e}")
        print("\nTo fix this, run:")
        print("    viresclient set_token https://vires.services/ows")
        return

    # Request 10 minutes of Swarm A data from a known good period
    start = datetime(2020, 1, 1, 0, 0)
    end = start + timedelta(minutes=10)

    print(f"\nRequesting Swarm A MAG data: {start} to {end}")

    try:
        request.set_collection("SW_OPER_MAGA_LR_1B")
        request.set_products(
            measurements=["B_NEC", "Latitude", "Longitude"],
            sampling_step="PT10S",
        )
        data = request.get_between(start, end)
        df = data.as_dataframe()

        print(f"✓ Retrieved {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        print("\n✓ VirES authentication successful!")

    except Exception as e:
        print(f"✗ Data request failed: {e}")
        print("\nPossible issues:")
        print("  1. Token not configured or expired")
        print("  2. Network connectivity")
        print("  3. VirES service temporarily unavailable")


if __name__ == "__main__":
    main()
