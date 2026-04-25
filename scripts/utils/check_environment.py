"""
TODO: refactor to combine all the scripts in here
Environment validation: checks all API credentials, dependencies, model files
"""
import datetime
from cdasws import CdasWs

def test_cdaweb():
    cdas = CdasWs()
    
    # 1. Explicitly setting UTC timezone (CDAWeb often rejects naive datetimes)
    start_dt = datetime.datetime(2015, 3, 1, tzinfo=datetime.timezone.utc)
    end_dt = datetime.datetime(2015, 3, 2, tzinfo=datetime.timezone.utc)
    
    print("Sending Canary Request to CDAWeb...")
    
    # 2. Requesting just 1 day and 1 variable to bypass any payload limits
    status, data = cdas.get_data(
        "OMNI_HRO2_1MIN",
        ["BX_GSE"], 
        start_dt,
        end_dt
    )
    
    print(f"Status Code: {status['http']['status_code']}")
    if data:
        print(f"✓ Success! Retrieved keys: {list(data.keys())}")
        print(f"✓ Number of records: {len(data['Epoch'])}")
    else:
        print("✗ Failed to retrieve data.")

if __name__ == "__main__":
    test_cdaweb()
#!/usr/bin/env python3
from cdasws import CdasWs

def find_goes_datasets():
    cdas = CdasWs()
    print("Searching CDAWeb strictly for GOES-15 Magnetometer datasets...")
    
    datasets = cdas.get_datasets(observatoryGroup="GOES")
    
    goes15_mag_candidates = []
    for ds in datasets:
        ds_id = ds['Id'].upper()
        # Look for GOES-15 AND "MAG" or "MAGNETOMETER", but EXCLUDE "MAGED" (the electron detector)
        if "GOES" in ds_id and "15" in ds_id and ("MAG" in ds_id or "FG" in ds_id):
            if "MAGED" not in ds_id and "PARTICLE" not in ds_id:
                goes15_mag_candidates.append(ds)
            
    if not goes15_mag_candidates:
        print("No GOES-15 MAG datasets found via CDAWeb. We will have to use NCEI directly.")
        return

    print(f"Found {len(goes15_mag_candidates)} candidates:\n")
    
    for ds in goes15_mag_candidates:
        print(f"Dataset ID: {ds['Id']}")
        print(f"Label: {ds['Label']}")
        
        try:
            vars_info = cdas.get_variables(ds['Id'])
            var_names = [v['Name'] for v in vars_info]
            
            coords = "Unknown"
            if any("GSM" in v.upper() for v in var_names):
                coords = "Looks like it contains GSM! (Jackpot)"
            elif any(v in ["B_P", "B_E", "B_N", "Hp", "He", "Hn"] for v in var_names):
                coords = "Spacecraft Coordinates (PEN or Hp/He/Hn) - Requires Rotation Matrix."
                
            print(f"Coordinates: {coords}")
            print(f"Variables: {var_names[:8]}")
            print("-" * 60)
        except Exception as e:
            pass

if __name__ == "__main__":
    find_goes_datasets()
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
