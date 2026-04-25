#!/usr/bin/env python3
"""
retrieve_goes_modern.py
DEPRECATED — kept for reference only.
Consolidated into retrieve_goes.py (Phase 2.1 refactor).
Do not add new features here. Use retrieve_goes.retrieve_goes_month() instead.

Original purpose:
Downloads GOES-R Series (16, 17, 18, 19) Level-2 1-Minute Averaged Magnetometer data
directly from the NOAA NGDC HTTP servers.
Includes self-healing retry logic to delete and re-download corrupted NetCDF files.
Surgically extracts GSM coordinates, manually decodes time, and saves to Parquet.
"""
import warnings
warnings.warn(
    "retrieve_goes_modern.py is deprecated. Use retrieve_goes.retrieve_goes_month() instead.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import re
import time
import requests
import numpy as np
import xarray as xr
import pandas as pd

# Base URL for GOES-R modern magnetometer data
NGDC_BASE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"

def retrieve_goes_modern_month(year: int, month: int, satellite: int, output_dir: str, raw_dir: str) -> None:
    # Validate modern GOES satellite numbers
    if satellite not in [16, 17, 18, 19]:
        raise ValueError(f"Satellite must be 16, 17, 18, or 19. Got: {satellite}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"goes{satellite}_1min_{year:04d}{month:02d}.parquet")

    if os.path.exists(output_path):
        print(f"✓ Already exists, skipping: {output_path}")
        return

    # Construct the target directory URL based on satellite, year, and month
    month_url = f"{NGDC_BASE_URL}/goes{satellite}/l2/data/magn-l2-avg1m/{year:04d}/{month:02d}/"
    print(f"Scanning NGDC directory: {month_url}")
    
    # Use a standard user-agent; NOAA servers sometimes block raw python requests
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(month_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Failed to access NGDC directory. Ensure the satellite/date combination exists. Error: {e}")
        return

    # Regex to find all NetCDF files for this specific month and satellite
    # Matches patterns like: dn_magn-l2-avg1m_g16_d20170412_v2_0_0.nc
    regex_pattern = rf'href="(dn_magn-l2-avg1m_g{satellite}_d\d{{8}}_v[\d_]+\.nc)"'
    nc_files = re.findall(regex_pattern, response.text)
    nc_files = sorted(list(set(nc_files)))

    if not nc_files:
        print(f"  [!] No NetCDF files found for GOES-{satellite} in {year}-{month:02d}.")
        return

    print(f"Found {len(nc_files)} daily files. Downloading and processing...")

    monthly_data = []

    for filename in nc_files:
        file_url = f"{month_url}{filename}"
        local_nc_path = os.path.join(raw_dir, filename)

        max_retries = 3
        day_processed = False

        for attempt in range(max_retries):
            # 1. Download Phase
            if not os.path.exists(local_nc_path):
                print(f"  Downloading {filename} (Attempt {attempt + 1}/{max_retries})...")
                try:
                    with requests.get(file_url, stream=True, timeout=30, headers=headers) as r:
                        r.raise_for_status()
                        with open(local_nc_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                except Exception as e:
                    print(f"  [!] Download failed: {e}")
                    # Clean up partial download before retrying
                    if os.path.exists(local_nc_path):
                        os.remove(local_nc_path)
                    time.sleep(2)
                    continue

            # 2. Processing Phase
            try:
                # Open dataset safely, explicitly disabling automatic time decoding
                ds = xr.open_dataset(local_nc_path, engine="netcdf4", decode_times=False)
                
                # Extract raw arrays based on your ncdump
                times = ds['time'].values
                b_gsm = ds['b_gsm'].values  # Shape is (time, 3) representing [Bx, By, Bz]
                
                # Manual Time Decoding: "seconds since 2000-01-01 12:00:00"
                j2k_epoch = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")
                timestamps = j2k_epoch + pd.to_timedelta(times, unit='s')
                
                df_target = pd.DataFrame({
                    "timestamp": timestamps,
                    "B_X_GSM": b_gsm[:, 0],
                    "B_Y_GSM": b_gsm[:, 1],
                    "B_Z_GSM": b_gsm[:, 2]
                })
                
                # Mask the -9999.0 fill value identified in the ncdump
                for col in ["B_X_GSM", "B_Y_GSM", "B_Z_GSM"]:
                    df_target[col] = df_target[col].where(df_target[col] > -9000.0, float("nan"))
                
                # Drop rows with invalid time data just to be safe
                df_target = df_target.dropna(subset=["timestamp"])
                
                # Resample to 1-minute means to ensure perfect alignment to the '00' second mark
                df_target = df_target.set_index("timestamp")
                df_1min = df_target.resample("1min").mean().reset_index()
                
                monthly_data.append(df_1min)
                ds.close()
                day_processed = True
                break  # Success! Escape the retry loop.
                
            except Exception as e:
                print(f"  [!] Error processing {filename}: {e}")
                print(f"  [*] Removing corrupted file and retrying...")
                # The crucial cleanup step
                if os.path.exists(local_nc_path):
                    try:
                        os.remove(local_nc_path)
                    except OSError as rm_e:
                        print(f"  [!] Could not delete file: {rm_e}")
                
                time.sleep(2) # Brief backoff before next attempt
                
        if not day_processed:
            print(f"  [!] Skipping {filename} entirely after {max_retries} failed attempts.")

    if not monthly_data:
        print("  [!] No data could be processed for this month.")
        return

    # Combine all daily chunks
    df_month = pd.concat(monthly_data, ignore_index=True)
    df_month = df_month.sort_values("timestamp").reset_index(drop=True)
    
    df_month.to_parquet(output_path, index=False)
    print(f"✓ Saved merged 1-minute GOES-{satellite} data to {output_path} ({len(df_month)} rows)")

if __name__ == "__main__":
    # Test pull using the GOES-16 directory we verified
    retrieve_goes_modern_month(
        year=2017,
        month=4,
        satellite=16,
        output_dir="data/processed/aligned_1min/2017/04",
        raw_dir="data/raw/goes/modern/2017/04"
    )
