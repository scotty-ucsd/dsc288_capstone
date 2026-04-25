#!/usr/bin/env python3
"""
retrieve_goes_legacy.py
DEPRECATED — kept for reference only.
Consolidated into retrieve_goes.py (Phase 2.1 refactor).
Do not add new features here. Use retrieve_goes.retrieve_goes_month() instead.

Original purpose:
Downloads GOES-15 Level-2 High-Res Magnetometer data from NOAA NCEI.
Includes self-healing retry logic to delete and re-download corrupted NetCDF files.
Surgically extracts GSM coordinates, manually decodes time, and saves to Parquet.
"""
import warnings
warnings.warn(
    "retrieve_goes_legacy.py is deprecated. Use retrieve_goes.retrieve_goes_month() instead.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import re
import time
import datetime
import requests
import numpy as np
import xarray as xr
import pandas as pd

NCEI_BASE_URL = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/mag/goes15/magn-l2-hires"

def retrieve_goes_legacy_month(year: int, month: int, output_dir: str, raw_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"goes15_1min_{year:04d}{month:02d}.parquet")

    if os.path.exists(output_path):
        print(f"✓ Already exists, skipping: {output_path}")
        return

    month_url = f"{NCEI_BASE_URL}/{year:04d}/{month:02d}/"
    print(f"Scanning NCEI directory: {month_url}")
    
    try:
        response = requests.get(month_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Failed to access NCEI directory: {e}")
        return

    nc_files = re.findall(r'href="(dn_magn-l2-hires_g15_d\d{8}_v[\d_]+\.nc)"', response.text)
    nc_files = sorted(list(set(nc_files)))

    if not nc_files:
        print(f"  [!] No NetCDF files found for {year}-{month:02d}.")
        return

    print(f"Found {len(nc_files)} daily files. Downloading and resampling to 1-minute...")

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
                    with requests.get(file_url, stream=True, timeout=30) as r:
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
                ds = xr.open_dataset(local_nc_path, engine="netcdf4", decode_times=False)
                
                times = ds['time'].values
                b_gsm = ds['b_gsm'].values
                
                j2k_epoch = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")
                timestamps = j2k_epoch + pd.to_timedelta(times, unit='s')
                
                df_target = pd.DataFrame({
                    "timestamp": timestamps,
                    "B_X_GSM": b_gsm[:, 0],
                    "B_Y_GSM": b_gsm[:, 1],
                    "B_Z_GSM": b_gsm[:, 2]
                })
                
                for col in ["B_X_GSM", "B_Y_GSM", "B_Z_GSM"]:
                    df_target[col] = df_target[col].where(df_target[col] > -9000.0, float("nan"))
                
                df_target = df_target.dropna(subset=["timestamp"])
                
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

    df_month = pd.concat(monthly_data, ignore_index=True)
    df_month = df_month.sort_values("timestamp").reset_index(drop=True)
    
    df_month.to_parquet(output_path, index=False)
    print(f"✓ Saved merged 1-minute GOES-15 data to {output_path} ({len(df_month)} rows)")

if __name__ == "__main__":
    retrieve_goes_legacy_month(
        year=2015,
        month=3,
        output_dir="data/processed/aligned_1min/2015/03",
        raw_dir="data/raw/goes/legacy/2015/03"
    )
