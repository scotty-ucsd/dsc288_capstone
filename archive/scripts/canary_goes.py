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
