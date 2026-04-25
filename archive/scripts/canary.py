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
