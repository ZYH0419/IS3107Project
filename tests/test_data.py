import requests
import pandas as pd

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

headers = {
    "AccountKey": "pY0I88l9QiuNXqWHAPk+6A==",
    "accept": "application/json"
}

all_data = []
skip = 0

while True:
    url = f"{API_URL}?$skip={skip}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    records = data.get("value", [])

    if not records:
        break

    all_data.extend(records)
    print(f"Fetched {len(records)} rows (skip={skip})")

    if len(records) < 500:
        break

    skip += 500

df = pd.DataFrame(all_data)

print("Total rows:", len(df))
print("Unique LinkID:", df["LinkID"].nunique())

df.to_excel("traffic_speed_bands_full.xlsx", index=False)
print("Saved to traffic_speed_bands_full.xlsx")