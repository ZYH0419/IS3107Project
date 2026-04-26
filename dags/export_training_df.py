from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

from ml_common import load_training_frame_from_r2

print("Starting export...")

df = load_training_frame_from_r2(
    lookahead_minutes=15,
    lookback_hours=12,   # change to 3, 6, 24, 72 etc
)

print("Shape:", df.shape)
print(df.head())
print(df.columns)

df = df.sort_values(["link_id", "collected_at"])

# simple lag
df["speed_band_lag_1"] = df.groupby("link_id")["current_speed_band"].shift(1)
df["speed_band_lag_2"] = df.groupby("link_id")["current_speed_band"].shift(2)

# rolling average
df["speed_band_roll_3"] = (
    df.groupby("link_id")["current_speed_band"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

df.to_parquet("combined_training_df.parquet", index=False)

print("Saved combined_training_df.parquet")