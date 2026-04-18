from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import pandas as pd
import requests


API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

# If you are running inside Docker, this is a good default.
# You can override it in .env if needed.
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def fetch_all_speed_bands() -> pd.DataFrame:
    headers = {
        "AccountKey": LTA_ACCOUNT_KEY,
        "accept": "application/json",
    }

    all_data = []
    skip = 0
    page_size = 500

    while True:
        url = f"{API_URL}?$skip={skip}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        payload = response.json()
        records = payload.get("value", [])

        if not records:
            break

        all_data.extend(records)

        if len(records) < page_size:
            break

        skip += page_size

    return pd.DataFrame(all_data)


def fetch_and_save_speed_bands_locally():
    collected_at = datetime.now(timezone.utc)
    ts_label = collected_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("Start retrieving for %s", ts_label)

    df = fetch_all_speed_bands()

    if df.empty:
        logger.info("Retrieved 0 rows at %s", ts_label)
        return

    snapshots_df = df[
        [
            "LinkID",
            "SpeedBand",
            "MinimumSpeed",
            "MaximumSpeed",
        ]
    ].copy()

    snapshots_df.columns = [
        "link_id",
        "speed_band",
        "minimum_speed",
        "maximum_speed",
    ]

    # Clean numeric fields
    snapshots_df["link_id"] = pd.to_numeric(snapshots_df["link_id"], errors="coerce")
    snapshots_df["speed_band"] = pd.to_numeric(snapshots_df["speed_band"], errors="coerce")
    snapshots_df["minimum_speed"] = pd.to_numeric(snapshots_df["minimum_speed"], errors="coerce")
    snapshots_df["maximum_speed"] = pd.to_numeric(snapshots_df["maximum_speed"], errors="coerce")

    # Drop rows without valid link_id
    snapshots_df = snapshots_df[snapshots_df["link_id"].notna()].copy()

    # Convert link_id to integer where possible
    snapshots_df["link_id"] = snapshots_df["link_id"].astype("int64")

    # Save one snapshot per file
    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"traffic_speed_snapshot_{file_label}.parquet"

    # Save as parquet
    snapshots_df.to_parquet(output_path, index=False)

    logger.info("Retrieved %s rows at %s", len(snapshots_df), ts_label)


with DAG(
    dag_id="5_collecting_training_data",
    default_args=default_args,
    description="Retrieve full LTA Traffic Speed Bands every 10 minutes and save locally as one parquet snapshot per file",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 3, 17),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "local", "raw"],
) as dag:

    run_collection = PythonOperator(
        task_id="fetch_and_save_speed_bands_locally",
        python_callable=fetch_and_save_speed_bands_locally,
    )