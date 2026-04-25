from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import boto3
import pandas as pd
import requests

# ================= CONFIG =================

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/Taxi-Availability"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

R2_TAXI_PREFIX = os.environ.get("R2_TAXI_PREFIX", "taxi_availability/")
WRITE_LATEST_TAXI_ALIAS = os.environ.get("WRITE_LATEST_TAXI_ALIAS", "true").strip().lower() == "true"

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

# ================= HELPERS =================

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def upload_dataframe_to_r2(df: pd.DataFrame, key: str) -> None:
    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_name = Path(key).name
    local_path = output_dir / local_name

    df.to_parquet(local_path, index=False)

    client = get_r2_client()
    client.upload_file(str(local_path), R2_BUCKET, key)

    logger.info("Uploaded dataframe to R2: bucket=%s | key=%s | rows=%s", R2_BUCKET, key, len(df))

    try:
        local_path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Could not delete temporary taxi parquet: %s", local_path)


def build_partitioned_key(prefix: str, file_stem: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    hour_part = file_label[9:11]
    return f"{prefix}date={date_part}/hour={hour_part}/{file_stem}_{file_label}.parquet"


def fetch_taxi_availability_payload() -> list[dict]:
    headers = {
        "AccountKey": LTA_ACCOUNT_KEY,
        "accept": "application/json",
    }

    all_rows = []
    skip = 0
    page_size = 500

    while True:
        url = f"{API_URL}?$skip={skip}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        payload = response.json()
        rows = payload.get("value", [])

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        skip += page_size

    return all_rows


def build_taxi_availability_df(rows: list[dict], collected_at: datetime) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["longitude", "latitude", "collected_at"])

    rename_map = {}
    if "Longitude" in df.columns:
        rename_map["Longitude"] = "longitude"
    if "Latitude" in df.columns:
        rename_map["Latitude"] = "latitude"

    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["longitude", "latitude"] if c in df.columns]
    df = df[keep_cols].copy()
    df["collected_at"] = collected_at

    return df

# ================= TASK =================

def collect_and_upload_taxi_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching taxi availability payload")
    rows = fetch_taxi_availability_payload()

    logger.info("STEP 2: building taxi availability dataframe")
    taxi_df = build_taxi_availability_df(rows, collected_at=collected_at)

    logger.info("Taxi training data prepared | rows=%s", len(taxi_df))

    if not taxi_df.empty:
        taxi_key = build_partitioned_key(
            R2_TAXI_PREFIX,
            "taxi_availability_snapshot",
            file_label,
        )
        upload_dataframe_to_r2(taxi_df, taxi_key)

        if WRITE_LATEST_TAXI_ALIAS:
            latest_taxi_key = "taxi_availability/latest/taxi_availability_latest.parquet"
            upload_dataframe_to_r2(taxi_df, latest_taxi_key)
    else:
        logger.warning("No taxi availability rows found; skipping taxi upload")

# ================= DAG =================

with DAG(
    dag_id="5_3_collecting_training_data_taxi",
    default_args=default_args,
    description="Collect LTA Taxi Availability data and save to Cloudflare R2 as partitioned parquet training data.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "taxi", "training-data", "r2"],
) as dag:

    task_collect_and_upload_taxi_training_data = PythonOperator(
        task_id="collect_and_upload_taxi_training_data",
        python_callable=collect_and_upload_taxi_training_data,
    )