
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import boto3
import pandas as pd

from weather_common import (
    fetch_rainfall_payload,
    build_weather_stations_df,
    build_rainfall_readings_df,
)

# ================= CONFIG =================

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

# Prefixes for rainfall training data in R2
R2_RAINFALL_READINGS_PREFIX = os.environ.get("R2_RAINFALL_READINGS_PREFIX", "rainfall_readings/")
R2_WEATHER_STATIONS_PREFIX = os.environ.get("R2_WEATHER_STATIONS_PREFIX", "weather_stations/")
WRITE_LATEST_RAINFALL_ALIAS = os.environ.get("WRITE_LATEST_RAINFALL_ALIAS", "true").strip().lower() == "true"

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
    """
    Save a dataframe to a temporary local parquet file, upload to R2, then delete the local file.
    """
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
        logger.warning("Could not delete temporary rainfall parquet: %s", local_path)


def build_partitioned_key(prefix: str, file_stem: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    hour_part = file_label[9:11]
    return f"{prefix}date={date_part}/hour={hour_part}/{file_stem}_{file_label}.parquet"


# ================= TASK =================

def collect_and_upload_rainfall_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching rainfall payload")
    payload = fetch_rainfall_payload()

    logger.info("STEP 2: building weather stations dataframe")
    stations_df = build_weather_stations_df(payload)

    logger.info("STEP 3: building rainfall readings dataframe")
    readings_df = build_rainfall_readings_df(payload)

    # Add collection timestamp to both snapshots for traceability
    if not stations_df.empty:
        stations_df = stations_df.copy()
        stations_df["collected_at"] = collected_at

    if not readings_df.empty:
        readings_df = readings_df.copy()
        readings_df["collected_at"] = collected_at

    logger.info(
        "Rainfall training data prepared | stations=%s | readings=%s",
        len(stations_df),
        len(readings_df),
    )

    # Save weather stations snapshot to R2
    if not stations_df.empty:
        stations_key = build_partitioned_key(
            R2_WEATHER_STATIONS_PREFIX,
            "weather_stations_snapshot",
            file_label,
        )
        upload_dataframe_to_r2(stations_df, stations_key)

        if WRITE_LATEST_RAINFALL_ALIAS:
            latest_stations_key = "weather_stations/latest/weather_stations_latest.parquet"
            upload_dataframe_to_r2(stations_df, latest_stations_key)

    else:
        logger.warning("No weather stations found in rainfall payload; skipping stations upload")

    # Save rainfall readings snapshot to R2
    if not readings_df.empty:
        readings_key = build_partitioned_key(
            R2_RAINFALL_READINGS_PREFIX,
            "rainfall_readings_snapshot",
            file_label,
        )
        upload_dataframe_to_r2(readings_df, readings_key)

        if WRITE_LATEST_RAINFALL_ALIAS:
            latest_readings_key = "rainfall_readings/latest/rainfall_readings_latest.parquet"
            upload_dataframe_to_r2(readings_df, latest_readings_key)

    else:
        logger.warning("No rainfall readings found in rainfall payload; skipping readings upload")


# ================= DAG =================

with DAG(
    dag_id="5_2_collecting_training_data_rainfall",
    default_args=default_args,
    description="Collect rainfall payload data and save weather stations + rainfall readings to Cloudflare R2 as partitioned parquet training data.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["weather", "rainfall", "training-data", "r2"],
) as dag:

    task_collect_and_upload_rainfall_training_data = PythonOperator(
        task_id="collect_and_upload_rainfall_training_data",
        python_callable=collect_and_upload_rainfall_training_data,
    )
