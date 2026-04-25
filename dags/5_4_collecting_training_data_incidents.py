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

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

R2_INCIDENTS_PREFIX = os.environ.get("R2_INCIDENTS_PREFIX", "traffic_incidents/")
WRITE_LATEST_INCIDENTS_ALIAS = os.environ.get("WRITE_LATEST_INCIDENTS_ALIAS", "true").strip().lower() == "true"

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
        logger.warning("Could not delete temporary incidents parquet: %s", local_path)


def build_partitioned_key(prefix: str, file_stem: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    hour_part = file_label[9:11]
    return f"{prefix}date={date_part}/hour={hour_part}/{file_stem}_{file_label}.parquet"


def fetch_traffic_incidents_payload() -> list[dict]:
    headers = {
        "AccountKey": LTA_ACCOUNT_KEY,
        "accept": "application/json",
    }

    response = requests.get(API_URL, headers=headers, timeout=30)
    response.raise_for_status()

    payload = response.json()
    return payload.get("value", [])


def build_traffic_incidents_df(rows: list[dict], collected_at: datetime) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=[
            "type",
            "message",
            "latitude",
            "longitude",
            "distance",
            "collected_at",
        ])

    rename_map = {}
    for src, dst in [
        ("Type", "type"),
        ("Message", "message"),
        ("Latitude", "latitude"),
        ("Longitude", "longitude"),
        ("Distance", "distance"),
    ]:
        if src in df.columns:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["type", "message", "latitude", "longitude", "distance"] if c in df.columns]
    df = df[keep_cols].copy()
    df["collected_at"] = collected_at

    return df

# ================= TASK =================

def collect_and_upload_traffic_incidents_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching traffic incidents payload")
    rows = fetch_traffic_incidents_payload()

    logger.info("STEP 2: building traffic incidents dataframe")
    incidents_df = build_traffic_incidents_df(rows, collected_at=collected_at)

    logger.info("Traffic incidents training data prepared | rows=%s", len(incidents_df))

    if not incidents_df.empty:
        incidents_key = build_partitioned_key(
            R2_INCIDENTS_PREFIX,
            "traffic_incidents_snapshot",
            file_label,
        )
        upload_dataframe_to_r2(incidents_df, incidents_key)

        if WRITE_LATEST_INCIDENTS_ALIAS:
            latest_incidents_key = "traffic_incidents/latest/traffic_incidents_latest.parquet"
            upload_dataframe_to_r2(incidents_df, latest_incidents_key)
    else:
        logger.warning("No traffic incidents found; skipping incidents upload")

# ================= DAG =================

with DAG(
    dag_id="5_4_collecting_training_data_incidents",
    default_args=default_args,
    description="Collect LTA Traffic Incidents data and save to Cloudflare R2 as partitioned parquet training data.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "incidents", "training-data", "r2"],
) as dag:

    task_collect_and_upload_traffic_incidents_training_data = PythonOperator(
        task_id="collect_and_upload_traffic_incidents_training_data",
        python_callable=collect_and_upload_traffic_incidents_training_data,
    )