from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os
import tempfile

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

R2_INCIDENTS_PREFIX = os.environ.get("R2_INCIDENTS_PREFIX", "traffic_incidents/")
R2_INCIDENTS_LATEST_KEY = os.environ.get(
    "R2_INCIDENTS_LATEST_KEY",
    "traffic_incidents/latest/traffic_incidents_latest.parquet",
)
WRITE_LATEST_INCIDENTS_ALIAS = (
    os.environ.get("WRITE_LATEST_INCIDENTS_ALIAS", "true").strip().lower() == "true"
)

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
    Write dataframe to a temporary parquet file, upload it to R2, then let the
    temporary directory clean itself up. This avoids leaving local parquet files
    in the Airflow worker/container.
    """
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        df.to_parquet(local_path, index=False)
        client.upload_file(str(local_path), R2_BUCKET, key)

    logger.info(
        "Uploaded dataframe to R2: bucket=%s | key=%s | rows=%s",
        R2_BUCKET,
        key,
        len(df),
    )


def build_partitioned_key(prefix: str, file_stem: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    hour_part = file_label[9:11]
    return f"{prefix}date={date_part}/hour={hour_part}/{file_stem}_{file_label}.parquet"


def empty_incidents_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "type",
            "message",
            "latitude",
            "longitude",
            "distance",
            "collected_at",
        ]
    )


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
        out = empty_incidents_df()
        out["collected_at"] = pd.to_datetime(out["collected_at"], utc=True, errors="coerce")
        return out

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

    for col in ["type", "message", "latitude", "longitude", "distance"]:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[["type", "message", "latitude", "longitude", "distance"]].copy()

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    out["type"] = out["type"].astype("string")
    out["message"] = out["message"].astype("string")
    out["collected_at"] = collected_at

    before = len(out)
    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out[
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    ].copy()
    dropped_invalid_coords = before - len(out)

    if dropped_invalid_coords:
        logger.warning("Dropped %s traffic incident rows with invalid coordinates", dropped_invalid_coords)

    out = out.drop_duplicates(
        subset=["type", "message", "latitude", "longitude", "distance"],
        keep="first",
    ).copy()

    return out[["type", "message", "latitude", "longitude", "distance", "collected_at"]]

# ================= TASK =================

def collect_and_upload_traffic_incidents_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching traffic incidents payload")
    rows = fetch_traffic_incidents_payload()
    logger.info("Fetched traffic incident rows from API: %s", len(rows))

    logger.info("STEP 2: building cleaned traffic incidents dataframe")
    incidents_df = build_traffic_incidents_df(rows, collected_at=collected_at)

    logger.info(
        "Traffic incidents dataframe prepared | rows=%s | collected_at=%s",
        len(incidents_df),
        collected_at.isoformat(),
    )

    incidents_key = build_partitioned_key(
        R2_INCIDENTS_PREFIX,
        "traffic_incidents_snapshot",
        file_label,
    )

    # Always write the historical snapshot, even when there are no incidents.
    # An empty snapshot is still useful because it records that the API was checked.
    upload_dataframe_to_r2(incidents_df, incidents_key)

    if WRITE_LATEST_INCIDENTS_ALIAS:
        # Always update latest, even if empty. Otherwise a no-incident period would
        # leave an old stale incident file behind and DAG 6_1 / DAG 8 may overcount.
        upload_dataframe_to_r2(incidents_df, R2_INCIDENTS_LATEST_KEY)

# ================= DAG =================

with DAG(
    dag_id="5_4_collecting_training_data_incidents",
    default_args=default_args,
    description="Collect LTA Traffic Incidents data and save history plus latest alias to Cloudflare R2.",
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
