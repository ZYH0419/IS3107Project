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

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/Taxi-Availability"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

R2_TAXI_PREFIX = os.environ.get("R2_TAXI_PREFIX", "taxi_availability/")
R2_TAXI_LATEST_KEY = os.environ.get(
    "R2_TAXI_LATEST_KEY",
    "taxi_availability/latest/taxi_availability_latest.parquet",
)
WRITE_LATEST_TAXI_ALIAS = (
    os.environ.get("WRITE_LATEST_TAXI_ALIAS", "true").strip().lower() == "true"
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
    Write dataframe to a temporary parquet file, upload to R2, then automatically
    remove the local temporary file.
    """
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        df.to_parquet(local_path, index=False)
        client.upload_file(str(local_path), R2_BUCKET, key)

    logger.info(
        "Uploaded dataframe to R2: bucket=%s | key=%s | rows=%s | columns=%s",
        R2_BUCKET,
        key,
        len(df),
        df.columns.tolist(),
    )


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
        logger.info("Fetching taxi availability page: skip=%s", skip)

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        payload = response.json()
        rows = payload.get("value", [])

        logger.info("Retrieved taxi rows=%s at skip=%s", len(rows), skip)

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        skip += page_size

    return all_rows


def build_taxi_availability_df(rows: list[dict], collected_at: datetime) -> pd.DataFrame:
    """
    Normalise LTA Taxi Availability rows into a clean point dataframe.
    Output is intentionally point-level, not mapped to link_id here.
    Mapping to road segments should happen in 6_1 context-feature building.
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["longitude", "latitude", "collected_at"])

    df.columns = [str(column).strip() for column in df.columns]

    rename_map = {}
    for src, dst in [
        ("Longitude", "longitude"),
        ("Latitude", "latitude"),
        ("longitude", "longitude"),
        ("latitude", "latitude"),
        ("Lon", "longitude"),
        ("Lat", "latitude"),
    ]:
        if src in df.columns:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = ["longitude", "latitude"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Taxi availability response missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = df[required_cols].copy()
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["collected_at"] = collected_at

    before_drop = len(out)
    out = out.dropna(subset=["longitude", "latitude"]).copy()
    out = out[
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    ].copy()
    out = out.drop_duplicates(subset=["longitude", "latitude"]).reset_index(drop=True)

    logger.info(
        "Cleaned taxi dataframe | raw_rows=%s | clean_unique_rows=%s | dropped_or_duplicate=%s",
        before_drop,
        len(out),
        before_drop - len(out),
    )

    return out[["longitude", "latitude", "collected_at"]]

# ================= TASK =================

def collect_and_upload_taxi_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching taxi availability payload")
    rows = fetch_taxi_availability_payload()

    logger.info("STEP 2: building taxi availability dataframe")
    taxi_df = build_taxi_availability_df(rows, collected_at=collected_at)

    logger.info("Taxi training data prepared | rows=%s", len(taxi_df))

    if taxi_df.empty:
        logger.warning("No taxi availability rows found after cleaning; skipping taxi upload")
        return

    logger.info("STEP 3: uploading taxi history parquet")
    taxi_history_key = build_partitioned_key(
        R2_TAXI_PREFIX,
        "taxi_availability_snapshot",
        file_label,
    )
    upload_dataframe_to_r2(taxi_df, taxi_history_key)

    if WRITE_LATEST_TAXI_ALIAS:
        logger.info("STEP 4: uploading taxi latest alias")
        upload_dataframe_to_r2(taxi_df, R2_TAXI_LATEST_KEY)
    else:
        logger.info("WRITE_LATEST_TAXI_ALIAS=false; skipping latest alias upload")

    logger.info(
        "SUCCESS: taxi data uploaded | history_key=%s | latest_key=%s | rows=%s",
        taxi_history_key,
        R2_TAXI_LATEST_KEY if WRITE_LATEST_TAXI_ALIAS else None,
        len(taxi_df),
    )

# ================= DAG =================

with DAG(
    dag_id="5_3_collecting_training_data_taxi",
    default_args=default_args,
    description=(
        "Collect LTA Taxi Availability data every 5 minutes and save to R2 as "
        "partitioned history plus a latest alias for prediction/context features."
    ),
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "taxi", "training-data", "r2", "latest"],
) as dag:

    task_collect_and_upload_taxi_training_data = PythonOperator(
        task_id="collect_and_upload_taxi_training_data",
        python_callable=collect_and_upload_taxi_training_data,
    )
