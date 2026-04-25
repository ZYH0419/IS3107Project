from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import boto3
import pandas as pd
import requests

from lta_common import build_snapshots_df

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

# Local directory for parquet snapshots
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

# Cloudflare R2 configuration
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

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


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def retrieve_raw_snapshot(**context):
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")
    ts_label = collected_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    logger.info("Start retrieving traffic speed bands for %s", ts_label)

    df = fetch_all_speed_bands()

    if df.empty:
        raise ValueError("Retrieved 0 rows from LTA API")

    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"traffic_speed_raw_snapshot_{file_label}.parquet"
    df.to_parquet(raw_path, index=False)

    logger.info("Saved raw parquet snapshot: %s | rows=%s", raw_path, len(df))

    ti = context["ti"]
    ti.xcom_push(key="file_label", value=file_label)
    ti.xcom_push(key="collected_at_iso", value=collected_at.isoformat())
    ti.xcom_push(key="raw_path", value=str(raw_path))


def clean_snapshot(**context):
    ti = context["ti"]

    raw_path = Path(ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="raw_path"))
    file_label = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="file_label")
    collected_at_iso = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="collected_at_iso")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw snapshot file not found: {raw_path}")

    collected_at = datetime.fromisoformat(collected_at_iso)

    logger.info("Cleaning raw snapshot: %s", raw_path)

    raw_df = pd.read_parquet(raw_path)

    snapshots_df = build_snapshots_df(raw_df, collected_at=collected_at)

    # Drop collected_at before saving parquet snapshot for feature engineering
    cleaned_df = snapshots_df.drop(columns=["collected_at"]).copy()

    output_dir = Path(RAW_DATA_DIR)
    cleaned_path = output_dir / f"traffic_speed_snapshot_{file_label}.parquet"
    cleaned_df.to_parquet(cleaned_path, index=False)

    logger.info(
        "Saved cleaned parquet snapshot: %s | rows=%s | null_speed_band=%s | null_minimum_speed=%s | null_maximum_speed=%s",
        cleaned_path,
        len(cleaned_df),
        int(cleaned_df["speed_band"].isna().sum()),
        int(cleaned_df["minimum_speed"].isna().sum()),
        int(cleaned_df["maximum_speed"].isna().sum()),
    )

    ti.xcom_push(key="cleaned_path", value=str(cleaned_path))


def upload_cleaned_snapshot_to_r2(**context):
    ti = context["ti"]

    cleaned_path = Path(ti.xcom_pull(task_ids="clean_snapshot", key="cleaned_path"))
    file_label = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="file_label")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned snapshot file not found: {cleaned_path}")

    client = get_r2_client()

    date_part = file_label[:8]
    hour_part = file_label[9:11]
    object_key = (
        f"traffic_speed/date={date_part}/hour={hour_part}/"
        f"traffic_speed_snapshot_{file_label}.parquet"
    )

    client.upload_file(str(cleaned_path), R2_BUCKET, object_key)

    logger.info(
        "Uploaded cleaned snapshot to R2: bucket=%s | key=%s | local_path=%s",
        R2_BUCKET,
        object_key,
        cleaned_path,
    )

    ti.xcom_push(key="r2_key", value=object_key)


def cleanup_local_files(**context):
    """
    Delete BOTH local files only after a successful R2 upload.
    """
    ti = context["ti"]

    uploaded_key = ti.xcom_pull(task_ids="upload_cleaned_snapshot_to_r2", key="r2_key")
    raw_path_str = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="raw_path")
    cleaned_path_str = ti.xcom_pull(task_ids="clean_snapshot", key="cleaned_path")

    if not uploaded_key:
        logger.info("No successful R2 upload key found; skip local cleanup for safety")
        return

    for path_str in [raw_path_str, cleaned_path_str]:
        if not path_str:
            continue
        path = Path(path_str)
        if path.exists():
            path.unlink()
            logger.info("Deleted local file after successful R2 upload: %s", path)
        else:
            logger.info("Local file already absent: %s", path)


with DAG(
    dag_id="5_1_collecting_training_data",
    default_args=default_args,
    description="Retrieve full LTA Traffic Speed Bands every 5 minutes, save raw parquet locally, clean it using lta_common, save cleaned parquet locally, upload the cleaned snapshot to Cloudflare R2, then delete both local files.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 3, 17),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "raw", "cleaning", "parquet", "r2"],
) as dag:

    task_retrieve_raw_snapshot = PythonOperator(
        task_id="retrieve_raw_snapshot",
        python_callable=retrieve_raw_snapshot,
    )

    task_clean_snapshot = PythonOperator(
        task_id="clean_snapshot",
        python_callable=clean_snapshot,
    )

    task_upload_cleaned_snapshot_to_r2 = PythonOperator(
        task_id="upload_cleaned_snapshot_to_r2",
        python_callable=upload_cleaned_snapshot_to_r2,
    )

    task_cleanup_local_files = PythonOperator(
        task_id="cleanup_local_files",
        python_callable=cleanup_local_files,
        trigger_rule="all_done",
    )

    (
        task_retrieve_raw_snapshot
        >> task_clean_snapshot
        >> task_upload_cleaned_snapshot_to_r2
        >> task_cleanup_local_files
    )
