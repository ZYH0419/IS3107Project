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

# ================= CONFIG =================

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]

# Local directory for temporary parquet snapshots.
# Both raw and cleaned local files are deleted after successful R2 upload.
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

# Cloudflare R2 configuration
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

# Historical partitioned traffic snapshots stay under this prefix.
# These historical files intentionally DO NOT contain collected_at.
R2_TRAFFIC_PREFIX = os.environ.get("R2_TRAFFIC_PREFIX", "traffic_speed/")

# Latest alias is used by prediction.
# This latest file DOES contain collected_at because the filename is constant.
WRITE_LATEST_TRAFFIC_ALIAS = (
    os.environ.get("WRITE_LATEST_TRAFFIC_ALIAS", "true")
    .strip()
    .lower()
    == "true"
)

R2_TRAFFIC_LATEST_KEY = os.environ.get(
    "R2_TRAFFIC_LATEST_KEY",
    "traffic_speed/latest/traffic_speed_latest.parquet",
)

logger = logging.getLogger(__name__)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


# ================= HELPERS =================

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
        logger.info("Fetching LTA TrafficSpeedBands page: skip=%s", skip)

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        payload = response.json()
        records = payload.get("value", [])

        if not records:
            break

        all_data.extend(records)

        logger.info(
            "Fetched TrafficSpeedBands page: skip=%s rows=%s cumulative_rows=%s",
            skip,
            len(records),
            len(all_data),
        )

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


def build_partitioned_key(prefix: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    hour_part = file_label[9:11]

    return (
        f"{prefix}date={date_part}/hour={hour_part}/"
        f"traffic_speed_snapshot_{file_label}.parquet"
    )


# ================= TASKS =================

def retrieve_raw_snapshot(**context):
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")
    ts_label = collected_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    logger.info("STEP 1: retrieving full LTA Traffic Speed Bands for %s", ts_label)

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

    logger.info("STEP 2: cleaning raw traffic snapshot: %s", raw_path)

    raw_df = pd.read_parquet(raw_path)

    snapshots_df = build_snapshots_df(raw_df, collected_at=collected_at)

    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Historical training file:
    # Keep previous behavior. Historical R2 key contains timestamp, so collected_at
    # can be reconstructed from filename by ml_common.py.
    historical_df = snapshots_df.drop(columns=["collected_at"]).copy()
    cleaned_path = output_dir / f"traffic_speed_snapshot_{file_label}.parquet"
    historical_df.to_parquet(cleaned_path, index=False)

    # Latest prediction alias:
    # Keep collected_at because latest alias has a constant filename.
    latest_df = snapshots_df.copy()
    latest_path = output_dir / f"traffic_speed_latest_{file_label}.parquet"
    latest_df.to_parquet(latest_path, index=False)

    logger.info(
        "Saved historical cleaned parquet: %s | rows=%s | null_speed_band=%s | null_minimum_speed=%s | null_maximum_speed=%s",
        cleaned_path,
        len(historical_df),
        int(historical_df["speed_band"].isna().sum()),
        int(historical_df["minimum_speed"].isna().sum()),
        int(historical_df["maximum_speed"].isna().sum()),
    )

    logger.info(
        "Saved latest alias parquet with collected_at: %s | rows=%s | collected_at=%s",
        latest_path,
        len(latest_df),
        collected_at.isoformat(),
    )

    ti.xcom_push(key="cleaned_path", value=str(cleaned_path))
    ti.xcom_push(key="latest_path", value=str(latest_path))


def upload_cleaned_snapshot_to_r2(**context):
    ti = context["ti"]

    cleaned_path = Path(ti.xcom_pull(task_ids="clean_snapshot", key="cleaned_path"))
    latest_path = Path(ti.xcom_pull(task_ids="clean_snapshot", key="latest_path"))
    file_label = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="file_label")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned snapshot file not found: {cleaned_path}")

    client = get_r2_client()

    historical_key = build_partitioned_key(R2_TRAFFIC_PREFIX, file_label)

    logger.info("STEP 3A: uploading historical traffic snapshot to R2")
    client.upload_file(str(cleaned_path), R2_BUCKET, historical_key)

    logger.info(
        "Uploaded historical traffic snapshot to R2: bucket=%s | key=%s | local_path=%s",
        R2_BUCKET,
        historical_key,
        cleaned_path,
    )

    latest_key = None

    if WRITE_LATEST_TRAFFIC_ALIAS:
        if not latest_path.exists():
            raise FileNotFoundError(f"Latest alias parquet file not found: {latest_path}")

        logger.info("STEP 3B: uploading latest traffic alias to R2")
        client.upload_file(str(latest_path), R2_BUCKET, R2_TRAFFIC_LATEST_KEY)

        latest_key = R2_TRAFFIC_LATEST_KEY

        logger.info(
            "Uploaded latest traffic alias to R2: bucket=%s | key=%s | local_path=%s",
            R2_BUCKET,
            latest_key,
            latest_path,
        )
    else:
        logger.info("WRITE_LATEST_TRAFFIC_ALIAS=false, skipped latest alias upload")

    ti.xcom_push(key="r2_key", value=historical_key)
    ti.xcom_push(key="latest_r2_key", value=latest_key)


def cleanup_local_files(**context):
    """
    Delete local files only after successful R2 upload.

    The task uses trigger_rule=all_done, but it checks upload XComs before deleting.
    This prevents accidental cleanup when upload fails before keys are pushed.
    """
    ti = context["ti"]

    uploaded_key = ti.xcom_pull(task_ids="upload_cleaned_snapshot_to_r2", key="r2_key")
    raw_path_str = ti.xcom_pull(task_ids="retrieve_raw_snapshot", key="raw_path")
    cleaned_path_str = ti.xcom_pull(task_ids="clean_snapshot", key="cleaned_path")
    latest_path_str = ti.xcom_pull(task_ids="clean_snapshot", key="latest_path")

    if not uploaded_key:
        logger.info("No successful historical R2 upload key found; skip local cleanup for safety")
        return

    for path_str in [raw_path_str, cleaned_path_str, latest_path_str]:
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
    description=(
        "Retrieve full LTA Traffic Speed Bands every 5 minutes, save historical "
        "partitioned snapshots to R2, write a latest traffic alias with collected_at "
        "for prediction, then delete local files."
    ),
    schedule="*/5 * * * *",
    start_date=datetime(2026, 3, 17),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "training", "prediction", "parquet", "r2"],
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
