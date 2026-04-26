from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import os
import tempfile
from pathlib import Path

import boto3
import pandas as pd

from lta_common import iter_speed_bands_pages, build_segments_df

# ================= CONFIG =================

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

# verbatum reference to the latest key used in 6_1_map_r2_context_features.py
R2_POI_LATEST_KEY = os.environ.get(
    "R2_POI_LATEST_KEY",
    "poi/latest/poi_latest.parquet",
)

# New key for road segments
R2_ROAD_SEGMENTS_LATEST_KEY = "road_segments/latest/road_segments_latest.parquet"

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
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

# ================= TASK =================

def archive_road_segments_to_r2():
    all_segments = []
    
    logger.info("STEP 1: Iterating through LTA speed band pages for segments")
    for skip, df_page in iter_speed_bands_pages(page_size=500):
        segments_df = build_segments_df(df_page)
        if not segments_df.empty:
            all_segments.append(segments_df)
        logger.info("Processed skip=%s, found %s segments", skip, len(segments_df))

    if not all_segments:
        logger.warning("No road segments found to archive.")
        return

    # Combine all pages into one master segment dataframe
    master_segments_df = pd.concat(all_segments, ignore_index=True)
    master_segments_df.drop_duplicates(subset=['link_id'], inplace=True)
    
    logger.info("STEP 2: Preparing upload for %s unique segments", len(master_segments_df))

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "road_segments_latest.parquet"
        master_segments_df.to_parquet(local_path, index=False)

        client = get_r2_client()
        client.upload_file(str(local_path), R2_BUCKET, R2_ROAD_SEGMENTS_LATEST_KEY)
        
        logger.info(
            "SUCCESS: Uploaded road segments to R2 | bucket=%s | key=%s", 
            R2_BUCKET, 
            R2_ROAD_SEGMENTS_LATEST_KEY
        )

# ================= DAG =================

with DAG(
    dag_id="1_1_archive_road_segments_to_r2",
    default_args=default_args,
    description="Fetch static road segment metadata and archive as a master Parquet file in R2",
    schedule=None, # Manually triggered like your Supabase version
    start_date=datetime(2026, 4, 21),
    catchup=False,
    tags=["lta", "traffic", "r2", "static"],
) as dag:

    run_archive_road_segments = PythonOperator(
        task_id="archive_road_segments_to_r2_task",
        python_callable=archive_road_segments_to_r2,
    )