from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import os
import logging
import traceback

import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]
SUPABASE_DB_URI = os.environ["SUPABASE_DB_URI"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def get_db_dsn():
    if SUPABASE_DB_URI.startswith("postgresql+psycopg2://"):
        return SUPABASE_DB_URI.replace("postgresql+psycopg2://", "postgresql://", 1)
    return SUPABASE_DB_URI


def test_lta_to_supabase():
    try:
        logger.info("STEP 1: starting test DAG")

        headers = {
            "AccountKey": LTA_ACCOUNT_KEY,
            "accept": "application/json",
        }

        logger.info("STEP 2: calling LTA API")
        response = requests.get(API_URL, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        records = data.get("value", [])

        logger.info("STEP 3: API returned %s rows", len(records))

        if not records:
            logger.warning("No records returned from API")
            return

        # only first 10 rows
        df = pd.DataFrame(records).head(10).copy()

        logger.info("STEP 4: trimmed to first %s rows", len(df))
        logger.info("STEP 4.1: sample LinkIDs = %s", df["LinkID"].tolist())

        collected_at = datetime.now(timezone.utc)

        insert_df = df[
            [
                "LinkID",
                "RoadName",
                "RoadCategory",
                "SpeedBand",
                "MinimumSpeed",
                "MaximumSpeed",
                "StartLon",
                "StartLat",
                "EndLon",
                "EndLat",
            ]
        ].copy()

        insert_df["collected_at"] = collected_at

        insert_df = insert_df[
            [
                "collected_at",
                "LinkID",
                "RoadName",
                "RoadCategory",
                "SpeedBand",
                "MinimumSpeed",
                "MaximumSpeed",
                "StartLon",
                "StartLat",
                "EndLon",
                "EndLat",
            ]
        ]

        insert_df.columns = [
            "collected_at",
            "link_id",
            "road_name",
            "road_category",
            "speed_band",
            "minimum_speed",
            "maximum_speed",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
        ]

        logger.info("STEP 5: prepared insert dataframe with %s rows", len(insert_df))

        rows = list(insert_df.itertuples(index=False, name=None))

        sql = """
            INSERT INTO lta_test_10 (
                collected_at,
                link_id,
                road_name,
                road_category,
                speed_band,
                minimum_speed,
                maximum_speed,
                start_lon,
                start_lat,
                end_lon,
                end_lat
            )
            VALUES %s
        """

        logger.info("STEP 6: connecting to Supabase")
        conn = psycopg2.connect(get_db_dsn())

        try:
            with conn.cursor() as cur:
                logger.info("STEP 7: inserting rows into lta_test_10")
                execute_values(cur, sql, rows, page_size=10)
            conn.commit()
            logger.info("STEP 8: insert committed successfully")
        finally:
            conn.close()
            logger.info("STEP 9: database connection closed")

        logger.info("SUCCESS: test DAG completed")

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise


with DAG(
    dag_id="test_lta_10_rows",
    default_args=default_args,
    description="Test DAG: retrieve first 10 rows from LTA and save to Supabase",
    schedule=None,
    start_date=datetime(2026, 3, 20),
    catchup=False,
    tags=["test", "lta", "supabase"],
) as dag:

    run_test_lta_to_supabase = PythonOperator(
        task_id="test_lta_to_supabase_task",
        python_callable=test_lta_to_supabase,
    )