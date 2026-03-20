from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import traceback

from lta_common import (
    iter_speed_bands_pages,
    build_segments_df,
    get_connection,
    upsert_road_segments,
)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def load_road_segments():
    conn = None

    try:
        logger.info("STEP 1: start load_road_segments")
        conn = get_connection()
        logger.info("STEP 2: connected to Supabase")

        total_rows = 0
        total_pages = 0

        for skip, df_page in iter_speed_bands_pages(page_size=500):
            logger.info("STEP 3: processing page skip=%s rows=%s", skip, len(df_page))

            segments_df = build_segments_df(df_page)
            logger.info("STEP 4: built %s unique segment rows for this page", len(segments_df))

            upsert_road_segments(conn, segments_df, batch_size=500)

            total_rows += len(segments_df)
            total_pages += 1

            logger.info(
                "STEP 5: page completed skip=%s | cumulative pages=%s | cumulative rows=%s",
                skip, total_pages, total_rows
            )

        logger.info("SUCCESS: load_road_segments finished")

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


with DAG(
    dag_id="1_load_road_segments",
    default_args=default_args,
    description="Load static road segment metadata page by page",
    schedule=None,
    start_date=datetime(2026, 3, 20),
    catchup=False,
    tags=["lta", "traffic", "supabase", "static"],
) as dag:

    run_load_road_segments = PythonOperator(
        task_id="load_road_segments_task",
        python_callable=load_road_segments,
    )