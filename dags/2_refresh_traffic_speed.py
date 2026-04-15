from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import logging
import traceback

from lta_common import (
    iter_speed_bands_pages,
    build_snapshots_df,
    get_connection,
    insert_snapshot_rows,
    upsert_latest_snapshot_rows,
)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def refresh_traffic_speed():
    conn = None

    try:
        logger.info("STEP 1: start refresh_traffic_speed")
        conn = get_connection()
        logger.info("STEP 2: connected to Supabase")

        collected_at = datetime.now(timezone.utc)
        logger.info("STEP 3: snapshot timestamp = %s", collected_at)

        total_rows_latest = 0
        total_rows_recent = 0
        total_pages = 0

        for skip, df_page in iter_speed_bands_pages(page_size=500):
            logger.info("STEP 4: processing page skip=%s rows=%s", skip, len(df_page))

            snapshots_df = build_snapshots_df(df_page, collected_at)
            logger.info("STEP 5: built cleaned snapshot dataframe rows=%s", len(snapshots_df))

            # latest table = latest known usable state
            upsert_latest_snapshot_rows(conn, snapshots_df, batch_size=500)
            total_rows_latest += len(snapshots_df)

            # recent table = raw cleaned history
            insert_snapshot_rows(conn, "traffic_speed_recent", snapshots_df, batch_size=500)
            total_rows_recent += len(snapshots_df)

            total_pages += 1

            logger.info(
                "STEP 6: page saved skip=%s | pages=%s | latest_rows=%s | recent_rows=%s",
                skip, total_pages, total_rows_latest, total_rows_recent
            )

        logger.info("SUCCESS: refresh_traffic_speed finished")

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


with DAG(
    dag_id="2_refresh_traffic_speed",
    default_args=default_args,
    description="Refresh traffic snapshot page by page and save immediately",
    schedule="*/10 * * * *",
    start_date=datetime(2026, 3, 20),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "supabase", "realtime"],
) as dag:

    run_refresh_traffic_speed = PythonOperator(
        task_id="refresh_traffic_speed_task",
        python_callable=refresh_traffic_speed,
    )