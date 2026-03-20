from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import traceback
import os

from lta_common import get_connection

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

RECENT_RETENTION_HOURS = int(os.environ.get("RECENT_RETENTION_HOURS", "24"))


def cleanup_recent_history():
    conn = None

    try:
        logger.info("STEP 1: cleaning old recent history > %s hours", RECENT_RETENTION_HOURS)

        conn = get_connection()
        logger.info("STEP 2: connected to Supabase")

        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM traffic_speed_recent
                WHERE collected_at < now() - (%s || ' hours')::interval
                """,
                (str(RECENT_RETENTION_HOURS),),
            )
            deleted_rows = cur.rowcount

        conn.commit()
        logger.info("SUCCESS: deleted %s old rows from traffic_speed_recent", deleted_rows)

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


with DAG(
    dag_id="4_cleanup_recent_history",
    default_args=default_args,
    description="Delete old rows from traffic_speed_recent",
    schedule="0 * * * *",
    start_date=datetime(2026, 3, 20),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "supabase", "cleanup"],
) as dag:

    run_cleanup_recent_history = PythonOperator(
        task_id="cleanup_recent_history_task",
        python_callable=cleanup_recent_history,
    )