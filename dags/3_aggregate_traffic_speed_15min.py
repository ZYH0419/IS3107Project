from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import logging
import traceback

from lta_common import get_connection

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def floor_to_15min(dt: datetime) -> datetime:
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def aggregate_traffic_speed_15min():
    conn = None

    try:
        now_utc = datetime.now(timezone.utc)
        window_end = floor_to_15min(now_utc)
        window_start = window_end - timedelta(minutes=15)

        logger.info("STEP 1: aggregating window %s to %s", window_start, window_end)

        conn = get_connection()
        logger.info("STEP 2: connected to Supabase")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO traffic_speed_15min (
                    interval_start,
                    link_id,
                    avg_speed_band,
                    min_speed_band,
                    max_speed_band,
                    avg_minimum_speed,
                    avg_maximum_speed,
                    samples
                )
                SELECT
                    %s AS interval_start,
                    link_id,
                    AVG(speed_band)::numeric(6,3) AS avg_speed_band,
                    MIN(speed_band) AS min_speed_band,
                    MAX(speed_band) AS max_speed_band,
                    AVG(minimum_speed)::numeric(8,3) AS avg_minimum_speed,
                    AVG(maximum_speed)::numeric(8,3) AS avg_maximum_speed,
                    COUNT(*) AS samples
                FROM traffic_speed_recent
                WHERE collected_at >= %s
                  AND collected_at < %s
                GROUP BY link_id
                ON CONFLICT (interval_start, link_id) DO UPDATE SET
                    avg_speed_band = EXCLUDED.avg_speed_band,
                    min_speed_band = EXCLUDED.min_speed_band,
                    max_speed_band = EXCLUDED.max_speed_band,
                    avg_minimum_speed = EXCLUDED.avg_minimum_speed,
                    avg_maximum_speed = EXCLUDED.avg_maximum_speed,
                    samples = EXCLUDED.samples,
                    inserted_at = now()
                """,
                (window_start, window_start, window_end),
            )
        conn.commit()

        logger.info("SUCCESS: aggregate_traffic_speed_15min finished")

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


with DAG(
    dag_id="3_aggregate_traffic_speed_15min",
    default_args=default_args,
    description="Aggregate recent traffic data into 15-minute history",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 3, 20),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "supabase", "aggregation"],
) as dag:

    run_aggregate_traffic_speed_15min = PythonOperator(
        task_id="aggregate_traffic_speed_15min_task",
        python_callable=aggregate_traffic_speed_15min,
    )