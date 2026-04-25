from datetime import datetime, timedelta
import logging
import traceback

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_common import collect_training_snapshot, get_connection

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def collect_training_data():
    conn = None
    try:
        logger.info("STEP 1: collecting traffic + rainfall + R2 context-feature ML training snapshot")
        conn = get_connection()
        affected_rows = collect_training_snapshot(conn)
        logger.info("SUCCESS: collected/upserted %s ML training rows", affected_rows)
    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn is not None:
            conn.close()


with DAG(
    dag_id="6_collect_training_data",
    default_args=default_args,
    description="Persist traffic + rainfall + context feature snapshots for ML training",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "traffic", "rainfall", "context-features"],
) as dag:

    run_collect_training_data = PythonOperator(
        task_id="collect_training_data_task",
        python_callable=collect_training_data,
    )
