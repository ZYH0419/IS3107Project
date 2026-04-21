from datetime import datetime, timedelta
import logging
import os
import traceback

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from weather_common import refresh_rainfall_data

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def refresh_rainfall():
    try:
        remap_all = os.environ.get("RAINFALL_REMAP_ALL", "false").lower() == "true"
        logger.info("STEP 1: start refresh_rainfall | remap_all=%s", remap_all)
        refresh_rainfall_data(remap_all=remap_all)
        logger.info("SUCCESS: refresh_rainfall finished")
    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise


with DAG(
    dag_id="5_refresh_rainfall",
    default_args=default_args,
    description="Collect 5-minute rainfall readings and map weather stations to road segments",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["weather", "rainfall", "traffic", "supabase"],
) as dag:

    run_refresh_rainfall = PythonOperator(
        task_id="refresh_rainfall_task",
        python_callable=refresh_rainfall,
    )
