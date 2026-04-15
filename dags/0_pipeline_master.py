from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# Reuse the real task functions from your existing DAG files
from importlib import import_module

refresh_traffic_speed = import_module("2_refresh_traffic_speed").refresh_traffic_speed
aggregate_traffic_speed_15min = import_module("3_aggregate_traffic_speed_15min").aggregate_traffic_speed_15min
cleanup_recent_history = import_module("4_cleanup_recent_history").cleanup_recent_history

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="0_pipeline_master",
    default_args=default_args,
    description="Master pipeline orchestrating refresh -> aggregate -> cleanup",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 3, 20),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "supabase", "master", "etl"],
) as dag:

    run_refresh_traffic_speed = PythonOperator(
        task_id="refresh_traffic_speed_task",
        python_callable=refresh_traffic_speed,
    )

    run_aggregate_traffic_speed_15min = PythonOperator(
        task_id="aggregate_traffic_speed_15min_task",
        python_callable=aggregate_traffic_speed_15min,
    )

    run_cleanup_recent_history = PythonOperator(
        task_id="cleanup_recent_history_task",
        python_callable=cleanup_recent_history,
    )

    run_refresh_traffic_speed >> run_aggregate_traffic_speed_15min >> run_cleanup_recent_history