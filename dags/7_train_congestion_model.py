from datetime import datetime, timedelta
import logging
import os
import traceback

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_common import get_connection, load_training_frame, save_best_model, train_candidate_models

logger = logging.getLogger(__name__)
ML_TRAINING_LOOKBACK_HOURS = int(os.environ.get("ML_TRAINING_LOOKBACK_HOURS", "24"))

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def train_congestion_model():
    conn = None
    try:
        logger.info(
            "STEP 1: loading labeled ML training frame (lookback_hours=%s)",
            ML_TRAINING_LOOKBACK_HOURS,
        )
        conn = get_connection()
        training_df = load_training_frame(
            conn,
            lookahead_minutes=15,
            lookback_hours=ML_TRAINING_LOOKBACK_HOURS,
        )
        logger.info("STEP 2: labeled rows=%s", len(training_df))

        results = train_candidate_models(training_df)
        best_result = results[0]

        for result in results:
            logger.info(
                "MODEL RESULT | name=%s train_rows=%s test_rows=%s mae=%.4f rmse=%.4f r2=%.4f",
                result["model_name"],
                result["train_rows"],
                result["test_rows"],
                result["mae"],
                result["rmse"],
                result["r2"],
            )

        model_id = save_best_model(conn, best_result)
        logger.info(
            "SUCCESS: saved active model_id=%s name=%s mae=%.4f rmse=%.4f r2=%.4f",
            model_id,
            best_result["model_name"],
            best_result["mae"],
            best_result["rmse"],
            best_result["r2"],
        )
    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn is not None:
            conn.close()


with DAG(
    dag_id="7_train_congestion_model",
    default_args=default_args,
    description="Train baseline and advanced congestion prediction models",
    schedule=None,
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "congestion", "model"],
) as dag:

    run_train_congestion_model = PythonOperator(
        task_id="train_congestion_model_task",
        python_callable=train_congestion_model,
    )
