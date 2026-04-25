from datetime import datetime, timedelta
import logging
import traceback

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_common import (
    FEATURE_COLUMNS,
    get_connection,
    load_active_model,
    load_latest_prediction_frame,
    save_predictions,
)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def predict_congestion():
    conn = None
    try:
        logger.info("STEP 1: loading active model")
        conn = get_connection()
        model_payload = load_active_model(conn)

        logger.info("STEP 2: loading latest traffic-rainfall rows for prediction")
        prediction_df = load_latest_prediction_frame(conn)
        if prediction_df.empty:
            logger.warning("No latest rows available for prediction")
            return

        prediction_df["is_weekend"] = prediction_df["is_weekend"].astype(int)
        for column in FEATURE_COLUMNS:
            prediction_df[column] = pd.to_numeric(prediction_df[column], errors="coerce")
        prediction_df[FEATURE_COLUMNS] = prediction_df[FEATURE_COLUMNS].fillna(0)

        logger.info("STEP 3: predicting rows=%s model_id=%s", len(prediction_df), model_payload["model_id"])
        prediction_df["predicted_congestion_score"] = model_payload["model"].predict(
            prediction_df[FEATURE_COLUMNS]
        )

        inserted_rows = save_predictions(
            conn,
            prediction_df,
            model_id=model_payload["model_id"],
            model_name=model_payload["model_name"],
            lookahead_minutes=15,
        )
        logger.info("SUCCESS: saved %s congestion predictions", inserted_rows)
    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn is not None:
            conn.close()


with DAG(
    dag_id="8_predict_congestion",
    default_args=default_args,
    description="Predict 15-minute-ahead congestion using the active trained model",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "prediction", "congestion"],
) as dag:

    run_predict_congestion = PythonOperator(
        task_id="predict_congestion_task",
        python_callable=predict_congestion,
    )
