from datetime import datetime, timedelta
import logging
import os
import tempfile
import traceback

import boto3
import joblib
import pandas as pd

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_common import (
    FEATURE_COLUMNS,
    get_connection,
    load_latest_prediction_frame,
    save_predictions,
)

logger = logging.getLogger(__name__)

R2_ENDPOINT_URL = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET"]

MODEL_R2_PREFIX = os.environ.get("MODEL_R2_PREFIX", "models")
LATEST_MODEL_KEY = f"{MODEL_R2_PREFIX}/latest/best_congestion_model.joblib"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def load_latest_model_from_r2():
    r2_client = get_r2_client()

    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        logger.info("Downloading latest model from R2: %s", LATEST_MODEL_KEY)

        r2_client.download_file(
            R2_BUCKET_NAME,
            LATEST_MODEL_KEY,
            tmp.name,
        )

        model = joblib.load(tmp.name)

    logger.info("Successfully loaded latest model from R2")
    return model


def predict_congestion():
    conn = None

    try:
        logger.info("STEP 1: loading latest model from R2")
        model = load_latest_model_from_r2()

        logger.info("STEP 2: connecting to Supabase serving layer")
        conn = get_connection()

        logger.info("STEP 3: loading latest traffic-rainfall rows for prediction")
        prediction_df = load_latest_prediction_frame(conn)

        if prediction_df.empty:
            logger.warning("No latest rows available for prediction")
            return

        prediction_df["is_weekend"] = prediction_df["is_weekend"].astype(int)

        for column in FEATURE_COLUMNS:
            prediction_df[column] = pd.to_numeric(
                prediction_df[column],
                errors="coerce",
            )

        prediction_df[FEATURE_COLUMNS] = prediction_df[FEATURE_COLUMNS].fillna(0)

        logger.info("STEP 4: predicting rows=%s using R2 latest model", len(prediction_df))

        prediction_df["predicted_congestion_score"] = model.predict(
            prediction_df[FEATURE_COLUMNS]
        )

        logger.info("STEP 5: saving predictions to Supabase")

        inserted_rows = save_predictions(
            conn,
            prediction_df,
            model_id=-1,
            model_name="r2_latest_model",
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
    description="Predict 15-minute-ahead congestion using latest R2 model and save predictions to Supabase",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "prediction", "congestion", "r2", "supabase"],
) as dag:

    run_predict_congestion = PythonOperator(
        task_id="predict_congestion_task",
        python_callable=predict_congestion,
    )