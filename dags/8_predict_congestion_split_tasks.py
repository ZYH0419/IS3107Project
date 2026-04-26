from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import re
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

# ================= CONFIG =================

R2_ENDPOINT_URL = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET"]

MODEL_R2_PREFIX = os.environ.get("MODEL_R2_PREFIX", "models")
LATEST_MODEL_KEY = f"{MODEL_R2_PREFIX}/latest/best_congestion_model.joblib"

R2_PREDICTION_TMP_PREFIX = os.environ.get(
    "R2_PREDICTION_TMP_PREFIX",
    "prediction_runs/tmp",
)

PREDICTION_LOOKAHEAD_MINUTES = int(os.environ.get("PREDICTION_LOOKAHEAD_MINUTES", "15"))

DEFAULT_MODEL_ID = int(os.environ.get("R2_LATEST_MODEL_ID", "-1"))
DEFAULT_MODEL_NAME = os.environ.get("R2_LATEST_MODEL_NAME", "r2_latest_model")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


# ================= HELPERS =================

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def safe_run_id(raw_run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=]+", "_", raw_run_id)


def build_tmp_key(run_id: str, filename: str) -> str:
    return f"{R2_PREDICTION_TMP_PREFIX.rstrip('/')}/{safe_run_id(run_id)}/{filename}"


def upload_dataframe_to_r2(df: pd.DataFrame, key: str) -> None:
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        df.to_parquet(local_path, index=False)
        client.upload_file(str(local_path), R2_BUCKET_NAME, key)

    logger.info("Uploaded dataframe to R2 | key=%s | rows=%s", key, len(df))


def read_dataframe_from_r2(key: str) -> pd.DataFrame:
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        logger.info("Downloading dataframe from R2 | key=%s", key)
        client.download_file(R2_BUCKET_NAME, key, str(local_path))
        return pd.read_parquet(local_path)


def load_latest_model_from_r2():
    r2_client = get_r2_client()

    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        logger.info("Downloading latest model from R2 | key=%s", LATEST_MODEL_KEY)
        r2_client.download_file(R2_BUCKET_NAME, LATEST_MODEL_KEY, tmp.name)
        model = joblib.load(tmp.name)

    logger.info("Successfully loaded latest model from R2")
    return model


def clean_prediction_features(prediction_df: pd.DataFrame) -> pd.DataFrame:
    prediction_df = prediction_df.copy()

    if prediction_df.empty:
        return prediction_df

    if "is_weekend" in prediction_df.columns:
        prediction_df["is_weekend"] = prediction_df["is_weekend"].astype(int)

    for column in FEATURE_COLUMNS:
        if column not in prediction_df.columns:
            logger.warning("Missing prediction feature column=%s. Filling with 0.", column)
            prediction_df[column] = 0

        prediction_df[column] = pd.to_numeric(
            prediction_df[column],
            errors="coerce",
        )

    prediction_df[FEATURE_COLUMNS] = prediction_df[FEATURE_COLUMNS].fillna(0)

    return prediction_df


# ================= TASKS =================

def prepare_prediction_frame(**context) -> str | None:
    conn = None

    try:
        logger.info("STEP 1: connecting to Supabase serving layer")
        conn = get_connection()

        logger.info("STEP 2: loading latest rows for prediction")
        prediction_df = load_latest_prediction_frame(conn)

        if prediction_df.empty:
            logger.warning("No latest rows available for prediction")
            return None

        logger.info("STEP 3: cleaning prediction features | rows=%s", len(prediction_df))
        prediction_df = clean_prediction_features(prediction_df)

        run_id = context["dag_run"].run_id
        input_key = build_tmp_key(run_id, "prediction_input.parquet")

        logger.info("STEP 4: saving prepared prediction frame to R2")
        upload_dataframe_to_r2(prediction_df, input_key)

        logger.info("SUCCESS: prepared prediction frame | key=%s | rows=%s", input_key, len(prediction_df))
        return input_key

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()


def score_prediction_frame(**context) -> str | None:
    try:
        ti = context["ti"]
        input_key = ti.xcom_pull(task_ids="prepare_prediction_frame")

        if not input_key:
            logger.warning("No input prediction frame key found. Skipping scoring.")
            return None

        logger.info("STEP 1: loading prepared prediction frame from R2")
        prediction_df = read_dataframe_from_r2(input_key)

        if prediction_df.empty:
            logger.warning("Prepared prediction frame is empty. Skipping scoring.")
            return None

        logger.info("STEP 2: loading latest model from R2")
        model = load_latest_model_from_r2()

        logger.info("STEP 3: scoring prediction frame | rows=%s", len(prediction_df))
        prediction_df["predicted_congestion_score"] = model.predict(
            prediction_df[FEATURE_COLUMNS]
        )

        run_id = context["dag_run"].run_id
        scored_key = build_tmp_key(run_id, "prediction_scored.parquet")

        logger.info("STEP 4: saving scored predictions to R2")
        upload_dataframe_to_r2(prediction_df, scored_key)

        logger.info("SUCCESS: scored predictions | key=%s | rows=%s", scored_key, len(prediction_df))
        return scored_key

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise


def save_predictions_to_supabase(**context) -> None:
    conn = None

    try:
        ti = context["ti"]
        scored_key = ti.xcom_pull(task_ids="score_prediction_frame")

        if not scored_key:
            logger.warning("No scored prediction key found. Nothing to save.")
            return

        logger.info("STEP 1: loading scored predictions from R2")
        prediction_df = read_dataframe_from_r2(scored_key)

        if prediction_df.empty:
            logger.warning("Scored prediction dataframe is empty. Nothing to save.")
            return

        logger.info("STEP 2: connecting to Supabase")
        conn = get_connection()

        logger.info("STEP 3: saving predictions to Supabase | rows=%s", len(prediction_df))
        inserted_rows = save_predictions(
            conn,
            prediction_df,
            model_id=DEFAULT_MODEL_ID,
            model_name=DEFAULT_MODEL_NAME,
            lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES,
        )

        logger.info("SUCCESS: saved predictions to Supabase | rows=%s", inserted_rows)

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()


# ================= DAG =================

with DAG(
    dag_id="8_predict_congestion",
    default_args=default_args,
    description="Predict 15 minute ahead congestion using the latest R2 model and save predictions to Supabase",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "prediction", "congestion", "r2", "supabase"],
) as dag:

    prepare_prediction_frame_task = PythonOperator(
        task_id="prepare_prediction_frame",
        python_callable=prepare_prediction_frame,
    )

    score_prediction_frame_task = PythonOperator(
        task_id="score_prediction_frame",
        python_callable=score_prediction_frame,
    )

    save_predictions_to_supabase_task = PythonOperator(
        task_id="save_predictions_to_supabase",
        python_callable=save_predictions_to_supabase,
    )

    prepare_prediction_frame_task >> score_prediction_frame_task >> save_predictions_to_supabase_task
