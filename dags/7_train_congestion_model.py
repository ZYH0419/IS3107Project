from datetime import datetime, timedelta
import logging
import os
import traceback
import tempfile

import boto3
import joblib

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from ml_common import (
    load_training_frame_from_r2,
    train_candidate_models,
)

logger = logging.getLogger(__name__)

ML_TRAINING_LOOKBACK_HOURS = int(os.environ.get("ML_TRAINING_LOOKBACK_HOURS", "24"))

R2_ENDPOINT_URL = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET_NAME = os.environ["R2_BUCKET"]

MODEL_R2_PREFIX = os.environ.get("MODEL_R2_PREFIX", "models")
LATEST_MODEL_KEY = f"{MODEL_R2_PREFIX}/latest/best_congestion_model.joblib"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def extract_model_from_result(best_result):
    for key in ["model", "pipeline", "estimator", "trained_model"]:
        if key in best_result:
            return best_result[key]

    raise KeyError(
        "Could not find trained model in best_result. "
        "Expected one of: model, pipeline, estimator, trained_model."
    )


def upload_model_to_r2(best_result):
    model = extract_model_from_result(best_result)

    trained_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = best_result["model_name"]

    versioned_model_key = (
        f"{MODEL_R2_PREFIX}/history/"
        f"{trained_at}_{model_name}_congestion_model.joblib"
    )

    metadata = {
        "model_name": str(best_result["model_name"]),
        "mae": str(best_result["mae"]),
        "rmse": str(best_result["rmse"]),
        "r2": str(best_result["r2"]),
        "train_rows": str(best_result["train_rows"]),
        "test_rows": str(best_result["test_rows"]),
        "trained_at": trained_at,
    }

    r2_client = get_r2_client()

    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        joblib.dump(model, tmp.name)

        logger.info("Uploading versioned model to R2: %s", versioned_model_key)
        r2_client.upload_file(
            tmp.name,
            R2_BUCKET_NAME,
            versioned_model_key,
            ExtraArgs={"Metadata": metadata},
        )

        logger.info("Uploading latest model alias to R2: %s", LATEST_MODEL_KEY)
        r2_client.upload_file(
            tmp.name,
            R2_BUCKET_NAME,
            LATEST_MODEL_KEY,
            ExtraArgs={"Metadata": metadata},
        )

    return versioned_model_key, LATEST_MODEL_KEY


def train_congestion_model():
    try:
        logger.info(
            "STEP 1: loading labeled ML training frame from R2 "
            "(lookback_hours=%s)",
            ML_TRAINING_LOOKBACK_HOURS,
        )

        training_df = load_training_frame_from_r2(
            lookahead_minutes=15,
            lookback_hours=ML_TRAINING_LOOKBACK_HOURS,
        )

        logger.info("STEP 2: R2 labeled rows=%s", len(training_df))

        results = train_candidate_models(training_df)
        best_result = results[0]

        for result in results:
            logger.info(
                "MODEL RESULT | name=%s train_rows=%s test_rows=%s "
                "mae=%.4f rmse=%.4f r2=%.4f",
                result["model_name"],
                result["train_rows"],
                result["test_rows"],
                result["mae"],
                result["rmse"],
                result["r2"],
            )

        logger.info("STEP 3: saving best model directly to R2")

        versioned_model_key, latest_model_key = upload_model_to_r2(best_result)

        logger.info(
            "SUCCESS: trained from R2 data and saved model to R2 | "
            "versioned_model=%s latest_model=%s name=%s mae=%.4f rmse=%.4f r2=%.4f",
            versioned_model_key,
            latest_model_key,
            best_result["model_name"],
            best_result["mae"],
            best_result["rmse"],
            best_result["r2"],
        )

    except Exception as e:
        logger.error("FAILED: %s", str(e))
        logger.error(traceback.format_exc())
        raise


with DAG(
    dag_id="7_train_congestion_model",
    default_args=default_args,
    description="Train congestion prediction models using R2 training data and save best model to R2",
    schedule=None,
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "congestion", "model", "r2"],
) as dag:

    run_train_congestion_model = PythonOperator(
        task_id="train_congestion_model_task",
        python_callable=train_congestion_model,
    )