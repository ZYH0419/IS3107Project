from dotenv import load_dotenv
load_dotenv()

import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from ml_common import train_candidate_models


# ================= LOGGING SETUP =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ================= CONFIG =================

TRAINING_DATA_PATH = Path("combined_training_df.parquet")
LOCAL_MODEL_OUTPUT_PATH = Path("local_best_congestion_model.joblib")


# ================= MAIN =================

def train_local() -> None:
    start_time = time.perf_counter()

    try:
        logger.info("STEP 1: loading local training dataset")
        logger.info("Training data path: %s", TRAINING_DATA_PATH)

        if not TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training data file not found: {TRAINING_DATA_PATH}. "
                "Run export_training_df.py first."
            )

        df = pd.read_parquet(TRAINING_DATA_PATH)

        logger.info("Loaded training dataframe shape=%s", df.shape)
        logger.info("Columns=%s", list(df.columns))

        logger.info("STEP 2: training candidate models")
        logger.info(
            "The training function will evaluate all available candidate models "
            "and return them sorted by MAE."
        )

        training_start = time.perf_counter()
        results = train_candidate_models(df)
        training_seconds = time.perf_counter() - training_start

        if not results:
            raise ValueError("No model results were returned.")

        logger.info("STEP 3: evaluation results for each model")

        for index, result in enumerate(results, start=1):
            logger.info(
                "MODEL %s | name=%s | train_rows=%s | test_rows=%s | "
                "MAE=%.4f | RMSE=%.4f | R2=%.4f | notes=%s",
                index,
                result["model_name"],
                result["train_rows"],
                result["test_rows"],
                result["mae"],
                result["rmse"],
                result["r2"],
                result.get("notes", ""),
            )

        best = results[0]

        logger.info("STEP 4: best model selected")
        logger.info(
            "BEST MODEL | name=%s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            best["model_name"],
            best["mae"],
            best["rmse"],
            best["r2"],
        )

        logger.info("STEP 5: saving best model locally")
        joblib.dump(best["model"], LOCAL_MODEL_OUTPUT_PATH)

        total_seconds = time.perf_counter() - start_time

        logger.info("SUCCESS: saved best model to %s", LOCAL_MODEL_OUTPUT_PATH)
        logger.info("Training runtime seconds=%.2f", training_seconds)
        logger.info("Total runtime seconds=%.2f", total_seconds)

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.exception("Full traceback")
        raise


if __name__ == "__main__":
    train_local()
