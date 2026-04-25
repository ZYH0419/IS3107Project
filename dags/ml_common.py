import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from lta_common import get_db_dsn

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "link_id",
    "road_category",
    "current_speed_band",
    "minimum_speed",
    "maximum_speed",
    "avg_speed",
    "rainfall_mm",
    "station_distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

TARGET_COLUMN = "future_congestion_score_15min"


def get_connection():
    # Keep ML DAG failures fast when Supabase/network is temporarily unreachable.
    return psycopg2.connect(get_db_dsn(), connect_timeout=20)


def ensure_ml_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_rainfall_training_data (
                collected_at timestamptz NOT NULL,
                link_id bigint NOT NULL,
                road_name text,
                road_category integer,
                speed_band integer,
                minimum_speed integer,
                maximum_speed integer,
                avg_speed double precision,
                congestion_score double precision,
                rainfall_mm double precision,
                station_id text,
                station_name text,
                station_distance_km double precision,
                rainfall_timestamp timestamptz,
                hour_of_day integer,
                day_of_week integer,
                is_weekend boolean,
                inserted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (collected_at, link_id)
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_data_link_time
            ON traffic_rainfall_training_data (link_id, collected_at)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_data_collected_at
            ON traffic_rainfall_training_data (collected_at)
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS congestion_model_registry (
                model_id bigserial PRIMARY KEY,
                model_name text NOT NULL,
                model_version text NOT NULL,
                target_name text NOT NULL,
                training_started_at timestamptz NOT NULL,
                training_finished_at timestamptz NOT NULL,
                train_rows integer NOT NULL,
                test_rows integer NOT NULL,
                mae double precision,
                rmse double precision,
                r2 double precision,
                feature_columns text[] NOT NULL,
                artifact bytea NOT NULL,
                is_active boolean NOT NULL DEFAULT false,
                notes text
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS congestion_predictions (
                prediction_created_at timestamptz NOT NULL,
                target_time timestamptz NOT NULL,
                link_id bigint NOT NULL,
                road_name text,
                road_category integer,
                current_speed_band integer,
                current_congestion_score double precision,
                rainfall_mm double precision,
                predicted_congestion_score double precision,
                predicted_speed_band double precision,
                model_id bigint REFERENCES congestion_model_registry(model_id),
                model_name text,
                PRIMARY KEY (target_time, link_id, model_id)
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_congestion_predictions_created
            ON congestion_predictions (prediction_created_at DESC)
            """
        )

    conn.commit()


def collect_training_snapshot(conn) -> int:
    ensure_ml_schema(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO traffic_rainfall_training_data (
                collected_at,
                link_id,
                road_name,
                road_category,
                speed_band,
                minimum_speed,
                maximum_speed,
                avg_speed,
                congestion_score,
                rainfall_mm,
                station_id,
                station_name,
                station_distance_km,
                rainfall_timestamp,
                hour_of_day,
                day_of_week,
                is_weekend
            )
            SELECT
                collected_at,
                link_id,
                road_name,
                road_category,
                speed_band,
                minimum_speed,
                maximum_speed,
                (minimum_speed + maximum_speed) / 2.0 AS avg_speed,
                9 - speed_band AS congestion_score,
                rainfall_mm,
                station_id,
                station_name,
                station_distance_km,
                rainfall_timestamp,
                EXTRACT(HOUR FROM collected_at)::integer AS hour_of_day,
                EXTRACT(DOW FROM collected_at)::integer AS day_of_week,
                EXTRACT(DOW FROM collected_at)::integer IN (0, 6) AS is_weekend
            FROM traffic_rainfall_latest
            WHERE speed_band IS NOT NULL
            ON CONFLICT (collected_at, link_id) DO UPDATE SET
                road_name = EXCLUDED.road_name,
                road_category = EXCLUDED.road_category,
                speed_band = EXCLUDED.speed_band,
                minimum_speed = EXCLUDED.minimum_speed,
                maximum_speed = EXCLUDED.maximum_speed,
                avg_speed = EXCLUDED.avg_speed,
                congestion_score = EXCLUDED.congestion_score,
                rainfall_mm = EXCLUDED.rainfall_mm,
                station_id = EXCLUDED.station_id,
                station_name = EXCLUDED.station_name,
                station_distance_km = EXCLUDED.station_distance_km,
                rainfall_timestamp = EXCLUDED.rainfall_timestamp,
                hour_of_day = EXCLUDED.hour_of_day,
                day_of_week = EXCLUDED.day_of_week,
                is_weekend = EXCLUDED.is_weekend,
                inserted_at = now()
            """
        )
        affected_rows = cur.rowcount

    conn.commit()
    return affected_rows


def load_training_frame(
    conn,
    lookahead_minutes: int = 15,
    lookback_hours: int | None = None,
) -> pd.DataFrame:
    ensure_ml_schema(conn)

    lookback_filter = ""
    if lookback_hours is not None:
        lookback_filter = (
            f"AND collected_at >= now() - interval '{int(lookback_hours)} hours'"
        )

    query = f"""
        WITH recent_current AS (
            SELECT
                collected_at,
                link_id,
                road_name,
                road_category,
                speed_band,
                minimum_speed,
                maximum_speed,
                avg_speed,
                congestion_score,
                rainfall_mm,
                station_distance_km,
                hour_of_day,
                day_of_week,
                is_weekend
            FROM traffic_rainfall_training_data
            WHERE speed_band IS NOT NULL
            {lookback_filter}
        ),
        labeled AS (
            SELECT
                current.collected_at,
                current.link_id,
                current.road_name,
                current.road_category,
                current.speed_band AS current_speed_band,
                current.minimum_speed,
                current.maximum_speed,
                current.avg_speed,
                current.congestion_score AS current_congestion_score,
                COALESCE(current.rainfall_mm, 0) AS rainfall_mm,
                COALESCE(current.station_distance_km, 0) AS station_distance_km,
                current.hour_of_day,
                current.day_of_week,
                current.is_weekend,
                future.collected_at AS future_collected_at,
                future.congestion_score AS {TARGET_COLUMN}
            FROM recent_current AS current
            JOIN LATERAL (
                SELECT collected_at, congestion_score
                FROM traffic_rainfall_training_data AS future
                WHERE future.link_id = current.link_id
                  AND future.collected_at >= current.collected_at + interval '{lookahead_minutes - 5} minutes'
                  AND future.collected_at <= current.collected_at + interval '{lookahead_minutes + 5} minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (
                    future.collected_at - (current.collected_at + interval '{lookahead_minutes} minutes')
                )))
                LIMIT 1
            ) AS future ON TRUE
            WHERE future.congestion_score IS NOT NULL
        )
        SELECT *
        FROM labeled
        ORDER BY collected_at, link_id
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean_df = df.copy()
    clean_df["is_weekend"] = clean_df["is_weekend"].astype(int)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.dropna(subset=[TARGET_COLUMN]).copy()
    clean_df[FEATURE_COLUMNS] = clean_df[FEATURE_COLUMNS].fillna(0)

    return clean_df[FEATURE_COLUMNS], clean_df[TARGET_COLUMN]


def train_candidate_models(df: pd.DataFrame) -> list[dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = prepare_features(df)
    if len(X) < 1000:
        raise ValueError(f"Not enough labeled training rows yet: {len(X)}. Collect more data first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    candidates: list[tuple[str, Any, str]] = [
        (
            "linear_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            "Baseline linear regression.",
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
            "Tree baseline suitable for nonlinear tabular patterns.",
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            ),
            "Sklearn gradient boosting model.",
        ),
    ]

    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
                "Optional XGBoost model. Used only when xgboost is installed.",
            )
        )
    except ImportError:
        logger.info("xgboost is not installed; skipping XGBoost candidate")

    results = []
    for model_name, model, notes in candidates:
        logger.info("Training %s", model_name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append(
            {
                "model_name": model_name,
                "model": model,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
                "r2": float(r2_score(y_test, predictions)),
                "notes": notes,
            }
        )

    return sorted(results, key=lambda result: result["mae"])


def save_best_model(conn, result: dict[str, Any]) -> int:
    ensure_ml_schema(conn)

    finished_at = datetime.now(timezone.utc)
    artifact = pickle.dumps(
        {
            "model": result["model"],
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        }
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE congestion_model_registry
            SET is_active = false
            """
        )
        cur.execute(
            """
            INSERT INTO congestion_model_registry (
                model_name,
                model_version,
                target_name,
                training_started_at,
                training_finished_at,
                train_rows,
                test_rows,
                mae,
                rmse,
                r2,
                feature_columns,
                artifact,
                is_active,
                notes
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s
            )
            RETURNING model_id
            """,
            (
                result["model_name"],
                finished_at.strftime("%Y%m%d_%H%M%S"),
                TARGET_COLUMN,
                finished_at,
                finished_at,
                result["train_rows"],
                result["test_rows"],
                result["mae"],
                result["rmse"],
                result["r2"],
                FEATURE_COLUMNS,
                artifact,
                result["notes"],
            ),
        )
        model_id = cur.fetchone()[0]

    conn.commit()
    return int(model_id)


def load_active_model(conn) -> dict[str, Any]:
    ensure_ml_schema(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_id, model_name, artifact
            FROM congestion_model_registry
            WHERE is_active = true
            ORDER BY training_finished_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError("No active congestion model found. Train a model first.")

    model_id, model_name, artifact = row
    payload = pickle.loads(bytes(artifact))
    payload["model_id"] = model_id
    payload["model_name"] = model_name
    return payload


def load_latest_prediction_frame(conn) -> pd.DataFrame:
    query = """
        SELECT
            collected_at,
            link_id,
            road_name,
            road_category,
            speed_band AS current_speed_band,
            minimum_speed,
            maximum_speed,
            (minimum_speed + maximum_speed) / 2.0 AS avg_speed,
            9 - speed_band AS current_congestion_score,
            COALESCE(rainfall_mm, 0) AS rainfall_mm,
            COALESCE(station_distance_km, 0) AS station_distance_km,
            EXTRACT(HOUR FROM collected_at)::integer AS hour_of_day,
            EXTRACT(DOW FROM collected_at)::integer AS day_of_week,
            EXTRACT(DOW FROM collected_at)::integer IN (0, 6) AS is_weekend
        FROM traffic_rainfall_latest
        WHERE speed_band IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def save_predictions(conn, prediction_df: pd.DataFrame, model_id: int, model_name: str, lookahead_minutes: int = 15) -> int:
    if prediction_df.empty:
        return 0

    prediction_created_at = datetime.now(timezone.utc)
    rows = []
    for row in prediction_df.itertuples(index=False):
        target_time = pd.Timestamp(row.collected_at).to_pydatetime() + pd.Timedelta(minutes=lookahead_minutes)
        predicted_congestion = max(1.0, min(8.0, float(row.predicted_congestion_score)))
        predicted_speed_band = max(1.0, min(8.0, 9 - predicted_congestion))
        rows.append(
            (
                prediction_created_at,
                target_time,
                int(row.link_id),
                row.road_name,
                int(row.road_category) if pd.notna(row.road_category) else None,
                int(row.current_speed_band) if pd.notna(row.current_speed_band) else None,
                float(row.current_congestion_score) if pd.notna(row.current_congestion_score) else None,
                float(row.rainfall_mm) if pd.notna(row.rainfall_mm) else None,
                predicted_congestion,
                predicted_speed_band,
                int(model_id),
                model_name,
            )
        )

    sql = """
        INSERT INTO congestion_predictions (
            prediction_created_at,
            target_time,
            link_id,
            road_name,
            road_category,
            current_speed_band,
            current_congestion_score,
            rainfall_mm,
            predicted_congestion_score,
            predicted_speed_band,
            model_id,
            model_name
        )
        VALUES %s
        ON CONFLICT (target_time, link_id, model_id) DO UPDATE SET
            prediction_created_at = EXCLUDED.prediction_created_at,
            road_name = EXCLUDED.road_name,
            road_category = EXCLUDED.road_category,
            current_speed_band = EXCLUDED.current_speed_band,
            current_congestion_score = EXCLUDED.current_congestion_score,
            rainfall_mm = EXCLUDED.rainfall_mm,
            predicted_congestion_score = EXCLUDED.predicted_congestion_score,
            predicted_speed_band = EXCLUDED.predicted_speed_band,
            model_name = EXCLUDED.model_name
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)

    conn.commit()
    return len(rows)
