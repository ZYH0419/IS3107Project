from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os
import tempfile
import traceback

import boto3
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from sklearn.neighbors import BallTree

from ml_common import ensure_ml_schema, get_connection, upsert_context_features

logger = logging.getLogger(__name__)

# ================= CONFIG =================

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

R2_ROAD_SEGMENTS_LATEST_KEY = "road_segments/latest/road_segments_latest.parquet"

R2_TAXI_LATEST_KEY = os.environ.get(
    "R2_TAXI_LATEST_KEY",
    "taxi_availability/latest/taxi_availability_latest.parquet",
)

R2_INCIDENTS_LATEST_KEY = os.environ.get(
    "R2_INCIDENTS_LATEST_KEY",
    "traffic_incidents/latest/traffic_incidents_latest.parquet",
)

R2_POI_LATEST_KEY = os.environ.get(
    "R2_POI_LATEST_KEY",
    "poi/latest/poi_latest.parquet",
)

R2_CONTEXT_FEATURES_PREFIX = os.environ.get(
    "R2_CONTEXT_FEATURES_PREFIX",
    "traffic_context_features/",
)

R2_CONTEXT_FEATURES_LATEST_KEY = os.environ.get(
    "R2_CONTEXT_FEATURES_LATEST_KEY",
    "traffic_context_features/latest/context_features.parquet",
)

FEATURE_RADIUS_KM = float(os.environ.get("FEATURE_RADIUS_KM", "0.3"))
EARTH_RADIUS_KM = 6371.0088

WRITE_CONTEXT_FEATURES_HISTORY = (
    os.environ.get("WRITE_CONTEXT_FEATURES_HISTORY", "true")
    .strip()
    .lower()
    == "true"
)

WRITE_CONTEXT_FEATURES_TO_SUPABASE = (
    os.environ.get("WRITE_CONTEXT_FEATURES_TO_SUPABASE", "true")
    .strip()
    .lower()
    == "true"
)

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
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def read_r2_parquet(key: str) -> pd.DataFrame:
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        logger.info("Downloading R2 parquet: bucket=%s key=%s", R2_BUCKET, key)
        client.download_file(R2_BUCKET, key, str(local_path))
        return pd.read_parquet(local_path)


def upload_to_r2(df: pd.DataFrame, key: str) -> None:
    client = get_r2_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        df.to_parquet(local_path, index=False)
        client.upload_file(str(local_path), R2_BUCKET, key)

    logger.info("Uploaded context features to R2: bucket=%s key=%s rows=%s", R2_BUCKET, key, len(df))


def build_context_history_key(feature_timestamp: datetime) -> str:
    file_label = feature_timestamp.strftime("%Y%m%d_%H%M%S")
    date_part = feature_timestamp.strftime("%Y%m%d")
    hour_part = feature_timestamp.strftime("%H")

    prefix = R2_CONTEXT_FEATURES_PREFIX.rstrip("/")

    return (
        f"{prefix}/date={date_part}/hour={hour_part}/"
        f"context_features_{file_label}.parquet"
    )


def normalise_point_df(
    df: pd.DataFrame,
    lat_candidates: list[str],
    lon_candidates: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    lat_col = next((column for column in lat_candidates if column in df.columns), None)
    lon_col = next((column for column in lon_candidates if column in df.columns), None)

    if lat_col is None or lon_col is None:
        logger.warning("Point dataframe missing lat/lon columns. Columns=%s", df.columns.tolist())
        return pd.DataFrame(columns=["latitude", "longitude", "collected_at"])

    out = pd.DataFrame(
        {
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    if "collected_at" in df.columns:
        out["collected_at"] = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")
    else:
        out["collected_at"] = pd.NaT

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out[
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    ].copy()

    return out


def normalise_road_segments_df(segments_df: pd.DataFrame) -> pd.DataFrame:
    segments_df = segments_df.copy()
    segments_df.columns = [str(column).strip().lower() for column in segments_df.columns]

    required_cols = ["link_id", "start_lat", "start_lon", "end_lat", "end_lon"]
    missing_cols = [col for col in required_cols if col not in segments_df.columns]

    if missing_cols:
        raise ValueError(f"Road segments missing required columns: {missing_cols}")

    for col in required_cols:
        segments_df[col] = pd.to_numeric(segments_df[col], errors="coerce")

    segments_df = segments_df.dropna(subset=required_cols).copy()
    segments_df["link_id"] = segments_df["link_id"].astype("int64")

    segments_df["mid_lat"] = (segments_df["start_lat"] + segments_df["end_lat"]) / 2.0
    segments_df["mid_lon"] = (segments_df["start_lon"] + segments_df["end_lon"]) / 2.0

    return segments_df.drop_duplicates("link_id")


def count_points_near_segments(
    segments_df: pd.DataFrame,
    points_df: pd.DataFrame,
    radius_km: float,
) -> np.ndarray:
    if segments_df.empty or points_df.empty:
        return np.zeros(len(segments_df), dtype=float)

    segment_coords_rad = np.radians(
        segments_df[["mid_lat", "mid_lon"]].to_numpy(dtype=float)
    )

    point_coords_rad = np.radians(
        points_df[["latitude", "longitude"]].to_numpy(dtype=float)
    )

    radius_rad = radius_km / EARTH_RADIUS_KM

    tree = BallTree(point_coords_rad, metric="haversine")
    counts = tree.query_radius(segment_coords_rad, r=radius_rad, count_only=True)

    return counts.astype(float)


def max_collected_at(df: pd.DataFrame):
    if df.empty or "collected_at" not in df.columns:
        return None

    ts = pd.to_datetime(df["collected_at"], utc=True, errors="coerce").max()

    if pd.isna(ts):
        return None

    return ts.to_pydatetime()


# ================= TASK =================

def map_r2_context_features_to_road_segments() -> None:
    conn = None

    try:
        feature_timestamp = datetime.now(timezone.utc)

        logger.info("STEP 1: loading road segments from R2")
        segments_raw = read_r2_parquet(R2_ROAD_SEGMENTS_LATEST_KEY)

        if segments_raw.empty:
            logger.warning("No road segments found in R2.")
            return

        segments_df = normalise_road_segments_df(segments_raw)

        logger.info("Road segments ready | rows=%s", len(segments_df))

        logger.info("STEP 2: reading latest taxi, incidents, and POI snapshots from R2")
        taxi_raw = read_r2_parquet(R2_TAXI_LATEST_KEY)
        incidents_raw = read_r2_parquet(R2_INCIDENTS_LATEST_KEY)
        poi_raw = read_r2_parquet(R2_POI_LATEST_KEY)

        taxi_df = normalise_point_df(
            taxi_raw,
            lat_candidates=["latitude", "lat"],
            lon_candidates=["longitude", "lon", "lng"],
        )

        incidents_df = normalise_point_df(
            incidents_raw,
            lat_candidates=["latitude", "lat"],
            lon_candidates=["longitude", "lon", "lng"],
        )

        poi_df = normalise_point_df(
            poi_raw,
            lat_candidates=["centroid_lat", "latitude", "lat"],
            lon_candidates=["centroid_lon", "longitude", "lon", "lng"],
        )

        logger.info(
            "Point data ready | taxi=%s incidents=%s poi=%s",
            len(taxi_df),
            len(incidents_df),
            len(poi_df),
        )

        logger.info("STEP 3: mapping point counts within %.3f km", FEATURE_RADIUS_KM)

        taxi_count = count_points_near_segments(
            segments_df,
            taxi_df,
            FEATURE_RADIUS_KM,
        )

        incident_count = count_points_near_segments(
            segments_df,
            incidents_df,
            FEATURE_RADIUS_KM,
        )

        poi_count = count_points_near_segments(
            segments_df,
            poi_df,
            FEATURE_RADIUS_KM,
        )

        radius_area_km2 = np.pi * (FEATURE_RADIUS_KM ** 2)
        poi_density = poi_count / radius_area_km2 if radius_area_km2 > 0 else poi_count

        features_df = pd.DataFrame(
            {
                "feature_timestamp": feature_timestamp,
                "link_id": segments_df["link_id"].to_numpy(),
                "taxi_count": taxi_count,
                "poi_density": poi_density,
                "traffic_incident_count": incident_count,
            }
        )

        logger.info("STEP 4: saving mapped context features to R2")

        if WRITE_CONTEXT_FEATURES_HISTORY:
            history_key = build_context_history_key(feature_timestamp)
            upload_to_r2(features_df, history_key)
        else:
            logger.info("Skipping historical context feature snapshot because WRITE_CONTEXT_FEATURES_HISTORY=false")

        upload_to_r2(features_df, R2_CONTEXT_FEATURES_LATEST_KEY)

        if WRITE_CONTEXT_FEATURES_TO_SUPABASE:
            logger.info("STEP 5: backing up mapped context features to Supabase")

            try:
                conn = get_connection()
                ensure_ml_schema(conn)

                db_features = features_df.copy()
                db_features["source_taxi_timestamp"] = max_collected_at(taxi_df)
                db_features["source_poi_timestamp"] = max_collected_at(poi_df)
                db_features["source_incident_timestamp"] = max_collected_at(incidents_df)

                affected_rows = upsert_context_features(conn, db_features)

                logger.info(
                    "Supabase context feature backup completed | rows=%s",
                    affected_rows,
                )

            except Exception as db_err:
                logger.warning(
                    "Supabase context feature backup failed, but R2 update succeeded: %s",
                    db_err,
                )

        else:
            logger.info("Skipping Supabase context feature backup because WRITE_CONTEXT_FEATURES_TO_SUPABASE=false")

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()


# ================= DAG =================

with DAG(
    dag_id="6_1_map_r2_context_features",
    default_args=default_args,
    description="Map latest R2 taxi, incident, and POI snapshots to road links and save latest + historical context features.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "features", "r2", "context"],
) as dag:

    run_map_r2_context_features = PythonOperator(
        task_id="map_r2_context_features_to_road_segments",
        python_callable=map_r2_context_features_to_road_segments,
    )