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

R2_CONTEXT_FEATURES_WORK_PREFIX = os.environ.get(
    "R2_CONTEXT_FEATURES_WORK_PREFIX",
    "traffic_context_features/work/",
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

DEFAULT_ARGS = {
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

    logger.info("Uploaded parquet to R2: bucket=%s key=%s rows=%s", R2_BUCKET, key, len(df))


def build_context_history_key(feature_timestamp: datetime) -> str:
    file_label = feature_timestamp.strftime("%Y%m%d_%H%M%S")
    date_part = feature_timestamp.strftime("%Y%m%d")
    hour_part = feature_timestamp.strftime("%H")
    prefix = R2_CONTEXT_FEATURES_PREFIX.rstrip("/")
    return f"{prefix}/date={date_part}/hour={hour_part}/context_features_{file_label}.parquet"


def build_work_key(run_id: str, name: str) -> str:
    safe_run_id = (
        run_id.replace(":", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace(".", "_")
    )
    prefix = R2_CONTEXT_FEATURES_WORK_PREFIX.rstrip("/")
    return f"{prefix}/{safe_run_id}/{name}.parquet"


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


def build_count_frame(
    segments_df: pd.DataFrame,
    points_df: pd.DataFrame,
    count_column: str,
) -> pd.DataFrame:
    counts = count_points_near_segments(segments_df, points_df, FEATURE_RADIUS_KM)
    out = pd.DataFrame(
        {
            "link_id": segments_df["link_id"].to_numpy(),
            count_column: counts,
        }
    )
    return out

# ================= AIRFLOW TASKS =================

def prepare_road_segments(**context) -> dict:
    run_id = context["run_id"]
    feature_timestamp = datetime.now(timezone.utc)

    logger.info("STEP 1: loading and normalising road segments")
    segments_raw = read_r2_parquet(R2_ROAD_SEGMENTS_LATEST_KEY)

    if segments_raw.empty:
        raise ValueError("No road segments found in R2.")

    segments_df = normalise_road_segments_df(segments_raw)
    segments_key = build_work_key(run_id, "road_segments_normalised")
    upload_to_r2(segments_df, segments_key)

    return {
        "feature_timestamp": feature_timestamp.isoformat(),
        "segments_key": segments_key,
    }


def map_taxi_to_segments(**context) -> dict:
    run_id = context["run_id"]
    ti = context["ti"]
    prepared = ti.xcom_pull(task_ids="prepare_road_segments")

    segments_df = read_r2_parquet(prepared["segments_key"])
    taxi_raw = read_r2_parquet(R2_TAXI_LATEST_KEY)
    taxi_df = normalise_point_df(
        taxi_raw,
        lat_candidates=["latitude", "lat"],
        lon_candidates=["longitude", "lon", "lng"],
    )

    taxi_counts_df = build_count_frame(segments_df, taxi_df, "taxi_count")
    taxi_counts_key = build_work_key(run_id, "taxi_counts")
    upload_to_r2(taxi_counts_df, taxi_counts_key)

    return {
        "taxi_counts_key": taxi_counts_key,
        "source_taxi_timestamp": max_collected_at(taxi_df).isoformat() if max_collected_at(taxi_df) else None,
    }


def map_incidents_to_segments(**context) -> dict:
    run_id = context["run_id"]
    ti = context["ti"]
    prepared = ti.xcom_pull(task_ids="prepare_road_segments")

    segments_df = read_r2_parquet(prepared["segments_key"])
    incidents_raw = read_r2_parquet(R2_INCIDENTS_LATEST_KEY)
    incidents_df = normalise_point_df(
        incidents_raw,
        lat_candidates=["latitude", "lat"],
        lon_candidates=["longitude", "lon", "lng"],
    )

    incident_counts_df = build_count_frame(segments_df, incidents_df, "traffic_incident_count")
    incident_counts_key = build_work_key(run_id, "incident_counts")
    upload_to_r2(incident_counts_df, incident_counts_key)

    return {
        "incident_counts_key": incident_counts_key,
        "source_incident_timestamp": max_collected_at(incidents_df).isoformat() if max_collected_at(incidents_df) else None,
    }


def map_poi_to_segments(**context) -> dict:
    run_id = context["run_id"]
    ti = context["ti"]
    prepared = ti.xcom_pull(task_ids="prepare_road_segments")

    segments_df = read_r2_parquet(prepared["segments_key"])
    poi_raw = read_r2_parquet(R2_POI_LATEST_KEY)
    poi_df = normalise_point_df(
        poi_raw,
        lat_candidates=["centroid_lat", "latitude", "lat"],
        lon_candidates=["centroid_lon", "longitude", "lon", "lng"],
    )

    poi_counts_df = build_count_frame(segments_df, poi_df, "poi_count")
    radius_area_km2 = np.pi * (FEATURE_RADIUS_KM ** 2)
    poi_counts_df["poi_density"] = (
        poi_counts_df["poi_count"] / radius_area_km2
        if radius_area_km2 > 0
        else poi_counts_df["poi_count"]
    )
    poi_counts_df = poi_counts_df[["link_id", "poi_density"]]

    poi_density_key = build_work_key(context["run_id"], "poi_density")
    upload_to_r2(poi_counts_df, poi_density_key)

    return {
        "poi_density_key": poi_density_key,
        "source_poi_timestamp": max_collected_at(poi_df).isoformat() if max_collected_at(poi_df) else None,
    }


def combine_and_save_context_features(**context) -> dict:
    ti = context["ti"]

    prepared = ti.xcom_pull(task_ids="prepare_road_segments")
    taxi_result = ti.xcom_pull(task_ids="map_taxi_to_segments")
    incident_result = ti.xcom_pull(task_ids="map_incidents_to_segments")
    poi_result = ti.xcom_pull(task_ids="map_poi_to_segments")

    feature_timestamp = datetime.fromisoformat(prepared["feature_timestamp"])

    segments_df = read_r2_parquet(prepared["segments_key"])
    taxi_counts_df = read_r2_parquet(taxi_result["taxi_counts_key"])
    incident_counts_df = read_r2_parquet(incident_result["incident_counts_key"])
    poi_density_df = read_r2_parquet(poi_result["poi_density_key"])

    features_df = pd.DataFrame(
        {
            "feature_timestamp": feature_timestamp,
            "link_id": segments_df["link_id"].to_numpy(),
        }
    )

    features_df = features_df.merge(taxi_counts_df, on="link_id", how="left")
    features_df = features_df.merge(incident_counts_df, on="link_id", how="left")
    features_df = features_df.merge(poi_density_df, on="link_id", how="left")

    for col in ["taxi_count", "traffic_incident_count", "poi_density"]:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

    if WRITE_CONTEXT_FEATURES_HISTORY:
        history_key = build_context_history_key(feature_timestamp)
        upload_to_r2(features_df, history_key)
    else:
        history_key = None
        logger.info("Skipping historical context feature snapshot because WRITE_CONTEXT_FEATURES_HISTORY=false")

    upload_to_r2(features_df, R2_CONTEXT_FEATURES_LATEST_KEY)

    return {
        "history_key": history_key,
        "latest_key": R2_CONTEXT_FEATURES_LATEST_KEY,
        "feature_timestamp": feature_timestamp.isoformat(),
        "source_taxi_timestamp": taxi_result.get("source_taxi_timestamp"),
        "source_poi_timestamp": poi_result.get("source_poi_timestamp"),
        "source_incident_timestamp": incident_result.get("source_incident_timestamp"),
    }


def backup_context_features_to_supabase(**context) -> None:
    if not WRITE_CONTEXT_FEATURES_TO_SUPABASE:
        logger.info("Skipping Supabase context feature backup because WRITE_CONTEXT_FEATURES_TO_SUPABASE=false")
        return

    conn = None

    try:
        ti = context["ti"]
        saved = ti.xcom_pull(task_ids="combine_and_save_context_features")

        features_df = read_r2_parquet(saved["latest_key"])
        features_df["source_taxi_timestamp"] = pd.to_datetime(saved.get("source_taxi_timestamp"), utc=True, errors="coerce")
        features_df["source_poi_timestamp"] = pd.to_datetime(saved.get("source_poi_timestamp"), utc=True, errors="coerce")
        features_df["source_incident_timestamp"] = pd.to_datetime(saved.get("source_incident_timestamp"), utc=True, errors="coerce")

        conn = get_connection()
        ensure_ml_schema(conn)
        affected_rows = upsert_context_features(conn, features_df)

        logger.info("Supabase context feature backup completed | rows=%s", affected_rows)

    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn is not None:
            conn.close()

# ================= DAG =================

with DAG(
    dag_id="6_feature_engineering",
    default_args=DEFAULT_ARGS,
    description="Map latest R2 taxi, incident, and POI snapshots to road links and save latest plus historical context features.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "features", "r2", "context"],
) as dag:

    task_prepare_road_segments = PythonOperator(
        task_id="prepare_road_segments",
        python_callable=prepare_road_segments,
    )

    task_map_taxi = PythonOperator(
        task_id="map_taxi_to_segments",
        python_callable=map_taxi_to_segments,
    )

    task_map_incidents = PythonOperator(
        task_id="map_incidents_to_segments",
        python_callable=map_incidents_to_segments,
    )

    task_map_poi = PythonOperator(
        task_id="map_poi_to_segments",
        python_callable=map_poi_to_segments,
    )

    task_combine_and_save = PythonOperator(
        task_id="combine_and_save_context_features",
        python_callable=combine_and_save_context_features,
    )

    task_backup_to_supabase = PythonOperator(
        task_id="backup_context_features_to_supabase",
        python_callable=backup_context_features_to_supabase,
    )

    task_prepare_road_segments >> [task_map_taxi, task_map_incidents, task_map_poi]
    [task_map_taxi, task_map_incidents, task_map_poi] >> task_combine_and_save
    task_combine_and_save >> task_backup_to_supabase
