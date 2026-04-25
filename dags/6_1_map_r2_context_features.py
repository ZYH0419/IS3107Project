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
from psycopg2.extras import execute_values
from sklearn.neighbors import BallTree

from ml_common import ensure_ml_schema, get_connection

logger = logging.getLogger(__name__)

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

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

FEATURE_RADIUS_KM = float(os.environ.get("FEATURE_RADIUS_KM", "0.3"))
EARTH_RADIUS_KM = 6371.0088

# Keep this reasonably large because this task maps all road links to three point datasets.
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


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


def load_road_segments(conn) -> pd.DataFrame:
    query = """
        SELECT
            link_id,
            start_lat,
            start_lon,
            end_lat,
            end_lon
        FROM road_segments
        WHERE start_lat IS NOT NULL
          AND start_lon IS NOT NULL
          AND end_lat IS NOT NULL
          AND end_lon IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    segments_df = pd.DataFrame(rows, columns=columns)
    if segments_df.empty:
        return segments_df

    for column in ["link_id", "start_lat", "start_lon", "end_lat", "end_lon"]:
        segments_df[column] = pd.to_numeric(segments_df[column], errors="coerce")
    segments_df = segments_df.dropna(subset=["link_id", "start_lat", "start_lon", "end_lat", "end_lon"]).copy()
    segments_df["link_id"] = segments_df["link_id"].astype("int64")
    segments_df["mid_lat"] = (segments_df["start_lat"] + segments_df["end_lat"]) / 2.0
    segments_df["mid_lon"] = (segments_df["start_lon"] + segments_df["end_lon"]) / 2.0
    return segments_df


def normalise_point_df(df: pd.DataFrame, lat_candidates: list[str], lon_candidates: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    lat_col = next((column for column in lat_candidates if column in df.columns), None)
    lon_col = next((column for column in lon_candidates if column in df.columns), None)
    if lat_col is None or lon_col is None:
        logger.warning("No usable lat/lon columns found. columns=%s", list(df.columns))
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
    out = out[out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)].copy()
    return out


def count_points_near_segments(segments_df: pd.DataFrame, points_df: pd.DataFrame, radius_km: float) -> np.ndarray:
    if segments_df.empty or points_df.empty:
        return np.zeros(len(segments_df), dtype=float)

    segment_coords_rad = np.radians(segments_df[["mid_lat", "mid_lon"]].to_numpy(dtype=float))
    point_coords_rad = np.radians(points_df[["latitude", "longitude"]].to_numpy(dtype=float))
    radius_rad = radius_km / EARTH_RADIUS_KM

    tree = BallTree(point_coords_rad, metric="haversine")
    counts = tree.query_radius(segment_coords_rad, r=radius_rad, count_only=True)
    return counts.astype(float)


def upsert_context_features(conn, features_df: pd.DataFrame, batch_size: int = 1000) -> int:
    if features_df.empty:
        return 0

    rows = list(features_df.itertuples(index=False, name=None))
    sql = """
        INSERT INTO traffic_context_features (
            feature_timestamp,
            link_id,
            taxi_count,
            poi_density,
            traffic_incident_count,
            source_taxi_timestamp,
            source_poi_timestamp,
            source_incident_timestamp
        )
        VALUES %s
        ON CONFLICT (feature_timestamp, link_id) DO UPDATE SET
            taxi_count = EXCLUDED.taxi_count,
            poi_density = EXCLUDED.poi_density,
            traffic_incident_count = EXCLUDED.traffic_incident_count,
            source_taxi_timestamp = EXCLUDED.source_taxi_timestamp,
            source_poi_timestamp = EXCLUDED.source_poi_timestamp,
            source_incident_timestamp = EXCLUDED.source_incident_timestamp,
            inserted_at = now()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=batch_size)
    conn.commit()
    return len(rows)


def map_r2_context_features_to_road_segments() -> None:
    conn = None
    try:
        feature_timestamp = datetime.now(timezone.utc)
        logger.info("STEP 1: loading road segments from Supabase")
        conn = get_connection()
        ensure_ml_schema(conn)
        segments_df = load_road_segments(conn)
        if segments_df.empty:
            logger.warning("No road segments found. Run 1_load_road_segments first.")
            return
        logger.info("Loaded road segments: %s", len(segments_df))

        logger.info("STEP 2: reading latest taxi, incidents, and POI snapshots from R2")
        taxi_raw = read_r2_parquet(R2_TAXI_LATEST_KEY)
        incidents_raw = read_r2_parquet(R2_INCIDENTS_LATEST_KEY)
        poi_raw = read_r2_parquet(R2_POI_LATEST_KEY)

        taxi_df = normalise_point_df(taxi_raw, ["latitude", "lat"], ["longitude", "lon", "lng"])
        incidents_df = normalise_point_df(incidents_raw, ["latitude", "lat"], ["longitude", "lon", "lng"])
        poi_df = normalise_point_df(poi_raw, ["centroid_lat", "latitude", "lat"], ["centroid_lon", "longitude", "lon", "lng"])

        logger.info(
            "Normalised point rows | taxi=%s incidents=%s poi=%s",
            len(taxi_df),
            len(incidents_df),
            len(poi_df),
        )

        logger.info("STEP 3: mapping point counts to road-link midpoints within %.3f km", FEATURE_RADIUS_KM)
        taxi_count = count_points_near_segments(segments_df, taxi_df, FEATURE_RADIUS_KM)
        incident_count = count_points_near_segments(segments_df, incidents_df, FEATURE_RADIUS_KM)
        poi_count = count_points_near_segments(segments_df, poi_df, FEATURE_RADIUS_KM)

        radius_area_km2 = np.pi * (FEATURE_RADIUS_KM ** 2)
        poi_density = poi_count / radius_area_km2 if radius_area_km2 > 0 else poi_count

        source_taxi_ts = taxi_df["collected_at"].max() if not taxi_df.empty else pd.NaT
        source_incident_ts = incidents_df["collected_at"].max() if not incidents_df.empty else pd.NaT
        source_poi_ts = poi_df["collected_at"].max() if not poi_df.empty else pd.NaT

        features_df = pd.DataFrame(
            {
                "feature_timestamp": feature_timestamp,
                "link_id": segments_df["link_id"].astype("int64"),
                "taxi_count": taxi_count,
                "poi_density": poi_density,
                "traffic_incident_count": incident_count,
                "source_taxi_timestamp": source_taxi_ts.to_pydatetime() if pd.notna(source_taxi_ts) else None,
                "source_poi_timestamp": source_poi_ts.to_pydatetime() if pd.notna(source_poi_ts) else None,
                "source_incident_timestamp": source_incident_ts.to_pydatetime() if pd.notna(source_incident_ts) else None,
            }
        )

        logger.info("STEP 4: upserting mapped context features into traffic_context_features")
        inserted_rows = upsert_context_features(conn, features_df)
        logger.info(
            "SUCCESS: mapped/upserted context features | rows=%s radius_km=%.3f taxi_total=%s incident_total=%s poi_density_avg=%.4f",
            inserted_rows,
            FEATURE_RADIUS_KM,
            int(taxi_count.sum()),
            int(incident_count.sum()),
            float(np.nanmean(poi_density)) if len(poi_density) else 0.0,
        )
    except Exception as exc:
        logger.error("FAILED: %s", str(exc))
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn is not None:
            conn.close()


with DAG(
    dag_id="6_1_map_r2_context_features",
    default_args=default_args,
    description="Map latest R2 taxi, POI, and incident snapshots to road segment link_id features for ML.",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "features", "r2", "taxi", "poi", "incidents"],
) as dag:
    run_map_r2_context_features = PythonOperator(
        task_id="map_r2_context_features_to_road_segments",
        python_callable=map_r2_context_features_to_road_segments,
    )
