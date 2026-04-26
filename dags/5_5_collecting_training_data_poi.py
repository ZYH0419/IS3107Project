from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import os

import boto3
import pandas as pd
import osmnx as ox

# ================= CONFIG =================

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/opt/airflow/raw_data")

R2_POI_PREFIX = os.environ.get("R2_POI_PREFIX", "poi/")
WRITE_LATEST_POI_ALIAS = os.environ.get("WRITE_LATEST_POI_ALIAS", "true").strip().lower() == "true"

POI_PLACE_QUERY = os.environ.get("POI_PLACE_QUERY", "Singapore")
POI_TAGS = {
    "amenity": True,
    "shop": True,
    "office": True,
    "tourism": True,
    "public_transport": True,
    "railway": ["station"],
}

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
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


def upload_dataframe_to_r2(df: pd.DataFrame, key: str) -> None:
    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_name = Path(key).name
    local_path = output_dir / local_name

    df.to_parquet(local_path, index=False)

    client = get_r2_client()
    client.upload_file(str(local_path), R2_BUCKET, key)

    logger.info("Uploaded dataframe to R2: bucket=%s | key=%s | rows=%s", R2_BUCKET, key, len(df))

    try:
        local_path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Could not delete temporary POI parquet: %s", local_path)


def build_dated_key(prefix: str, file_stem: str, file_label: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    date_part = file_label[:8]
    return f"{prefix}date={date_part}/{file_stem}_{file_label}.parquet"


def fetch_poi_gdf():
    return ox.features.features_from_place(
        query=POI_PLACE_QUERY,
        tags=POI_TAGS,
    )


def build_poi_df(collected_at: datetime) -> pd.DataFrame:
    gdf = fetch_poi_gdf()

    if gdf.empty:
        return pd.DataFrame(columns=[
            "element_type",
            "osm_id",
            "name",
            "amenity",
            "shop",
            "office",
            "tourism",
            "public_transport",
            "railway",
            "geometry_wkt",
            "centroid_lon",
            "centroid_lat",
            "collected_at",
        ])

    gdf = gdf.reset_index()

    if "element" in gdf.columns:
        gdf = gdf.rename(columns={"element": "element_type"})
    if "id" in gdf.columns:
        gdf = gdf.rename(columns={"id": "osm_id"})

    # Force object/string-like columns to string so parquet does not infer mixed types
    for col in [
        "element_type",
        "osm_id",
        "name",
        "amenity",
        "shop",
        "office",
        "tourism",
        "public_transport",
        "railway",
    ]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("string")

    # Save geometry as WKT string
    gdf["geometry_wkt"] = gdf.geometry.to_wkt().astype("string")

    # Better centroid calculation:
    # project to Singapore SVY21 first, calculate centroid, then convert back to WGS84
    try:
        projected = gdf.to_crs(epsg=3414)
        centroids = projected.geometry.centroid.to_crs(epsg=4326)
        gdf["centroid_lon"] = centroids.x
        gdf["centroid_lat"] = centroids.y
    except Exception:
        logger.warning("Could not project geometry before centroid; falling back to raw centroid.")
        centroids = gdf.geometry.centroid
        gdf["centroid_lon"] = centroids.x
        gdf["centroid_lat"] = centroids.y

    gdf["centroid_lon"] = pd.to_numeric(gdf["centroid_lon"], errors="coerce")
    gdf["centroid_lat"] = pd.to_numeric(gdf["centroid_lat"], errors="coerce")
    gdf["collected_at"] = pd.to_datetime(collected_at, utc=True)

    desired_cols = [
        "element_type",
        "osm_id",
        "name",
        "amenity",
        "shop",
        "office",
        "tourism",
        "public_transport",
        "railway",
        "geometry_wkt",
        "centroid_lon",
        "centroid_lat",
        "collected_at",
    ]
    existing_cols = [c for c in desired_cols if c in gdf.columns]

    poi_df = gdf[existing_cols].copy()

    for col in poi_df.select_dtypes(include=["object"]).columns:
        poi_df[col] = poi_df[col].astype("string")

    return poi_df

# ================= TASK =================

def collect_and_upload_poi_training_data():
    collected_at = datetime.now(timezone.utc)
    file_label = collected_at.strftime("%Y%m%d_%H%M%S")

    logger.info("STEP 1: fetching POI data from OpenStreetMap for place=%s", POI_PLACE_QUERY)
    poi_df = build_poi_df(collected_at=collected_at)

    logger.info("POI training data prepared | rows=%s", len(poi_df))

    if not poi_df.empty:
        poi_key = build_dated_key(
            R2_POI_PREFIX,
            "poi_snapshot",
            file_label,
        )
        upload_dataframe_to_r2(poi_df, poi_key)

        if WRITE_LATEST_POI_ALIAS:
            latest_poi_key = "poi/latest/poi_latest.parquet"
            upload_dataframe_to_r2(poi_df, latest_poi_key)
    else:
        logger.warning("No POI rows found; skipping POI upload")

# ================= DAG =================

with DAG(
    dag_id="5_5_collecting_training_data_poi",
    default_args=default_args,
    description="Collect OpenStreetMap POI data and save to Cloudflare R2 as parquet training data.",
    schedule="@daily",
    start_date=datetime(2026, 4, 21),
    catchup=False,
    max_active_runs=1,
    tags=["osm", "poi", "training-data", "r2"],
) as dag:

    task_collect_and_upload_poi_training_data = PythonOperator(
        task_id="collect_and_upload_poi_training_data",
        python_callable=collect_and_upload_poi_training_data,
    )