from __future__ import annotations

import logging
import os
import pickle
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.neighbors import BallTree

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
    "taxi_count",
    "poi_density",
    "traffic_incident_count",

    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

TARGET_COLUMN = "future_congestion_score_15min"

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# SUPABASE HELPERS
# ============================================================

def get_connection():
    return psycopg2.connect(get_db_dsn(), connect_timeout=20)


# ============================================================
# R2 HELPERS
# ============================================================

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
    )


def read_r2_parquet(key: str) -> pd.DataFrame:
    client = get_r2_client()
    bucket = os.environ["R2_BUCKET"]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(key).name
        logger.info("Downloading R2 parquet: bucket=%s key=%s", bucket, key)
        client.download_file(bucket, key, str(local_path))
        return pd.read_parquet(local_path)


def list_r2_parquet_keys(prefix: str) -> list[str]:
    client = get_r2_client()
    bucket = os.environ["R2_BUCKET"]

    keys: list[str] = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": bucket,
            "Prefix": prefix,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

    return sorted(keys)


def get_latest_r2_key(prefix: str) -> str | None:
    keys = list_r2_parquet_keys(prefix)
    return sorted(keys)[-1] if keys else None


def extract_timestamp_from_key(key: str) -> datetime | None:
    match = re.search(r"(\d{8}_\d{6})", key)
    if not match:
        return None

    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def filter_keys_by_lookback(keys: list[str], lookback_hours: int | None) -> list[str]:
    if lookback_hours is None:
        return keys

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - pd.Timedelta(hours=lookback_hours)

    selected = []
    for key in keys:
        ts = extract_timestamp_from_key(key)
        if ts is not None and ts >= cutoff:
            selected.append(key)

    return selected


# ============================================================
# R2 TRAFFIC LOADING
# ============================================================

def load_traffic_history_from_r2(
    lookback_hours: int | None = None,
    traffic_prefix: str = "traffic_speed/",
) -> pd.DataFrame:
    keys = list_r2_parquet_keys(traffic_prefix)
    keys = [
        key for key in keys
        if "/latest/" not in key and "traffic_speed_snapshot_" in key
    ]
    keys = filter_keys_by_lookback(keys, lookback_hours)

    if not keys:
        logger.warning("No traffic parquet files found in R2 under prefix=%s", traffic_prefix)
        return pd.DataFrame()

    frames = []

    for key in keys:
        collected_at = extract_timestamp_from_key(key)
        if collected_at is None:
            logger.warning("Could not parse collected_at from key, skipping: %s", key)
            continue

        try:
            df = read_r2_parquet(key)

            if df.empty:
                continue

            df = df.copy()
            df["collected_at"] = pd.Timestamp(collected_at)
            frames.append(df)

        except Exception as exc:
            logger.warning("Failed to read traffic parquet key=%s error=%s", key, exc)

    if not frames:
        return pd.DataFrame()

    traffic_df = pd.concat(frames, ignore_index=True)

    required_numeric_cols = [
        "link_id",
        "road_category",
        "speed_band",
        "minimum_speed",
        "maximum_speed",
    ]

    for col in required_numeric_cols:
        if col in traffic_df.columns:
            traffic_df[col] = pd.to_numeric(traffic_df[col], errors="coerce")

    traffic_df = traffic_df.dropna(subset=["link_id", "speed_band", "collected_at"]).copy()
    traffic_df["link_id"] = traffic_df["link_id"].astype("int64")

    traffic_df["current_speed_band"] = traffic_df["speed_band"]
    traffic_df["avg_speed"] = (
        traffic_df["minimum_speed"] + traffic_df["maximum_speed"]
    ) / 2.0
    traffic_df["current_congestion_score"] = 9 - traffic_df["speed_band"]

    traffic_df["collected_at"] = pd.to_datetime(
        traffic_df["collected_at"],
        utc=True,
        errors="coerce",
    )

    traffic_df = traffic_df.dropna(subset=["collected_at"]).copy()

    return traffic_df


# ============================================================
# R2 ROAD SEGMENT + WEATHER STATION LOADING
# ============================================================

def load_road_segments_from_r2(
    key: str = "road_segments/latest/road_segments_latest.parquet",
) -> pd.DataFrame:
    segments_df = read_r2_parquet(key)

    if segments_df.empty:
        return pd.DataFrame(columns=["link_id", "mid_lat", "mid_lon", "road_category"])

    segments_df = segments_df.copy()
    segments_df.columns = [str(c).strip().lower() for c in segments_df.columns]

    required_cols = ["link_id", "start_lat", "start_lon", "end_lat", "end_lon"]
    missing_cols = [col for col in required_cols if col not in segments_df.columns]

    if missing_cols:
        raise ValueError(f"Road segments file missing columns: {missing_cols}")

    for col in required_cols:
        segments_df[col] = pd.to_numeric(segments_df[col], errors="coerce")

    if "road_category" in segments_df.columns:
        segments_df["road_category"] = pd.to_numeric(
            segments_df["road_category"],
            errors="coerce",
        )
    else:
        segments_df["road_category"] = pd.NA
        logger.warning("Road segments file does not contain road_category.")

    segments_df = segments_df.dropna(subset=["link_id", "start_lat", "start_lon", "end_lat", "end_lon"]).copy()
    segments_df["link_id"] = segments_df["link_id"].astype("int64")

    segments_df["mid_lat"] = (segments_df["start_lat"] + segments_df["end_lat"]) / 2.0
    segments_df["mid_lon"] = (segments_df["start_lon"] + segments_df["end_lon"]) / 2.0

    return segments_df[
        ["link_id", "mid_lat", "mid_lon", "road_category"]
    ].drop_duplicates("link_id")


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalise_weather_stations_df(stations_df: pd.DataFrame) -> pd.DataFrame:
    if stations_df.empty:
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude"])

    stations_df = stations_df.copy()
    stations_df.columns = [str(c).strip().lower() for c in stations_df.columns]

    station_id_col = _find_first_existing_column(
        stations_df,
        ["station_id", "id", "device_id"],
    )
    station_name_col = _find_first_existing_column(
        stations_df,
        ["station_name", "name", "description"],
    )
    lat_col = _find_first_existing_column(
        stations_df,
        ["latitude", "lat"],
    )
    lon_col = _find_first_existing_column(
        stations_df,
        ["longitude", "lon", "lng"],
    )

    if station_id_col is None or lat_col is None or lon_col is None:
        logger.warning("Weather stations missing required columns. Columns=%s", stations_df.columns.tolist())
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude"])

    out = pd.DataFrame()
    out["station_id"] = stations_df[station_id_col].astype(str)

    if station_name_col is not None:
        out["station_name"] = stations_df[station_name_col].astype(str)
    else:
        out["station_name"] = out["station_id"]

    out["latitude"] = pd.to_numeric(stations_df[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(stations_df[lon_col], errors="coerce")

    out = out.dropna(subset=["station_id", "latitude", "longitude"]).copy()
    out = out[
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    ].copy()

    return out.drop_duplicates("station_id")


def load_weather_stations_from_r2(
    latest_key: str = "weather_stations/latest/weather_stations_latest.parquet",
    fallback_prefix: str = "weather_stations/",
) -> pd.DataFrame:
    try:
        stations_df = read_r2_parquet(latest_key)
        stations_df = normalise_weather_stations_df(stations_df)
        if not stations_df.empty:
            return stations_df
    except Exception as exc:
        logger.warning("Could not load latest weather stations key=%s error=%s", latest_key, exc)

    keys = list_r2_parquet_keys(fallback_prefix)
    keys = [
        key for key in keys
        if "/latest/" not in key and "weather_stations_snapshot_" in key
    ]

    if not keys:
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude"])

    latest_history_key = sorted(keys)[-1]
    stations_df = read_r2_parquet(latest_history_key)

    return normalise_weather_stations_df(stations_df)


def build_nearest_station_map_from_r2() -> pd.DataFrame:
    segments_df = load_road_segments_from_r2()
    stations_df = load_weather_stations_from_r2()

    if segments_df.empty:
        raise ValueError("No road segments found in R2.")

    if stations_df.empty:
        raise ValueError("No weather stations found in R2.")

    segment_coords_rad = np.radians(
        segments_df[["mid_lat", "mid_lon"]].to_numpy(dtype=float)
    )
    station_coords_rad = np.radians(
        stations_df[["latitude", "longitude"]].to_numpy(dtype=float)
    )

    tree = BallTree(station_coords_rad, metric="haversine")
    distances_rad, indices = tree.query(segment_coords_rad, k=1)

    nearest_indices = indices[:, 0]
    distances_km = distances_rad[:, 0] * EARTH_RADIUS_KM

    nearest_stations = stations_df.iloc[nearest_indices].reset_index(drop=True)

    station_map = pd.DataFrame({
        "link_id": segments_df["link_id"].to_numpy(),
        "station_id": nearest_stations["station_id"].to_numpy(),
        "station_name": nearest_stations["station_name"].to_numpy(),
        "station_distance_km": distances_km,
    })

    station_map["link_id"] = station_map["link_id"].astype("int64")

    return station_map


# ============================================================
# R2 RAINFALL LOADING + STATION-BASED JOINING
# ============================================================

def normalise_rainfall_readings_df(df: pd.DataFrame, source_key: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_id", "collected_at", "rainfall_mm"])

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    station_id_col = _find_first_existing_column(
        df,
        ["station_id", "id", "device_id"],
    )
    rainfall_value_col = _find_first_existing_column(
        df,
        ["rainfall_mm", "value", "reading", "rainfall", "amount"],
    )

    if station_id_col is None or rainfall_value_col is None:
        logger.warning(
            "Rainfall readings missing required columns. Columns=%s",
            df.columns.tolist(),
        )
        return pd.DataFrame(columns=["station_id", "collected_at", "rainfall_mm"])

    out = pd.DataFrame()
    out["station_id"] = df[station_id_col].astype(str)
    out["rainfall_mm"] = pd.to_numeric(df[rainfall_value_col], errors="coerce")

    if "collected_at" in df.columns:
        out["collected_at"] = pd.to_datetime(
            df["collected_at"],
            utc=True,
            errors="coerce",
        )
    else:
        parsed_ts = extract_timestamp_from_key(source_key or "")
        if parsed_ts is None:
            logger.warning("Rainfall has no collected_at and timestamp cannot be parsed from key=%s", source_key)
            return pd.DataFrame(columns=["station_id", "collected_at", "rainfall_mm"])

        out["collected_at"] = pd.Timestamp(parsed_ts)

    out = out.dropna(subset=["station_id", "collected_at"]).copy()
    out["rainfall_mm"] = out["rainfall_mm"].fillna(0)

    return out[["station_id", "collected_at", "rainfall_mm"]]


def load_rainfall_history_from_r2(
    lookback_hours: int | None = None,
    rainfall_prefix: str = "rainfall_readings/",
) -> pd.DataFrame:
    keys = list_r2_parquet_keys(rainfall_prefix)
    keys = [
        key for key in keys
        if "/latest/" not in key and "rainfall_readings_snapshot_" in key
    ]
    keys = filter_keys_by_lookback(keys, lookback_hours)

    if not keys:
        logger.warning("No rainfall parquet files found in R2 under prefix=%s", rainfall_prefix)
        return pd.DataFrame(columns=["station_id", "collected_at", "rainfall_mm"])

    frames = []

    for key in keys:
        try:
            raw_df = read_r2_parquet(key)
            rainfall_df = normalise_rainfall_readings_df(raw_df, source_key=key)

            if rainfall_df.empty:
                continue

            frames.append(rainfall_df)

        except Exception as exc:
            logger.warning("Failed to read rainfall parquet key=%s error=%s", key, exc)

    if not frames:
        return pd.DataFrame(columns=["station_id", "collected_at", "rainfall_mm"])

    rainfall_df = pd.concat(frames, ignore_index=True)

    rainfall_df = (
        rainfall_df
        .groupby(["station_id", "collected_at"], as_index=False)["rainfall_mm"]
        .mean()
        .sort_values(["station_id", "collected_at"])
    )

    return rainfall_df


def attach_rainfall_to_traffic(
    traffic_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
    station_map_df: pd.DataFrame,
    tolerance_minutes: int = 10,
) -> pd.DataFrame:
    """
    traffic link_id
    -> nearest weather station
    -> rainfall reading from that station
    -> nearest timestamp within tolerance
    """
    traffic_df = traffic_df.copy()

    if traffic_df.empty:
        return traffic_df

    traffic_df["collected_at"] = pd.to_datetime(
        traffic_df["collected_at"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    traffic_df = traffic_df.dropna(subset=["collected_at"]).copy()

    if rainfall_df.empty or station_map_df.empty:
        logger.warning("Rainfall or station map is empty. Filling rainfall_mm=0.")
        traffic_df["rainfall_mm"] = 0.0
        traffic_df["station_id"] = None
        traffic_df["station_name"] = None
        traffic_df["station_distance_km"] = 0.0
        return traffic_df

    rainfall_df = rainfall_df.copy()
    rainfall_df["collected_at"] = pd.to_datetime(
        rainfall_df["collected_at"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    rainfall_df = rainfall_df.dropna(subset=["collected_at"]).copy()

    traffic_df = traffic_df.merge(
        station_map_df,
        on="link_id",
        how="left",
    )

    traffic_df["station_id"] = traffic_df["station_id"].astype("string")
    rainfall_df["station_id"] = rainfall_df["station_id"].astype("string")

    output_frames = []

    for station_id, traffic_group in traffic_df.groupby("station_id", dropna=False):
        traffic_group = traffic_group.sort_values("collected_at").copy()

        if pd.isna(station_id):
            traffic_group["rainfall_mm"] = 0.0
            output_frames.append(traffic_group)
            continue

        rainfall_group = rainfall_df[
            rainfall_df["station_id"] == station_id
        ].sort_values("collected_at").copy()

        if rainfall_group.empty:
            traffic_group["rainfall_mm"] = 0.0
            output_frames.append(traffic_group)
            continue

        merged_group = pd.merge_asof(
            traffic_group,
            rainfall_group[["collected_at", "rainfall_mm"]],
            on="collected_at",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )

        merged_group["rainfall_mm"] = pd.to_numeric(
            merged_group["rainfall_mm"],
            errors="coerce",
        ).fillna(0)

        output_frames.append(merged_group)

    return pd.concat(output_frames, ignore_index=True)


# ============================================================
# R2 CONTEXT FEATURES
# ============================================================

def load_context_features_latest_from_r2(
    context_key: str = "traffic_context_features/latest/context_features.parquet",
) -> pd.DataFrame:
    try:
        context_df = read_r2_parquet(context_key)
    except Exception as exc:
        logger.warning("Could not read R2 context features key=%s error=%s", context_key, exc)
        return pd.DataFrame(columns=[
            "link_id",
            "taxi_count",
            "poi_density",
            "traffic_incident_count",
        ])

    if context_df.empty:
        return pd.DataFrame(columns=[
            "link_id",
            "taxi_count",
            "poi_density",
            "traffic_incident_count",
        ])

    context_df = context_df.copy()
    context_df.columns = [str(c).strip().lower() for c in context_df.columns]

    if "link_id" not in context_df.columns:
        logger.warning("Context features missing link_id. Columns=%s", context_df.columns.tolist())
        return pd.DataFrame(columns=[
            "link_id",
            "taxi_count",
            "poi_density",
            "traffic_incident_count",
        ])

    keep_cols = ["link_id", "taxi_count", "poi_density", "traffic_incident_count"]

    for col in keep_cols:
        if col not in context_df.columns:
            context_df[col] = 0

    context_df = context_df[keep_cols].copy()
    context_df["link_id"] = pd.to_numeric(context_df["link_id"], errors="coerce")
    context_df = context_df.dropna(subset=["link_id"]).copy()
    context_df["link_id"] = context_df["link_id"].astype("int64")

    for col in ["taxi_count", "poi_density", "traffic_incident_count"]:
        context_df[col] = pd.to_numeric(context_df[col], errors="coerce").fillna(0)

    return context_df


def attach_context_features_to_traffic(
    traffic_df: pd.DataFrame,
    context_df: pd.DataFrame,
    tolerance_minutes: int = 10,
) -> pd.DataFrame:
    """
    traffic.link_id + traffic.collected_at
    -> same context.link_id
    -> nearest context.feature_timestamp within tolerance
    """
    traffic_df = traffic_df.copy()

    if traffic_df.empty:
        return traffic_df

    traffic_df["collected_at"] = pd.to_datetime(
        traffic_df["collected_at"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    traffic_df = traffic_df.dropna(subset=["collected_at"]).copy()

    if context_df.empty:
        logger.warning("Context features are empty. Filling context feature columns with 0.")
        traffic_df["taxi_count"] = 0.0
        traffic_df["poi_density"] = 0.0
        traffic_df["traffic_incident_count"] = 0.0
        traffic_df["context_feature_timestamp"] = pd.NaT
        return traffic_df

    context_df = context_df.copy()

    traffic_df["link_id"] = pd.to_numeric(
        traffic_df["link_id"],
        errors="coerce",
    )

    context_df["link_id"] = pd.to_numeric(
        context_df["link_id"],
        errors="coerce",
    )

    traffic_df = traffic_df.dropna(subset=["link_id"]).copy()
    context_df = context_df.dropna(subset=["link_id"]).copy()

    traffic_df["link_id"] = traffic_df["link_id"].astype("int64")
    context_df["link_id"] = context_df["link_id"].astype("int64")

    context_df["feature_timestamp"] = pd.to_datetime(
        context_df["feature_timestamp"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    context_df = context_df.dropna(subset=["feature_timestamp"]).copy()

    for col in ["taxi_count", "poi_density", "traffic_incident_count"]:
        if col not in context_df.columns:
            context_df[col] = 0.0

        context_df[col] = pd.to_numeric(
            context_df[col],
            errors="coerce",
        ).fillna(0)

    output_frames = []

    for link_id, traffic_group in traffic_df.groupby("link_id", sort=False):
        traffic_group = traffic_group.sort_values("collected_at").copy()

        context_group = context_df[
            context_df["link_id"] == link_id
        ].sort_values("feature_timestamp").copy()

        if context_group.empty:
            traffic_group["taxi_count"] = 0.0
            traffic_group["poi_density"] = 0.0
            traffic_group["traffic_incident_count"] = 0.0
            traffic_group["context_feature_timestamp"] = pd.NaT
            output_frames.append(traffic_group)
            continue

        merged_group = pd.merge_asof(
            traffic_group,
            context_group[
                [
                    "feature_timestamp",
                    "taxi_count",
                    "poi_density",
                    "traffic_incident_count",
                ]
            ],
            left_on="collected_at",
            right_on="feature_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )

        merged_group = merged_group.rename(
            columns={"feature_timestamp": "context_feature_timestamp"}
        )

        for col in ["taxi_count", "poi_density", "traffic_incident_count"]:
            merged_group[col] = pd.to_numeric(
                merged_group[col],
                errors="coerce",
            ).fillna(0)

        output_frames.append(merged_group)

    if not output_frames:
        traffic_df["taxi_count"] = 0.0
        traffic_df["poi_density"] = 0.0
        traffic_df["traffic_incident_count"] = 0.0
        traffic_df["context_feature_timestamp"] = pd.NaT
        return traffic_df

    return pd.concat(output_frames, ignore_index=True)

# ============================================================
# LABEL BUILDING
# ============================================================

def build_future_labels(
    df: pd.DataFrame,
    lookahead_minutes: int = 15,
    tolerance_minutes: int = 5,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["collected_at"] = pd.to_datetime(
        df["collected_at"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    df = df.dropna(subset=["collected_at"]).copy()

    frames = []

    label_source = df[
        [
            "collected_at",
            "link_id",
            "current_congestion_score",
        ]
    ].copy()

    label_source = label_source.rename(
        columns={
            "collected_at": "future_collected_at",
            "current_congestion_score": TARGET_COLUMN,
        }
    )

    label_source["future_collected_at"] = pd.to_datetime(
        label_source["future_collected_at"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    for link_id, current_group in df.groupby("link_id", sort=False):
        current_group = current_group.sort_values("collected_at").copy()

        future_group = label_source[
            label_source["link_id"] == link_id
        ].sort_values("future_collected_at").copy()

        if future_group.empty:
            continue

        current_group["desired_future_at"] = (
            current_group["collected_at"] + pd.Timedelta(minutes=lookahead_minutes)
        )

        current_group["desired_future_at"] = pd.to_datetime(
            current_group["desired_future_at"],
            utc=True,
            errors="coerce",
        ).astype("datetime64[ns, UTC]")

        labeled = pd.merge_asof(
            current_group.sort_values("desired_future_at"),
            future_group,
            left_on="desired_future_at",
            right_on="future_collected_at",
            by="link_id",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )

        frames.append(labeled)

    if not frames:
        return pd.DataFrame()

    labeled_df = pd.concat(frames, ignore_index=True)
    labeled_df = labeled_df.dropna(subset=[TARGET_COLUMN]).copy()

    return labeled_df

# ============================================================
# R2 TRAINING FRAME
# ============================================================

def load_training_frame_from_r2(
    lookahead_minutes: int = 15,
    lookback_hours: int | None = None,
) -> pd.DataFrame:
    logger.info(
        "Loading R2 training frame | lookahead_minutes=%s lookback_hours=%s",
        lookahead_minutes,
        lookback_hours,
    )

    traffic_df = load_traffic_history_from_r2(
        lookback_hours=lookback_hours,
    )

    if traffic_df.empty:
        raise ValueError("No traffic history loaded from R2.")
    
    road_segments_df = load_road_segments_from_r2()

    if "road_category" not in traffic_df.columns or traffic_df["road_category"].isna().all():
        traffic_df = traffic_df.drop(columns=["road_category"], errors="ignore")
        traffic_df = traffic_df.merge(
            road_segments_df[["link_id", "road_category"]],
            on="link_id",
            how="left",
        )

    rainfall_df = load_rainfall_history_from_r2(
        lookback_hours=lookback_hours,
    )

    station_map_df = build_nearest_station_map_from_r2()

    context_df = load_context_features_history_from_r2(
        lookback_hours=lookback_hours,
    )

    df = attach_rainfall_to_traffic(
        traffic_df=traffic_df,
        rainfall_df=rainfall_df,
        station_map_df=station_map_df,
        tolerance_minutes=10,
    )

    df = attach_context_features_to_traffic(
        traffic_df=df,
        context_df=context_df,
        tolerance_minutes=10,
    )

    df["hour_of_day"] = df["collected_at"].dt.hour
    df["day_of_week"] = df["collected_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "road_name" not in df.columns:
        df["road_name"] = None

    if "station_distance_km" not in df.columns:
        df["station_distance_km"] = 0.0

    labeled_df = build_future_labels(
        df,
        lookahead_minutes=lookahead_minutes,
        tolerance_minutes=5,
    )

    logger.info("R2 labeled training frame rows=%s", len(labeled_df))

    return labeled_df.sort_values(
        ["collected_at", "link_id"]
    ).reset_index(drop=True)


# ============================================================
# SUPABASE SCHEMA + OLD TRAINING SNAPSHOT FUNCTIONS
# Kept for compatibility.
# ============================================================

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
                taxi_count double precision DEFAULT 0,
                poi_density double precision DEFAULT 0,
                traffic_incident_count double precision DEFAULT 0,
                station_id text,
                station_name text,
                station_distance_km double precision,
                rainfall_timestamp timestamptz,
                context_feature_timestamp timestamptz,
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
            ALTER TABLE traffic_rainfall_training_data
            ADD COLUMN IF NOT EXISTS taxi_count double precision DEFAULT 0
            """
        )
        cur.execute(
            """
            ALTER TABLE traffic_rainfall_training_data
            ADD COLUMN IF NOT EXISTS poi_density double precision DEFAULT 0
            """
        )
        cur.execute(
            """
            ALTER TABLE traffic_rainfall_training_data
            ADD COLUMN IF NOT EXISTS traffic_incident_count double precision DEFAULT 0
            """
        )
        cur.execute(
            """
            ALTER TABLE traffic_rainfall_training_data
            ADD COLUMN IF NOT EXISTS context_feature_timestamp timestamptz
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_context_features (
                feature_timestamp timestamptz NOT NULL,
                link_id bigint NOT NULL,
                taxi_count double precision DEFAULT 0,
                poi_density double precision DEFAULT 0,
                traffic_incident_count double precision DEFAULT 0,
                source_taxi_timestamp timestamptz,
                source_poi_timestamp timestamptz,
                source_incident_timestamp timestamptz,
                inserted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (feature_timestamp, link_id)
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_context_features_link_time
            ON traffic_context_features (link_id, feature_timestamp DESC)
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
                taxi_count double precision DEFAULT 0,
                poi_density double precision DEFAULT 0,
                traffic_incident_count double precision DEFAULT 0,
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
            ALTER TABLE congestion_predictions
            ADD COLUMN IF NOT EXISTS taxi_count double precision DEFAULT 0
            """
        )
        cur.execute(
            """
            ALTER TABLE congestion_predictions
            ADD COLUMN IF NOT EXISTS poi_density double precision DEFAULT 0
            """
        )
        cur.execute(
            """
            ALTER TABLE congestion_predictions
            ADD COLUMN IF NOT EXISTS traffic_incident_count double precision DEFAULT 0
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
                taxi_count,
                poi_density,
                traffic_incident_count,
                station_id,
                station_name,
                station_distance_km,
                rainfall_timestamp,
                context_feature_timestamp,
                hour_of_day,
                day_of_week,
                is_weekend
            )
            SELECT
                trl.collected_at,
                trl.link_id,
                trl.road_name,
                trl.road_category,
                trl.speed_band,
                trl.minimum_speed,
                trl.maximum_speed,
                (trl.minimum_speed + trl.maximum_speed) / 2.0 AS avg_speed,
                9 - trl.speed_band AS congestion_score,
                COALESCE(trl.rainfall_mm, 0) AS rainfall_mm,
                COALESCE(context.taxi_count, 0) AS taxi_count,
                COALESCE(context.poi_density, 0) AS poi_density,
                COALESCE(context.traffic_incident_count, 0) AS traffic_incident_count,
                trl.station_id,
                trl.station_name,
                trl.station_distance_km,
                trl.rainfall_timestamp,
                context.feature_timestamp AS context_feature_timestamp,
                EXTRACT(HOUR FROM trl.collected_at)::integer AS hour_of_day,
                EXTRACT(DOW FROM trl.collected_at)::integer AS day_of_week,
                EXTRACT(DOW FROM trl.collected_at)::integer IN (0, 6) AS is_weekend
            FROM traffic_rainfall_latest AS trl
            LEFT JOIN LATERAL (
                SELECT
                    feature_timestamp,
                    taxi_count,
                    poi_density,
                    traffic_incident_count
                FROM traffic_context_features AS tcf
                WHERE tcf.link_id = trl.link_id
                  AND tcf.feature_timestamp >= trl.collected_at - interval '10 minutes'
                  AND tcf.feature_timestamp <= trl.collected_at + interval '10 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (tcf.feature_timestamp - trl.collected_at)))
                LIMIT 1
            ) AS context ON TRUE
            WHERE trl.speed_band IS NOT NULL
            ON CONFLICT (collected_at, link_id) DO UPDATE SET
                road_name = EXCLUDED.road_name,
                road_category = EXCLUDED.road_category,
                speed_band = EXCLUDED.speed_band,
                minimum_speed = EXCLUDED.minimum_speed,
                maximum_speed = EXCLUDED.maximum_speed,
                avg_speed = EXCLUDED.avg_speed,
                congestion_score = EXCLUDED.congestion_score,
                rainfall_mm = EXCLUDED.rainfall_mm,
                taxi_count = EXCLUDED.taxi_count,
                poi_density = EXCLUDED.poi_density,
                traffic_incident_count = EXCLUDED.traffic_incident_count,
                station_id = EXCLUDED.station_id,
                station_name = EXCLUDED.station_name,
                station_distance_km = EXCLUDED.station_distance_km,
                rainfall_timestamp = EXCLUDED.rainfall_timestamp,
                context_feature_timestamp = EXCLUDED.context_feature_timestamp,
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
                taxi_count,
                poi_density,
                traffic_incident_count,
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
                COALESCE(current.taxi_count, 0) AS taxi_count,
                COALESCE(current.poi_density, 0) AS poi_density,
                COALESCE(current.traffic_incident_count, 0) AS traffic_incident_count,
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


# ============================================================
# MODEL TRAINING
# ============================================================

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


# ============================================================
# PREDICTION FRAME
# ============================================================

def load_latest_prediction_frame(conn) -> pd.DataFrame:
    ensure_ml_schema(conn)

    query = """
        SELECT
            trl.collected_at,
            trl.link_id,
            trl.road_name,
            trl.road_category,
            trl.speed_band AS current_speed_band,
            trl.minimum_speed,
            trl.maximum_speed,
            (trl.minimum_speed + trl.maximum_speed) / 2.0 AS avg_speed,
            9 - trl.speed_band AS current_congestion_score,
            COALESCE(trl.rainfall_mm, 0) AS rainfall_mm,
            COALESCE(context.taxi_count, 0) AS taxi_count,
            COALESCE(context.poi_density, 0) AS poi_density,
            COALESCE(context.traffic_incident_count, 0) AS traffic_incident_count,
            COALESCE(trl.station_distance_km, 0) AS station_distance_km,
            EXTRACT(HOUR FROM trl.collected_at)::integer AS hour_of_day,
            EXTRACT(DOW FROM trl.collected_at)::integer AS day_of_week,
            EXTRACT(DOW FROM trl.collected_at)::integer IN (0, 6) AS is_weekend
        FROM traffic_rainfall_latest AS trl
        LEFT JOIN LATERAL (
            SELECT
                feature_timestamp,
                taxi_count,
                poi_density,
                traffic_incident_count
            FROM traffic_context_features AS tcf
            WHERE tcf.link_id = trl.link_id
              AND tcf.feature_timestamp >= trl.collected_at - interval '10 minutes'
              AND tcf.feature_timestamp <= trl.collected_at + interval '10 minutes'
            ORDER BY ABS(EXTRACT(EPOCH FROM (tcf.feature_timestamp - trl.collected_at)))
            LIMIT 1
        ) AS context ON TRUE
        WHERE trl.speed_band IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def load_latest_prediction_frame_from_r2() -> pd.DataFrame:
    traffic_key = get_latest_r2_key("traffic_speed/")

    if traffic_key is None:
        raise ValueError("No latest traffic_speed parquet found in R2.")

    traffic_df = read_r2_parquet(traffic_key)
    collected_at = extract_timestamp_from_key(traffic_key)

    if collected_at is None:
        raise ValueError(f"Could not parse collected_at from traffic key: {traffic_key}")

    traffic_df = traffic_df.copy()
    traffic_df["collected_at"] = pd.Timestamp(collected_at)

    for col in ["link_id", "road_category", "speed_band", "minimum_speed", "maximum_speed"]:
        if col in traffic_df.columns:
            traffic_df[col] = pd.to_numeric(traffic_df[col], errors="coerce")

    traffic_df = traffic_df.dropna(subset=["link_id", "speed_band"]).copy()
    traffic_df["link_id"] = traffic_df["link_id"].astype("int64")
    traffic_df["current_speed_band"] = traffic_df["speed_band"]
    traffic_df["avg_speed"] = (
        traffic_df["minimum_speed"] + traffic_df["maximum_speed"]
    ) / 2.0
    traffic_df["current_congestion_score"] = 9 - traffic_df["speed_band"]

    rainfall_df = load_rainfall_history_from_r2(lookback_hours=2)
    station_map_df = build_nearest_station_map_from_r2()
    context_df = load_context_features_latest_from_r2()

    df = attach_rainfall_to_traffic(
        traffic_df=traffic_df,
        rainfall_df=rainfall_df,
        station_map_df=station_map_df,
        tolerance_minutes=10,
    )

    df = attach_context_features_to_traffic(df, context_df)

    df["hour_of_day"] = df["collected_at"].dt.hour
    df["day_of_week"] = df["collected_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "road_name" not in df.columns:
        df["road_name"] = None

    if "station_distance_km" not in df.columns:
        df["station_distance_km"] = 0.0

    return df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    conn,
    prediction_df: pd.DataFrame,
    model_id: int,
    model_name: str,
    lookahead_minutes: int = 15,
) -> int:
    if prediction_df.empty:
        return 0

    ensure_ml_schema(conn)

    prediction_created_at = datetime.now(timezone.utc)
    rows = []

    for row in prediction_df.itertuples(index=False):
        target_time = (
            pd.Timestamp(row.collected_at).to_pydatetime()
            + pd.Timedelta(minutes=lookahead_minutes)
        )

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
                float(row.taxi_count) if hasattr(row, "taxi_count") and pd.notna(row.taxi_count) else 0.0,
                float(row.poi_density) if hasattr(row, "poi_density") and pd.notna(row.poi_density) else 0.0,
                float(row.traffic_incident_count) if hasattr(row, "traffic_incident_count") and pd.notna(row.traffic_incident_count) else 0.0,
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
            taxi_count,
            poi_density,
            traffic_incident_count,
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
            taxi_count = EXCLUDED.taxi_count,
            poi_density = EXCLUDED.poi_density,
            traffic_incident_count = EXCLUDED.traffic_incident_count,
            predicted_congestion_score = EXCLUDED.predicted_congestion_score,
            predicted_speed_band = EXCLUDED.predicted_speed_band,
            model_name = EXCLUDED.model_name
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)

    conn.commit()
    return len(rows)

def upsert_context_features(conn, context_df: pd.DataFrame) -> int:
    """
    Upsert mapped taxi / POI / incident context features into Supabase.

    Expected columns:
    - feature_timestamp
    - link_id
    - taxi_count
    - poi_density
    - traffic_incident_count
    - source_taxi_timestamp
    - source_poi_timestamp
    - source_incident_timestamp
    """
    ensure_ml_schema(conn)

    if context_df.empty:
        return 0

    df = context_df.copy()

    required_cols = [
        "feature_timestamp",
        "link_id",
        "taxi_count",
        "poi_density",
        "traffic_incident_count",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"context_df missing required column: {col}")

    optional_timestamp_cols = [
        "source_taxi_timestamp",
        "source_poi_timestamp",
        "source_incident_timestamp",
    ]

    for col in optional_timestamp_cols:
        if col not in df.columns:
            df[col] = None

    df["feature_timestamp"] = pd.to_datetime(
        df["feature_timestamp"],
        utc=True,
        errors="coerce",
    )

    df["link_id"] = pd.to_numeric(df["link_id"], errors="coerce")
    df["taxi_count"] = pd.to_numeric(df["taxi_count"], errors="coerce").fillna(0)
    df["poi_density"] = pd.to_numeric(df["poi_density"], errors="coerce").fillna(0)
    df["traffic_incident_count"] = pd.to_numeric(
        df["traffic_incident_count"],
        errors="coerce",
    ).fillna(0)

    for col in optional_timestamp_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df = df.dropna(subset=["feature_timestamp", "link_id"]).copy()
    df["link_id"] = df["link_id"].astype("int64")

    if df.empty:
        return 0

    rows = []

    for row in df.itertuples(index=False):
        rows.append(
            (
                pd.Timestamp(row.feature_timestamp).to_pydatetime(),
                int(row.link_id),
                float(row.taxi_count),
                float(row.poi_density),
                float(row.traffic_incident_count),
                (
                    pd.Timestamp(row.source_taxi_timestamp).to_pydatetime()
                    if pd.notna(row.source_taxi_timestamp)
                    else None
                ),
                (
                    pd.Timestamp(row.source_poi_timestamp).to_pydatetime()
                    if pd.notna(row.source_poi_timestamp)
                    else None
                ),
                (
                    pd.Timestamp(row.source_incident_timestamp).to_pydatetime()
                    if pd.notna(row.source_incident_timestamp)
                    else None
                ),
            )
        )

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
        execute_values(cur, sql, rows, page_size=1000)

    conn.commit()

    return len(rows)

def load_context_features_history_from_r2(
    lookback_hours: int | None = None,
    context_prefix: str = "traffic_context_features/",
) -> pd.DataFrame:
    """
    Load historical context feature snapshots from R2.

    These are produced by DAG 6_1 as:
    traffic_context_features/date=YYYYMMDD/hour=HH/context_features_YYYYMMDD_HHMMSS.parquet

    Each file contains:
    - feature_timestamp
    - link_id
    - taxi_count
    - poi_density
    - traffic_incident_count
    """
    keys = list_r2_parquet_keys(context_prefix)

    keys = [
        key for key in keys
        if "/latest/" not in key and "context_features_" in key
    ]

    keys = filter_keys_by_lookback(keys, lookback_hours)

    if not keys:
        logger.warning(
            "No historical context feature parquet files found in R2 under prefix=%s",
            context_prefix,
        )
        return pd.DataFrame(
            columns=[
                "feature_timestamp",
                "link_id",
                "taxi_count",
                "poi_density",
                "traffic_incident_count",
            ]
        )

    frames = []

    for key in keys:
        try:
            df = read_r2_parquet(key)

            if df.empty:
                continue

            df = df.copy()
            df.columns = [str(c).strip().lower() for c in df.columns]

            if "feature_timestamp" not in df.columns:
                parsed_ts = extract_timestamp_from_key(key)

                if parsed_ts is None:
                    logger.warning(
                        "Context feature file has no feature_timestamp and timestamp cannot be parsed from key=%s",
                        key,
                    )
                    continue

                df["feature_timestamp"] = pd.Timestamp(parsed_ts)

            keep_cols = [
                "feature_timestamp",
                "link_id",
                "taxi_count",
                "poi_density",
                "traffic_incident_count",
            ]

            for col in keep_cols:
                if col not in df.columns:
                    if col in ["taxi_count", "poi_density", "traffic_incident_count"]:
                        df[col] = 0
                    else:
                        raise ValueError(f"Context feature file missing required column: {col}")

            df = df[keep_cols].copy()

            frames.append(df)

        except Exception as exc:
            logger.warning(
                "Failed to read context feature parquet key=%s error=%s",
                key,
                exc,
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "feature_timestamp",
                "link_id",
                "taxi_count",
                "poi_density",
                "traffic_incident_count",
            ]
        )

    context_df = pd.concat(frames, ignore_index=True)

    context_df["feature_timestamp"] = pd.to_datetime(
        context_df["feature_timestamp"],
        utc=True,
        errors="coerce",
    )

    context_df["link_id"] = pd.to_numeric(
        context_df["link_id"],
        errors="coerce",
    )

    for col in ["taxi_count", "poi_density", "traffic_incident_count"]:
        context_df[col] = pd.to_numeric(
            context_df[col],
            errors="coerce",
        ).fillna(0)

    context_df = context_df.dropna(
        subset=["feature_timestamp", "link_id"]
    ).copy()

    context_df["link_id"] = context_df["link_id"].astype("int64")

    context_df = (
        context_df
        .groupby(["link_id", "feature_timestamp"], as_index=False)[
            ["taxi_count", "poi_density", "traffic_incident_count"]
        ]
        .mean()
        .sort_values(["link_id", "feature_timestamp"])
    )

    return context_df