import logging
import math
import os
import time
from datetime import datetime

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

RAINFALL_API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
SUPABASE_DB_URI = os.environ["SUPABASE_DB_URI"]
SUPABASE_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("SUPABASE_CONNECT_TIMEOUT_SECONDS", "20"))
SUPABASE_CONNECT_RETRIES = int(os.environ.get("SUPABASE_CONNECT_RETRIES", "3"))
SUPABASE_CONNECT_RETRY_DELAY_SECONDS = float(
    os.environ.get("SUPABASE_CONNECT_RETRY_DELAY_SECONDS", "5")
)


def get_db_dsn() -> str:
    if SUPABASE_DB_URI.startswith("postgresql+psycopg2://"):
        return SUPABASE_DB_URI.replace("postgresql+psycopg2://", "postgresql://", 1)
    return SUPABASE_DB_URI


def get_connection():
    last_error = None

    for attempt in range(1, SUPABASE_CONNECT_RETRIES + 1):
        try:
            return psycopg2.connect(
                get_db_dsn(),
                connect_timeout=SUPABASE_CONNECT_TIMEOUT_SECONDS,
            )
        except psycopg2.OperationalError as exc:
            last_error = exc
            if attempt == SUPABASE_CONNECT_RETRIES:
                break

            delay_seconds = SUPABASE_CONNECT_RETRY_DELAY_SECONDS * attempt
            logger.warning(
                "Supabase connection attempt %s/%s failed; retrying in %.1fs",
                attempt,
                SUPABASE_CONNECT_RETRIES,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    raise last_error


def fetch_rainfall_payload() -> dict:
    logger.info("Fetching rainfall readings from data.gov.sg")
    response = requests.get(RAINFALL_API_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if payload.get("code") not in (None, 0):
        raise RuntimeError(f"Rainfall API returned non-zero code: {payload.get('code')}")

    return payload


def _payload_data(payload: dict) -> dict:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("Rainfall payload does not contain a valid data object")
    return data


def build_weather_stations_df(payload: dict) -> pd.DataFrame:
    data = _payload_data(payload)
    stations = data.get("stations", [])

    rows = []
    for station in stations:
        location = station.get("location") or {}
        rows.append(
            {
                "station_id": station.get("id") or station.get("stationId"),
                "device_id": station.get("deviceId"),
                "station_name": station.get("name"),
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            }
        )

    stations_df = pd.DataFrame(rows)
    if stations_df.empty:
        return stations_df

    stations_df["latitude"] = pd.to_numeric(stations_df["latitude"], errors="coerce")
    stations_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")
    stations_df = stations_df.dropna(subset=["station_id", "latitude", "longitude"]).copy()
    stations_df["station_id"] = stations_df["station_id"].astype(str)
    stations_df["device_id"] = stations_df["device_id"].fillna(stations_df["station_id"]).astype(str)
    stations_df["station_name"] = stations_df["station_name"].fillna("Unknown station").astype(str)

    return stations_df


def build_rainfall_readings_df(payload: dict) -> pd.DataFrame:
    data = _payload_data(payload)
    readings = data.get("readings", [])

    rows = []
    for reading_group in readings:
        timestamp = pd.to_datetime(reading_group.get("timestamp"), errors="coerce")
        if pd.isna(timestamp):
            logger.warning("Skipping rainfall group with invalid timestamp: %s", reading_group)
            continue

        for reading in reading_group.get("data", []):
            rows.append(
                {
                    "reading_timestamp": timestamp.to_pydatetime(),
                    "station_id": reading.get("stationId") or reading.get("station_id"),
                    "rainfall_mm": reading.get("value"),
                }
            )

    readings_df = pd.DataFrame(rows)
    if readings_df.empty:
        return readings_df

    readings_df["rainfall_mm"] = pd.to_numeric(readings_df["rainfall_mm"], errors="coerce")
    readings_df = readings_df.dropna(subset=["reading_timestamp", "station_id"]).copy()
    readings_df["station_id"] = readings_df["station_id"].astype(str)
    readings_df = readings_df.astype(object).where(pd.notnull(readings_df), None)

    return readings_df


def ensure_weather_schema(conn) -> None:
    logger.info("Ensuring rainfall/weather tables and views exist")

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_stations (
                station_id text PRIMARY KEY,
                device_id text,
                station_name text,
                latitude double precision NOT NULL,
                longitude double precision NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rainfall_readings (
                reading_timestamp timestamptz NOT NULL,
                station_id text NOT NULL REFERENCES weather_stations(station_id),
                rainfall_mm double precision,
                inserted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (reading_timestamp, station_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS road_segment_weather_station (
                link_id bigint PRIMARY KEY REFERENCES road_segments(link_id),
                station_id text NOT NULL REFERENCES weather_stations(station_id),
                distance_km double precision NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rainfall_readings_station_time
            ON rainfall_readings (station_id, reading_timestamp)
            """
        )

        cur.execute(
            """
            CREATE OR REPLACE VIEW traffic_rainfall_recent AS
            SELECT
                tsr.collected_at,
                tsr.link_id,
                rs.road_name,
                rs.road_category,
                tsr.speed_band,
                tsr.minimum_speed,
                tsr.maximum_speed,
                mapping.station_id,
                station.station_name,
                mapping.distance_km AS station_distance_km,
                rainfall.reading_timestamp AS rainfall_timestamp,
                rainfall.rainfall_mm
            FROM traffic_speed_recent AS tsr
            LEFT JOIN road_segments AS rs
              ON tsr.link_id = rs.link_id
            LEFT JOIN road_segment_weather_station AS mapping
              ON tsr.link_id = mapping.link_id
            LEFT JOIN weather_stations AS station
              ON mapping.station_id = station.station_id
            LEFT JOIN LATERAL (
                SELECT rr.reading_timestamp, rr.rainfall_mm
                FROM rainfall_readings AS rr
                WHERE rr.station_id = mapping.station_id
                  AND rr.reading_timestamp >= tsr.collected_at - interval '10 minutes'
                  AND rr.reading_timestamp <= tsr.collected_at + interval '10 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (rr.reading_timestamp - tsr.collected_at)))
                LIMIT 1
            ) AS rainfall ON TRUE
            """
        )

        cur.execute(
            """
            CREATE OR REPLACE VIEW traffic_rainfall_latest AS
            SELECT
                tsl.collected_at,
                tsl.link_id,
                rs.road_name,
                rs.road_category,
                tsl.speed_band,
                tsl.minimum_speed,
                tsl.maximum_speed,
                mapping.station_id,
                station.station_name,
                mapping.distance_km AS station_distance_km,
                rainfall.reading_timestamp AS rainfall_timestamp,
                rainfall.rainfall_mm
            FROM traffic_speed_latest AS tsl
            LEFT JOIN road_segments AS rs
              ON tsl.link_id = rs.link_id
            LEFT JOIN road_segment_weather_station AS mapping
              ON tsl.link_id = mapping.link_id
            LEFT JOIN weather_stations AS station
              ON mapping.station_id = station.station_id
            LEFT JOIN LATERAL (
                SELECT rr.reading_timestamp, rr.rainfall_mm
                FROM rainfall_readings AS rr
                WHERE rr.station_id = mapping.station_id
                  AND rr.reading_timestamp >= tsl.collected_at - interval '10 minutes'
                  AND rr.reading_timestamp <= tsl.collected_at + interval '10 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (rr.reading_timestamp - tsl.collected_at)))
                LIMIT 1
            ) AS rainfall ON TRUE
            """
        )

    conn.commit()


def upsert_weather_stations(conn, stations_df: pd.DataFrame, batch_size: int = 500) -> None:
    if stations_df.empty:
        logger.warning("No weather stations found in rainfall payload")
        return

    rows = list(stations_df.itertuples(index=False, name=None))
    sql = """
        INSERT INTO weather_stations (
            station_id,
            device_id,
            station_name,
            latitude,
            longitude
        )
        VALUES %s
        ON CONFLICT (station_id) DO UPDATE SET
            device_id = EXCLUDED.device_id,
            station_name = EXCLUDED.station_name,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            updated_at = now()
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=batch_size)
    conn.commit()
    logger.info("Upserted %s weather stations", len(rows))


def insert_rainfall_readings(conn, readings_df: pd.DataFrame, batch_size: int = 500) -> None:
    if readings_df.empty:
        logger.warning("No rainfall readings found in rainfall payload")
        return

    rows = list(readings_df.itertuples(index=False, name=None))
    sql = """
        INSERT INTO rainfall_readings (
            reading_timestamp,
            station_id,
            rainfall_mm
        )
        VALUES %s
        ON CONFLICT (reading_timestamp, station_id) DO UPDATE SET
            rainfall_mm = EXCLUDED.rainfall_mm,
            inserted_at = now()
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=batch_size)
    conn.commit()
    logger.info("Upserted %s rainfall readings", len(rows))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def refresh_road_station_mapping(conn, remap_all: bool = False, batch_size: int = 1000) -> None:
    stations_sql = """
        SELECT station_id, latitude, longitude
        FROM weather_stations
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
    """

    if remap_all:
        segments_sql = """
            SELECT link_id, start_lat, start_lon, end_lat, end_lon
            FROM road_segments
            WHERE start_lat IS NOT NULL
              AND start_lon IS NOT NULL
              AND end_lat IS NOT NULL
              AND end_lon IS NOT NULL
        """
    else:
        segments_sql = """
            SELECT rs.link_id, rs.start_lat, rs.start_lon, rs.end_lat, rs.end_lon
            FROM road_segments AS rs
            LEFT JOIN road_segment_weather_station AS mapping
              ON rs.link_id = mapping.link_id
            WHERE mapping.link_id IS NULL
              AND rs.start_lat IS NOT NULL
              AND rs.start_lon IS NOT NULL
              AND rs.end_lat IS NOT NULL
              AND rs.end_lon IS NOT NULL
        """

    with conn.cursor() as cur:
        cur.execute(stations_sql)
        stations_df = pd.DataFrame(
            cur.fetchall(),
            columns=["station_id", "latitude", "longitude"],
        )

        cur.execute(segments_sql)
        segments_df = pd.DataFrame(
            cur.fetchall(),
            columns=["link_id", "start_lat", "start_lon", "end_lat", "end_lon"],
        )

    if stations_df.empty:
        logger.warning("Skipping road-to-station mapping because no weather stations exist")
        return

    if segments_df.empty:
        logger.info("No road segments need weather-station mapping")
        return

    station_rows = stations_df.to_dict("records")
    mapping_rows = []

    for segment in segments_df.itertuples(index=False):
        midpoint_lat = (float(segment.start_lat) + float(segment.end_lat)) / 2
        midpoint_lon = (float(segment.start_lon) + float(segment.end_lon)) / 2

        nearest_station = min(
            station_rows,
            key=lambda station: _haversine_km(
                midpoint_lat,
                midpoint_lon,
                float(station["latitude"]),
                float(station["longitude"]),
            ),
        )
        distance_km = _haversine_km(
            midpoint_lat,
            midpoint_lon,
            float(nearest_station["latitude"]),
            float(nearest_station["longitude"]),
        )

        mapping_rows.append(
            (
                int(segment.link_id),
                nearest_station["station_id"],
                distance_km,
            )
        )

    sql = """
        INSERT INTO road_segment_weather_station (
            link_id,
            station_id,
            distance_km
        )
        VALUES %s
        ON CONFLICT (link_id) DO UPDATE SET
            station_id = EXCLUDED.station_id,
            distance_km = EXCLUDED.distance_km,
            updated_at = now()
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, mapping_rows, page_size=batch_size)
    conn.commit()

    logger.info("Mapped %s road segments to nearest rainfall stations", len(mapping_rows))


def refresh_rainfall_data(remap_all: bool = False) -> None:
    payload = fetch_rainfall_payload()
    stations_df = build_weather_stations_df(payload)
    readings_df = build_rainfall_readings_df(payload)

    conn = None
    try:
        conn = get_connection()
        ensure_weather_schema(conn)
        upsert_weather_stations(conn, stations_df)
        insert_rainfall_readings(conn, readings_df)
        refresh_road_station_mapping(conn, remap_all=remap_all)
    finally:
        if conn is not None:
            conn.close()

    latest_ts = readings_df["reading_timestamp"].max() if not readings_df.empty else None
    logger.info(
        "Rainfall refresh complete | stations=%s readings=%s latest_timestamp=%s",
        len(stations_df),
        len(readings_df),
        latest_ts,
    )
