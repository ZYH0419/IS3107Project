import os
import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]
SUPABASE_DB_URI = os.environ["SUPABASE_DB_URI"]


def get_db_dsn() -> str:
    if SUPABASE_DB_URI.startswith("postgresql+psycopg2://"):
        return SUPABASE_DB_URI.replace("postgresql+psycopg2://", "postgresql://", 1)
    return SUPABASE_DB_URI


def get_connection():
    return psycopg2.connect(get_db_dsn())


def fetch_speed_bands_page(skip: int, page_size: int = 500) -> pd.DataFrame:
    headers = {
        "AccountKey": LTA_ACCOUNT_KEY,
        "accept": "application/json",
    }

    url = f"{API_URL}?$skip={skip}"
    logger.info("Fetching page from LTA: skip=%s", skip)

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    payload = response.json()
    records = payload.get("value", [])

    logger.info("Retrieved %s rows from LTA at skip=%s", len(records), skip)

    return pd.DataFrame(records)


def iter_speed_bands_pages(page_size: int = 500):
    skip = 0

    while True:
        logger.info("========== PAGE START ==========")
        logger.info("[FETCH] requesting rows %s–%s", skip, skip + page_size - 1)

        df_page = fetch_speed_bands_page(skip=skip, page_size=page_size)

        retrieved = len(df_page)
        logger.info("[FETCH DONE] retrieved=%s rows at skip=%s", retrieved, skip)

        if retrieved == 0:
            logger.info("[STOP] no more rows returned")
            break

        yield skip, df_page

        if retrieved < page_size:
            logger.info("[LAST PAGE] retrieved=%s (<%s)", retrieved, page_size)
            break

        skip += page_size


def build_segments_df(df: pd.DataFrame) -> pd.DataFrame:
    segments_df = df[
        [
            "LinkID",
            "RoadName",
            "RoadCategory",
            "StartLon",
            "StartLat",
            "EndLon",
            "EndLat",
        ]
    ].drop_duplicates(subset=["LinkID"]).copy()

    segments_df.columns = [
        "link_id",
        "road_name",
        "road_category",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
    ]

    return segments_df


def build_snapshots_df(df: pd.DataFrame, collected_at: datetime | None = None) -> pd.DataFrame:
    if collected_at is None:
        collected_at = datetime.now(timezone.utc)

    snapshots_df = df[
        [
            "LinkID",
            "SpeedBand",
            "MinimumSpeed",
            "MaximumSpeed",
        ]
    ].copy()

    snapshots_df["collected_at"] = collected_at

    snapshots_df = snapshots_df[
        [
            "collected_at",
            "LinkID",
            "SpeedBand",
            "MinimumSpeed",
            "MaximumSpeed",
        ]
    ]

    snapshots_df.columns = [
        "collected_at",
        "link_id",
        "speed_band",
        "minimum_speed",
        "maximum_speed",
    ]

    return snapshots_df


def upsert_road_segments(conn, segments_df: pd.DataFrame, batch_size: int = 500):
    if segments_df.empty:
        logger.info("No road segment rows to upsert")
        return

    rows = list(segments_df.itertuples(index=False, name=None))

    logger.info("[UPSERT START] %s rows into road_segments", len(rows))

    sql = """
        INSERT INTO road_segments (
            link_id, road_name, road_category,
            start_lon, start_lat, end_lon, end_lat
        )
        VALUES %s
        ON CONFLICT (link_id) DO UPDATE SET
            road_name = EXCLUDED.road_name,
            road_category = EXCLUDED.road_category,
            start_lon = EXCLUDED.start_lon,
            start_lat = EXCLUDED.start_lat,
            end_lon = EXCLUDED.end_lon,
            end_lat = EXCLUDED.end_lat,
            updated_at = now()
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=batch_size)
    conn.commit()

    logger.info("[UPSERT DONE] %s rows into road_segments", len(rows))


def insert_snapshot_rows(conn, table_name: str, snapshots_df: pd.DataFrame, batch_size: int = 500):
    if snapshots_df.empty:
        logger.info("[SKIP] no rows to insert into %s", table_name)
        return

    rows = list(snapshots_df.itertuples(index=False, name=None))

    logger.info("[SAVE START] inserting %s rows into %s", len(rows), table_name)

    sql = f"""
        INSERT INTO {table_name} (
            collected_at,
            link_id,
            speed_band,
            minimum_speed,
            maximum_speed
        )
        VALUES %s
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=batch_size)

    conn.commit()

    logger.info("[SAVE DONE] inserted=%s rows into %s", len(rows), table_name)