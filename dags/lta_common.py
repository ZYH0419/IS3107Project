import os
import math
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]
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


def _to_python_int_or_none(value):
    if pd.isna(value):
        return None
    return int(value)


def _sanitize_speed_value(value, field_name: str, link_id):
    """
    Treat invalid or suspicious speed-like values as missing.
    Keep logic conservative:
    - null stays null
    - non-numeric already handled earlier
    - negative values become null
    - clearly suspicious sentinel values like 999 become null
    """
    #handle nulls and NaNs first
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    # Round to nearest int if it's a valid number but not an integer (e.g. 45.0 -> 45)
    try:
        value = int(round(value))
    except Exception:
        logger.warning(
            "[SANITIZE ERROR] link_id=%s field=%s value=%s -> NULL",
            link_id,
            field_name,
            value,
        )
        return None


    # Negative speeds are invalid
    if value < 0:
        logger.warning(
            "[SANITIZE] link_id=%s field=%s invalid negative value=%s -> NULL",
            link_id,
            field_name,
            value,
        )
        return None

    # 999 is clearly not a realistic speed here; treat like missing
    if value == 999:
        logger.warning(
            "[SANITIZE] link_id=%s field=%s suspicious sentinel value=%s -> NULL",
            link_id,
            field_name,
            value,
        )
        return None

    return value


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

    # Keep original raw values for debug
    raw_debug_df = snapshots_df.copy()

    # Convert blank strings / invalid numeric values to NaN
    snapshots_df["link_id"] = pd.to_numeric(snapshots_df["link_id"], errors="coerce")
    snapshots_df["speed_band"] = pd.to_numeric(snapshots_df["speed_band"], errors="coerce")
    snapshots_df["minimum_speed"] = pd.to_numeric(snapshots_df["minimum_speed"], errors="coerce")
    snapshots_df["maximum_speed"] = pd.to_numeric(snapshots_df["maximum_speed"], errors="coerce")

    # link_id must exist and fit bigint
    BIGINT_MIN = -(2**63)
    BIGINT_MAX = 2**63 - 1

    invalid_link_mask = (
        snapshots_df["link_id"].isna()
        | (snapshots_df["link_id"] < BIGINT_MIN)
        | (snapshots_df["link_id"] > BIGINT_MAX)
    )

    invalid_link_count = int(invalid_link_mask.sum())
    if invalid_link_count > 0:
        logger.warning("Dropping %s rows with invalid link_id", invalid_link_count)
        logger.warning(
            "[INVALID LINK_ID ROWS] %s",
            raw_debug_df.loc[
                invalid_link_mask,
                ["link_id", "speed_band", "minimum_speed", "maximum_speed"],
            ].head(10).to_dict("records"),
        )
        snapshots_df = snapshots_df.loc[~invalid_link_mask].copy()

    # Convert to clean Python ints / None
    snapshots_df["link_id"] = snapshots_df["link_id"].apply(_to_python_int_or_none)
    snapshots_df["speed_band"] = snapshots_df["speed_band"].apply(_to_python_int_or_none)
    snapshots_df["minimum_speed"] = snapshots_df["minimum_speed"].apply(_to_python_int_or_none)
    snapshots_df["maximum_speed"] = snapshots_df["maximum_speed"].apply(_to_python_int_or_none)

    # Sanitize suspicious speed-like values
    snapshots_df["speed_band"] = snapshots_df.apply(
        lambda row: _sanitize_speed_value(row["speed_band"], "speed_band", row["link_id"]),
        axis=1,
    )
    snapshots_df["minimum_speed"] = snapshots_df.apply(
        lambda row: _sanitize_speed_value(row["minimum_speed"], "minimum_speed", row["link_id"]),
        axis=1,
    )
    snapshots_df["maximum_speed"] = snapshots_df.apply(
        lambda row: _sanitize_speed_value(row["maximum_speed"], "maximum_speed", row["link_id"]),
        axis=1,
    )

    logger.info(
        "[CLEAN DONE] rows=%s | null_speed_band=%s | null_minimum_speed=%s | null_maximum_speed=%s",
        len(snapshots_df),
        snapshots_df["speed_band"].isna().sum(),
        snapshots_df["minimum_speed"].isna().sum(),
        snapshots_df["maximum_speed"].isna().sum(),
    )

    snapshots_df = snapshots_df.astype(object).where(pd.notnull(snapshots_df), None)

    return snapshots_df


def build_latest_upsert_df(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare rows for traffic_speed_latest.

    If both minimum_speed and maximum_speed are missing, treat this as
    "no fresh usable measurement". Do not allow it to overwrite an
    existing valid latest value or timestamp.

    If the link does not already exist in traffic_speed_latest, it will
    still be inserted with NULL values.
    """
    latest_df = snapshots_df.copy()

    missing_measurement_mask = (
        latest_df["minimum_speed"].isna() & latest_df["maximum_speed"].isna()
    )

    # Also null out speed_band in this case so it does not overwrite a good old value
    latest_df.loc[missing_measurement_mask, "speed_band"] = None

    logger.info(
        "[LATEST PREP] rows=%s | missing_measurement_rows=%s",
        len(latest_df),
        int(missing_measurement_mask.sum()),
    )

    return latest_df


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


def upsert_latest_snapshot_rows(conn, snapshots_df: pd.DataFrame, batch_size: int = 500):
    if snapshots_df.empty:
        logger.info("[SKIP] no rows to upsert into traffic_speed_latest")
        return

    latest_df = build_latest_upsert_df(snapshots_df)
    rows = list(latest_df.itertuples(index=False, name=None))

    logger.info("[UPSERT START] upserting %s rows into traffic_speed_latest", len(rows))

    logger.info(
        "[UPSERT DEBUG] max_link_id=%s | min_link_id=%s | max_speed_band=%s | min_speed_band=%s | max_minimum_speed=%s | min_minimum_speed=%s | max_maximum_speed=%s | min_maximum_speed=%s",
        latest_df["link_id"].max(),
        latest_df["link_id"].min(),
        latest_df["speed_band"].max(),
        latest_df["speed_band"].min(),
        latest_df["minimum_speed"].max(),
        latest_df["minimum_speed"].min(),
        latest_df["maximum_speed"].max(),
        latest_df["maximum_speed"].min(),
    )

    sql = """
        INSERT INTO traffic_speed_latest (
            collected_at,
            link_id,
            speed_band,
            minimum_speed,
            maximum_speed
        )
        VALUES %s
        ON CONFLICT (link_id) DO UPDATE SET
            speed_band = COALESCE(EXCLUDED.speed_band, traffic_speed_latest.speed_band),
            minimum_speed = COALESCE(EXCLUDED.minimum_speed, traffic_speed_latest.minimum_speed),
            maximum_speed = COALESCE(EXCLUDED.maximum_speed, traffic_speed_latest.maximum_speed),
            collected_at = CASE
                WHEN EXCLUDED.speed_band IS NOT NULL
                  OR EXCLUDED.minimum_speed IS NOT NULL
                  OR EXCLUDED.maximum_speed IS NOT NULL
                THEN EXCLUDED.collected_at
                ELSE traffic_speed_latest.collected_at
            END
    """

    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=batch_size)
        conn.commit()
        logger.info("[UPSERT DONE] upserted=%s rows into traffic_speed_latest", len(rows))

    except Exception:
        logger.exception("[UPSERT BATCH FAILED] Falling back to row-by-row debug")

        conn.rollback()

        single_sql = """
            INSERT INTO traffic_speed_latest (
                collected_at,
                link_id,
                speed_band,
                minimum_speed,
                maximum_speed
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (link_id) DO UPDATE SET
                speed_band = COALESCE(EXCLUDED.speed_band, traffic_speed_latest.speed_band),
                minimum_speed = COALESCE(EXCLUDED.minimum_speed, traffic_speed_latest.minimum_speed),
                maximum_speed = COALESCE(EXCLUDED.maximum_speed, traffic_speed_latest.maximum_speed),
                collected_at = CASE
                    WHEN EXCLUDED.speed_band IS NOT NULL
                      OR EXCLUDED.minimum_speed IS NOT NULL
                      OR EXCLUDED.maximum_speed IS NOT NULL
                    THEN EXCLUDED.collected_at
                    ELSE traffic_speed_latest.collected_at
                END
        """

        with conn.cursor() as cur:
            for row in rows:
                try:
                    cur.execute(single_sql, row)
                except Exception:
                    logger.exception(
                        "[BAD ROW FOUND] row=%s | repr=%s | types=%s",
                        row,
                        tuple(repr(x) for x in row),
                        tuple(type(x).__name__ for x in row),
                    )
                    conn.rollback()
                    raise

        conn.commit()
        logger.info("[UPSERT DONE AFTER FALLBACK] upserted=%s rows into traffic_speed_latest", len(rows))
