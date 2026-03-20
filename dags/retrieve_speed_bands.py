from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

LTA_ACCOUNT_KEY = os.environ["LTA_ACCOUNT_KEY"]
SUPABASE_DB_URI = os.environ["SUPABASE_DB_URI"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def fetch_all_speed_bands() -> pd.DataFrame:
    headers = {
        "AccountKey": LTA_ACCOUNT_KEY,
        "accept": "application/json"
    }

    all_data = []
    skip = 0
    page_size = 500

    while True:
        url = f"{API_URL}?$skip={skip}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        records = data.get("value", [])

        print(f"Fetched {len(records)} rows (skip={skip})")

        if not records:
            break

        all_data.extend(records)

        if len(records) < page_size:
            break

        skip += page_size

    df = pd.DataFrame(all_data)

    print(f"Total rows fetched: {len(df)}")
    if "LinkID" in df.columns:
        print(f"Unique LinkIDs fetched: {df['LinkID'].nunique()}")

    return df


def fetch_and_store_speed_bands():
    df = fetch_all_speed_bands()

    if df.empty:
        print("No records returned from API.")
        return

    collected_at = datetime.now(timezone.utc)

    # -----------------------------
    # Dimension table: static road segment metadata
    # -----------------------------
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

    # -----------------------------
    # Fact table: changing speed values
    # -----------------------------
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

    print(f"Prepared {len(segments_df)} unique road segments")
    print(f"Prepared {len(snapshots_df)} snapshot rows")

    engine = create_engine(SUPABASE_DB_URI)

    with engine.begin() as conn:
        # -----------------------------
        # Stage road segments into temp table
        # -----------------------------
        conn.execute(text("""
            create temporary table tmp_road_segments (
                link_id bigint,
                road_name text,
                road_category integer,
                start_lon double precision,
                start_lat double precision,
                end_lon double precision,
                end_lat double precision
            ) on commit drop;
        """))

        segments_df.to_sql(
            "tmp_road_segments",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

        # Bulk upsert into road_segments
        conn.execute(text("""
            insert into road_segments (
                link_id, road_name, road_category,
                start_lon, start_lat, end_lon, end_lat
            )
            select
                link_id, road_name, road_category,
                start_lon, start_lat, end_lon, end_lat
            from tmp_road_segments
            on conflict (link_id) do update set
                road_name = excluded.road_name,
                road_category = excluded.road_category,
                start_lon = excluded.start_lon,
                start_lat = excluded.start_lat,
                end_lon = excluded.end_lon,
                end_lat = excluded.end_lat;
        """))

        # -----------------------------
        # Insert snapshots
        # -----------------------------
        snapshots_df.to_sql(
            "traffic_speed_snapshots",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    print(f"Inserted {len(snapshots_df)} snapshot rows.")
    print(f"Upserted {len(segments_df)} road segments.")


with DAG(
    dag_id="retrieve_speed_bands",
    default_args=default_args,
    description="Retrieve full LTA Traffic Speed Bands every 5 minutes",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 3, 17),
    catchup=False,
    max_active_runs=1,
    tags=["lta", "traffic", "supabase"],
) as dag:

    run_collection = PythonOperator(
        task_id="fetch_and_store_speed_bands",
        python_callable=fetch_and_store_speed_bands,
    )