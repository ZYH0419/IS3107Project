"""
Download and build a local ML-ready congestion training dataframe from Cloudflare R2.

This script matches the current project pipeline:
- traffic snapshots from: traffic_speed/date=YYYYMMDD/hour=HH/traffic_speed_snapshot_YYYYMMDD_HHMMSS.parquet
- rainfall snapshots from: rainfall_readings/date=YYYYMMDD/hour=HH/rainfall_readings_snapshot_YYYYMMDD_HHMMSS.parquet
- weather stations latest/history from: weather_stations/
- road segments latest from: road_segments/latest/road_segments_latest.parquet
- context features from: traffic_context_features/date=YYYYMMDD/hour=HH/context_features_YYYYMMDD_HHMMSS.parquet

It calls ml_common.load_training_frame_from_r2(), so the local output should match DAG 7's training input.

Run from your project/dags folder, where ml_common.py exists:

    export R2_ENDPOINT="https://...r2.cloudflarestorage.com"
    export R2_ACCESS_KEY="..."
    export R2_SECRET_KEY="..."
    export R2_BUCKET="smart-city"
    export LOOKBACK_HOURS=1
    python download_r2_training_data_local.py

Optional output path:

    export OUTPUT_FILE="ml_ready_last_hour_training_data.parquet"
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------------
# Environment compatibility
# ---------------------------------------------------------------------
# Your current ml_common.py expects these names:
#   R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
# Some earlier scripts used:
#   R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
# This maps either style to the names ml_common.py needs.

ENV_ALIASES = {
    "R2_ENDPOINT": ["R2_ENDPOINT", "R2_ENDPOINT_URL"],
    "R2_ACCESS_KEY": ["R2_ACCESS_KEY", "R2_ACCESS_KEY_ID"],
    "R2_SECRET_KEY": ["R2_SECRET_KEY", "R2_SECRET_ACCESS_KEY"],
    "R2_BUCKET": ["R2_BUCKET", "R2_BUCKET_NAME"],
}


def apply_env_aliases() -> None:
    missing = []

    for canonical_name, possible_names in ENV_ALIASES.items():
        value = None
        for name in possible_names:
            if os.environ.get(name):
                value = os.environ[name]
                break

        if value:
            os.environ[canonical_name] = value
        else:
            missing.append(canonical_name)

    if missing:
        raise EnvironmentError(
            "Missing R2 environment variables: "
            + ", ".join(missing)
            + "\n\nSet them like this:\n"
            + 'export R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"\n'
            + 'export R2_ACCESS_KEY="<access-key>"\n'
            + 'export R2_SECRET_KEY="<secret-key>"\n'
            + 'export R2_BUCKET="smart-city"\n'
        )


apply_env_aliases()


# ---------------------------------------------------------------------
# Import project ML loader after env mapping
# ---------------------------------------------------------------------
try:
    from ml_common import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        load_training_frame_from_r2,
        list_r2_parquet_keys,
        filter_keys_by_lookback,
    )
except Exception as exc:
    raise ImportError(
        "Could not import ml_common.py. Run this script from the folder that contains "
        "ml_common.py, probably your project/dags folder.\n"
        f"Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "1"))
LOOKAHEAD_MINUTES = int(os.environ.get("LOOKAHEAD_MINUTES", "15"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "ml_ready_last_hour_training_data.parquet")
PREVIEW_ROWS = int(os.environ.get("PREVIEW_ROWS", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("download_r2_training_data_local")


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def print_r2_diagnostics() -> None:
    """Print how many files are visible under the real prefixes used by the pipeline."""
    prefixes = {
        "traffic": "traffic_speed/",
        "rainfall": "rainfall_readings/",
        "weather_stations": "weather_stations/",
        "context_features": "traffic_context_features/",
        "road_segments": "road_segments/",
    }

    logger.info("R2 bucket: %s", os.environ["R2_BUCKET"])
    logger.info("Lookback hours: %s", LOOKBACK_HOURS)
    logger.info("Lookahead minutes: %s", LOOKAHEAD_MINUTES)

    for label, prefix in prefixes.items():
        try:
            keys = list_r2_parquet_keys(prefix)
            recent_keys = filter_keys_by_lookback(keys, LOOKBACK_HOURS)
            logger.info(
                "R2 prefix check | %-16s prefix=%s total_parquet=%s recent_by_filename_timestamp=%s",
                label,
                prefix,
                len(keys),
                len(recent_keys),
            )

            sample = recent_keys[-3:] if recent_keys else keys[-3:]
            for key in sample:
                logger.info("  sample key: %s", key)

        except Exception as exc:
            logger.warning("Could not list prefix=%s error=%s", prefix, exc)


# ---------------------------------------------------------------------
# Build dataframe
# ---------------------------------------------------------------------
def build_local_training_dataframe() -> pd.DataFrame:
    """
    Build exactly the same kind of labeled dataframe DAG 7 trains on.
    This includes traffic, rainfall, road_category, context features, time features,
    and future_congestion_score_15min target label.
    """
    df = load_training_frame_from_r2(
        lookahead_minutes=LOOKAHEAD_MINUTES,
        lookback_hours=LOOKBACK_HOURS,
    )

    if df.empty:
        raise ValueError(
            "Training dataframe is empty. Try increasing LOOKBACK_HOURS, e.g.:\n"
            "export LOOKBACK_HOURS=24\n"
            "python download_r2_training_data_local.py"
        )

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"ML-ready dataframe missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"ML-ready dataframe missing target column: {TARGET_COLUMN}")

    # Keep helpful columns first, but do not discard extra debug columns.
    preferred_cols = [
        "collected_at",
        "desired_future_at",
        "future_collected_at",
        "road_name",
        *FEATURE_COLUMNS,
        "current_congestion_score",
        TARGET_COLUMN,
        "station_id",
        "station_name",
        "station_distance_km",
        "context_feature_timestamp",
    ]

    ordered_cols = [col for col in preferred_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + remaining_cols].copy()

    # Make feature columns numeric for easier local training/debugging.
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    return df.sort_values(["collected_at", "link_id"]).reset_index(drop=True)


def main() -> None:
    print_r2_diagnostics()

    logger.info("Building ML-ready local training dataframe...")
    df = build_local_training_dataframe()

    output_path = Path(OUTPUT_FILE)
    df.to_parquet(output_path, index=False)

    logger.info("Saved ML-ready dataframe to: %s", output_path.resolve())
    logger.info("Rows=%s Columns=%s", len(df), len(df.columns))
    logger.info("Time range: %s -> %s", df["collected_at"].min(), df["collected_at"].max())
    logger.info("Target column: %s", TARGET_COLUMN)
    logger.info("Feature columns: %s", FEATURE_COLUMNS)

    print("\nPreview:")
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(df.head(PREVIEW_ROWS))

    print("\nSaved:", output_path.resolve())
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("FAILED: %s", exc)
        sys.exit(1)
