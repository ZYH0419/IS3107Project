# IS3107Project

📌 Project Title

Real-Time Smart Traffic Monitoring and Congestion Prediction System

🏙️ Problem Statement (Smart City Context)

Urban traffic congestion is a major challenge in modern smart cities. It leads to increased travel time, higher fuel consumption, and environmental pollution. Traditional traffic monitoring systems often rely on static analysis or delayed reporting, which limits their ability to support real-time decision-making.

This project aims to build a real-time traffic data pipeline and analytics system that:

continuously ingests traffic speed data across Singapore

visualizes current traffic conditions on an interactive map

captures short-term temporal patterns for congestion analysis

lays the foundation for predictive modeling using machine learning

The system follows a modern data engineering + MLOps architecture, enabling scalable, automated, and intelligent traffic analysis.

📡 Data Source

The system uses the LTA DataMall Traffic Speed Bands API, which provides:

road segment-level traffic speed data

speed categorized into discrete “speed bands”

geospatial information (start/end coordinates of road segments)

Each API snapshot contains approximately 140k+ records, representing traffic conditions across major road segments in Singapore.

To ensure complete data retrieval, the system uses paginated API requests ($skip), iteratively fetching all pages until no further data is returned.

⚙️ Data Pipeline Architecture

The system is orchestrated using Apache Airflow, which automates data ingestion, transformation, and aggregation through scheduled workflows (DAGs).

🔄 DAG 1: load_road_segments

Purpose: populate static road metadata

Extracts:

LinkID, RoadName, RoadCategory

geographic coordinates

Stores into:

road_segments (dimension table)

Execution: manual or infrequent

🔁 DAG 2: refresh_traffic_speed (every 5 minutes)

This is the core real-time pipeline.

Steps:

Fetch full dataset (~143k rows) via paginated API calls

Construct snapshot dataframe

Update:

traffic_speed_latest (overwrite)

traffic_speed_recent (append)

Prune old data from traffic_speed_recent

Purpose:

maintain real-time traffic state

preserve short-term history for temporal analysis

📊 DAG 3: aggregate_traffic_speed_15min (every 15 minutes)

Steps:

Read from traffic_speed_recent

Aggregate into 15-minute intervals

Compute:

average speed band

min/max speed band

sample counts

Upsert into traffic_speed_15min

Purpose:

reduce storage footprint

preserve meaningful temporal patterns

support downstream analytics and ML

🗄️ Database Design (Supabase / PostgreSQL)

The system uses a layered data storage strategy to balance performance and scalability.

1. road_segments (static dimension table)

Stores:

road metadata

geospatial coordinates

Used for:

map visualization

feature enrichment

2. traffic_speed_latest (real-time layer)

Stores:

most recent snapshot only

Used for:

Streamlit dashboard

live traffic visualization

3. traffic_speed_recent (short-term raw history)

Stores:

last 30 minutes to 24 hours of raw data

Used for:

temporal consistency analysis

lag feature construction

near-term congestion reasoning

4. traffic_speed_15min (compressed historical layer)

Stores:

15-minute aggregated traffic data

Used for:

peak-hour analysis

long-term trends

machine learning feature generation

📊 Real-Time Visualization

A Streamlit dashboard is used to visualize:

road segments on a map

traffic conditions using color-coded speed bands

real-time updates from traffic_speed_latest

The visualization enables users to:

identify congestion hotspots

observe spatial traffic patterns

interactively explore traffic conditions

🤖 MLOps (Planned Extension)

The system is designed to support a full MLOps pipeline, including:

Feature Engineering

time-based features (hour, weekday)

lag features (previous traffic states)

rolling statistics

weather integration

Model Training

predict congestion or future speed bands

models such as XGBoost or time-series models

Monitoring

track model performance (RMSE, accuracy)

detect:

data drift

feature drift

prediction drift

Automated Retraining

triggered by:

performance degradation

drift thresholds

orchestrated via Airflow DAGs