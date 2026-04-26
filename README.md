# IS3107 Smart City Traffic Congestion Prediction System

Real-time and near-real-time traffic congestion monitoring, feature engineering, machine learning prediction, and dashboard visualization system for Singapore.

The system ingests live traffic speed data from LTA DataMall, rainfall readings from data.gov.sg, taxi availability, traffic incidents, POI/context features, and temporal features. It stores raw and processed data across Cloudflare R2 and Supabase/PostgreSQL, orchestrates pipelines with Apache Airflow, trains supervised machine learning models, predicts 15-minute-ahead congestion, and visualizes current and predicted road conditions through a Streamlit dashboard with PyDeck maps.

---

##  Features

- LTA traffic speed ingestion from Traffic Speed Bands API
- Rainfall ingestion from data.gov.sg real-time rainfall API
- Taxi availability and traffic incident ingestion
- POI feature integration for contextual road-segment features
- Cloudflare R2 data lake storage for parquet snapshots and feature inputs
- Supabase/PostgreSQL serving layer for dashboard and ML outputs
- Apache Airflow DAG orchestration for ingestion, aggregation, cleanup, feature mapping, training, and prediction
- ML training and prediction pipeline for 15-minute-ahead congestion forecasting
- Streamlit dashboard and PyDeck map visualization for real-time and predicted road congestion
- Handling of missing traffic data and incremental updates

---

##  Tech Stack

- Python
- Apache Airflow
- Docker
- Supabase / PostgreSQL
- Cloudflare R2
- Streamlit
- PyDeck
- Pandas
- SQLAlchemy
- scikit-learn
- XGBoost

---

##  Project Structure (Simplified)

```text
.
├── dags/
│   ├── 0_pipeline_master.py
│   ├── 1_load_road_segments.py
│   ├── 2_refresh_traffic_speed.py
│   ├── 3_aggregate_traffic_speed_15min.py
│   ├── 4_cleanup_recent_history.py
│   ├── 5_refresh_rainfall.py
│   ├── 5_1_collecting_training_data.py
│   ├── 5_2_collecting_training_data_rainfall.py
│   ├── 5_3_collecting_training_data_taxi.py
│   ├── 5_4_collecting_training_data_incidents.py
│   ├── 6_collect_training_data.py
│   ├── 6_1_map_r2_context_features.py
│   ├── 7_train_congestion_model.py
│   ├── 8_predict_congestion.py
│   ├── lta_common.py
│   ├── weather_common.py
│   └── ml_common.py
├── streamlit_app/
│   ├── app.py
│   └── data_analysis.py
├── docker-compose.yaml
├── requirements-airflow.txt
├── requirements-streamlit.txt
└── README.md
```

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone <https://github.com/ZYH0419/IS3107Project>
cd <IS3107PROJECT>
```

---

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements-airflow.txt
pip install -r requirements-streamlit.txt
```

---

### 4. Set up environment variables

Create a `.env` file:

```env
SUPABASE_DB_URI=your_supabase_connection_string
LTA_ACCOUNT_KEY=your_lta_datamall_account_key

R2_ENDPOINT=your_cloudflare_r2_endpoint
R2_ACCESS_KEY=your_cloudflare_r2_access_key
R2_SECRET_KEY=your_cloudflare_r2_secret_key
R2_BUCKET=your_cloudflare_r2_bucket_name

RECENT_RETENTION_HOURS=24
ML_TRAINING_LOOKBACK_HOURS=24
```

---

### 5. Start Docker (Airflow + Streamlit)

Make sure Docker is running, then:

```bash
docker compose up --build
```

To run in the background:

```bash
docker compose up --build -d
```

This will start:

- Airflow scheduler and webserver
- Streamlit dashboard
- Supporting project services defined in `docker-compose.yaml`

If you have made changes and want to restart the containers:

```bash
docker compose restart
```

---

##  Access the Applications

### Airflow UI

```text
http://localhost:8080
```

Default login:

```text
username: airflow
password: airflow
```

---

### Streamlit Dashboard

To open the dashboard, run Streamlit directly:

```bash
streamlit run streamlit_app/app.py
```

---

##  Running the Pipeline

### DAG Overview

| DAG | Purpose |
|-----|---------|
| `0_pipeline_master` | Master orchestration DAG for traffic refresh, 15-minute aggregation, and recent-history cleanup |
| `1_load_road_segments` | Load static LTA road segment metadata into Supabase |
| `2_refresh_traffic_speed` | Fetch live LTA traffic speed bands and update latest/recent traffic tables |
| `3_aggregate_traffic_speed_15min` | Aggregate recent traffic speed records into 15-minute time-series data |
| `4_cleanup_recent_history` | Delete old rows from `traffic_speed_recent` based on retention settings |
| `5_refresh_rainfall` | Collect rainfall readings from data.gov.sg and map weather stations to road segments |
| `5_1_collecting_training_data` | Collect full LTA traffic speed snapshots and upload cleaned parquet files to Cloudflare R2 |
| `5_2_collecting_training_data_rainfall` | Collect rainfall payloads and store weather/rainfall parquet data in Cloudflare R2 |
| `5_3_collecting_training_data_taxi` | Collect LTA taxi availability data and store parquet snapshots in Cloudflare R2 |
| `5_4_collecting_training_data_incidents` | Collect LTA traffic incident data and store parquet snapshots in Cloudflare R2 |
| `6_1_map_r2_context_features` | Map R2 taxi, POI, and incident snapshots to road-segment context features |
| `6_collect_training_data` | Persist traffic, rainfall, context, and temporal features for ML training |
| `7_train_congestion_model` | Train candidate congestion prediction models and save the best active model |
| `8_predict_congestion` | Generate 15-minute-ahead congestion predictions using the active model |

---

### Recommended Execution Order

1. Run `1_load_road_segments` to load the road network.
2. Run `2_refresh_traffic_speed` to populate live traffic data.
3. Run `3_aggregate_traffic_speed_15min` to build 15-minute aggregates.
4. Run `5_refresh_rainfall` to populate rainfall and weather-station mappings.
5. Run `5_1_collecting_training_data`, `5_2_collecting_training_data_rainfall`, `5_3_collecting_training_data_taxi`, and `5_4_collecting_training_data_incidents` to collect R2 training snapshots.
6. Run `6_1_map_r2_context_features` to map contextual features to road segments.
7. Run `6_collect_training_data` to persist ML-ready training rows.
8. Run `7_train_congestion_model` to train and register the active model.
9. Run `8_predict_congestion` to write 15-minute-ahead predictions.

---

### Scheduling

Typical setup:

- Traffic refresh: every 10 minutes
- Rainfall, taxi, incident, feature, and prediction DAGs: every 5 minutes
- Traffic aggregation: every 15-30 minutes
- Recent-history cleanup: every 1 hour
- Model training: manual or scheduled after enough training data is available

---

##  Machine Learning

The machine learning pipeline uses a supervised regression approach to predict future road congestion.

- Target: `future_congestion_score_15min`, representing congestion score 15 minutes ahead
- Core features:
  - Current traffic speed band
  - Minimum, maximum, and average speed
  - Rainfall amount
  - Taxi availability count
  - POI density
  - Traffic incident count
  - Hour of day, day of week, and weekend flag
- Candidate models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Evaluation metrics:
  - MAE
  - RMSE
  - R²
- The best model is selected by validation performance and saved to the Supabase model registry table, `congestion_model_registry`.
- The active model is used by the prediction DAG to write prediction results to `congestion_predictions`.

---

##  Dashboard

The Streamlit dashboard provides operational and analytical views of the traffic system:

- Traffic data analysis
- Rainfall impact analysis
- Singapore real-time road speed map
- Congestion prediction outlook
- Singapore predicted road congestion map

The map views use PyDeck to render road segments by speed band or predicted congestion level. Tooltips show road name, link ID, speed details, rainfall/context values, and prediction metadata where available.

---

##  Database Tables

| Table | Description |
|-------|-------------|
| `road_segments` | Static LTA road segment metadata and geometry endpoints |
| `traffic_speed_latest` | Latest valid speed record per road segment |
| `traffic_speed_recent` | Rolling short-term traffic speed history |
| `traffic_speed_15min` | 15-minute aggregated traffic speed time series |
| `weather_stations` | Weather station metadata from data.gov.sg rainfall payloads |
| `rainfall_readings` | Real-time rainfall readings by station and timestamp |
| `road_segment_weather_station` | Nearest-station mapping between road segments and rainfall stations |
| `traffic_context_features` | Road-segment context features from taxi, POI, and incident data |
| `traffic_rainfall_training_data` | ML training snapshots combining traffic, rainfall, context, and temporal features |
| `congestion_model_registry` | Trained model artifacts, metrics, feature columns, and active-model flag |
| `congestion_predictions` | 15-minute-ahead congestion predictions by road segment |

---

##  Visualization

The Streamlit app:

- Displays road-level congestion using `speed_band`
- Visualizes rainfall impact and recent prediction patterns
- Uses PyDeck for interactive map rendering
- Shows grey roads for unavailable speed or prediction data
- Displays current and predicted road metrics through dashboard cards, charts, maps, and tables

---

### Stop the Services

To stop the containers:

```bash
docker compose down
```

---

##  Notes

- Supabase free tier has storage limits; `traffic_speed_recent` is automatically cleaned via DAG 4.
- Missing data is not treated as 0 unless explicitly converted for ML feature defaults.
- Cloudflare R2 stores parquet snapshots for data lake and feature engineering workflows.
- Some networks may block direct database connections; use connection pooling or an allowed network if needed.

---

##  Future Improvements

- Scalable database infrastructure for larger traffic history and higher ingestion frequency
- Better indexing and partitioning for high-volume time-series tables
- Precomputed dashboard aggregates for faster Streamlit rendering
- Time-series or graph-based prediction models for road-network-aware forecasting
- Classification model for severe congestion alerts

---

## 👤 Author

Built as part of an IS3107 Smart City / Data Engineering project.
