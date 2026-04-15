# Smart City Traffic Analytics System

Real-time traffic data pipeline and visualization system built using Airflow, Supabase, and Streamlit.  
The system ingests live traffic data from LTA DataMall, processes it, and visualizes road-level congestion across Singapore.

---

## 🚀 Features

- Real-time traffic ingestion (LTA API, 5–10 min frequency)
- Airflow-based ETL pipeline (DAG 2–4)
- Optimized data models (`latest`, `recent`, `aggregated`)
- Geospatial visualization using Streamlit + PyDeck
- Handling of missing data and incremental updates

---

## 🏗️ Tech Stack

- Python
- Apache Airflow (Docker)
- Supabase (PostgreSQL)
- Streamlit + PyDeck
- Pandas / SQLAlchemy

---

## 📁 Project Structure (Simplified)

```
.
├── dags/
│   ├── 2_refresh_traffic_speed.py
│   ├── 3_aggregate_traffic_speed_15min.py
│   ├── 4_cleanup_recent_history.py
│   └── lta_common.py
├── streamlit_app/
│   └── app.py
├── docker-compose.yaml
├── requirements.txt
└── .env
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <https://github.com/ZYH0419/IS3107Project>
cd <IS3107PROJECT>
```

---

### 2. Set up environment variables

Create a `.env` file:

```env
SUPABASE_DB_URI=your_supabase_connection_string
LTA_API_KEY=your_lta_api_key
```

---

### 3. Start Docker (Airflow + Streamlit)

Make sure Docker is running, then:

```bash
docker compose up --build
```

This will start:
- Airflow (scheduler + webserver)
- Streamlit app

---

## 🌐 Access the Applications

### Airflow UI

http://localhost:8080

Default login:

username: airflow  
password: airflow  

---

### Streamlit Dashboard

http://localhost:8501

---

## 🔄 Running the Pipeline

### DAG Overview

| DAG | Purpose |
|-----|--------|
| DAG 2 | Fetch + clean traffic data → Supabase |
| DAG 3 | Aggregate into 15-min intervals |
| DAG 4 | Clean up old historical data |

---

### Recommended Execution Order

1. Run **DAG 2** (data ingestion)
2. Run **DAG 3** (aggregation)
3. Optionally run **DAG 4** (cleanup)

---

### Scheduling

Typical setup:

- DAG 2 → every **10 minutes**
- DAG 3 → every **15–30 minutes**
- DAG 4 → every **1 hour**

---

## 🗄️ Database Tables

| Table | Description |
|------|------------|
| `traffic_speed_latest` | Latest valid speed per road |
| `traffic_speed_recent` | Short-term history (rolling window) |
| `traffic_speed_15min` | Aggregated time-series data |
| `road_segments` | Geospatial road network |

---

## 🗺️ Visualization

The Streamlit app:

- Displays road-level congestion using `speed_band`
- Uses PyDeck for map rendering
- Grey roads = unavailable data
- Tooltip shows:
  - Speed band
  - Min / max speed
  - Last valid update time

---

## ⚠️ Notes

- Supabase free tier has storage limits — `traffic_speed_recent` is automatically cleaned via DAG 4
- Missing data is **not treated as 0**; historical values are preserved where applicable
- Some networks (e.g. school WiFi) may block direct DB connections — use connection pooling if needed

---

## 🧠 Future Improvements

- Time-series prediction model (traffic forecasting)
- Delta table for incremental updates
- Parquet-based storage for ML pipelines
- Time slider / animation in Streamlit

---

## 👤 Author

Built as part of a Smart City / Data Engineering project.
