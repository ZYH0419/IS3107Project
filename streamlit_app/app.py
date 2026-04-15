from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
from sqlalchemy import create_engine, text


st.set_page_config(
    page_title="Smart City",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Styling ----------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b1220;
            color: white;
        }

        .block-container {
            padding-top: 0;
            padding-bottom: 0;
            max-width: 100%;
        }

        header[data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        .hero-wrap {
            position: relative;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background:
                linear-gradient(rgba(8, 15, 28, 0.45), rgba(8, 15, 28, 0.72)),
                url('https://images.unsplash.com/photo-1525625293386-3f8f99389edd?auto=format&fit=crop&w=1800&q=80') center center / cover no-repeat;
        }

        .hero-overlay {
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at center, rgba(255,255,255,0.06), rgba(0,0,0,0.28));
        }

        .hero-content {
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 2rem;
        }

        .hero-title {
            font-size: clamp(3rem, 10vw, 7rem);
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0;
            color: #f8fafc;
            text-shadow: 0 12px 30px rgba(0,0,0,0.45);
        }

        .hero-subtitle {
            margin-top: 1rem;
            font-size: clamp(1rem, 2vw, 1.25rem);
            color: rgba(248,250,252,0.9);
        }

        .scroll-hint {
            position: absolute;
            bottom: 28px;
            left: 50%;
            transform: translateX(-50%);
            color: rgba(255,255,255,0.82);
            font-size: 0.95rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .section-wrap {
            min-height: 100vh;
            padding: 2rem 2rem 3rem 2rem;
            background: #0b1220;
        }

        .map-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.35rem;
        }

        .map-subtitle {
            color: rgba(226,232,240,0.85);
            margin-bottom: 1rem;
        }

        .legend-box {
            background: rgba(15,23,42,0.92);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 14px;
            padding: 1rem;
            margin-top: 3.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }

        .legend-title {
            font-weight: 700;
            margin-bottom: 0.8rem;
            color: #f8fafc;
        }

        .legend-gradient {
            height: 220px;
            width: 22px;
            border-radius: 999px;
            background: linear-gradient(to top, #7f1d1d 0%, #b91c1c 25%, #f97316 50%, #facc15 75%, #86efac 100%);
            margin-right: 0.75rem;
        }

        .legend-flex {
            display: flex;
            align-items: stretch;
        }

        .legend-labels {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 0.95rem;
            color: #e2e8f0;
        }

        .metric-card {
            background: rgba(15,23,42,0.92);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 14px;
            padding: 1rem;
            margin-top: 1rem;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Database ----------
def get_database_url() -> str:
    candidates = [
        os.getenv("SUPABASE_DB_URI"),
        os.getenv("DATABASE_URL"),
        os.getenv("SUPABASE_DB_URL"),
    ]

    try:
        candidates.extend([
            st.secrets.get("SUPABASE_DB_URI"),
            st.secrets.get("DATABASE_URL"),
            st.secrets.get("SUPABASE_DB_URL"),
        ])
    except Exception:
        pass

    database_url = next((value for value in candidates if value), None)
    if not database_url:
        raise RuntimeError(
            "No database URL found. Add SUPABASE_DB_URI, DATABASE_URL, or SUPABASE_DB_URL to environment variables or Streamlit secrets."
        )

    if database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql://", 1)

    return database_url


@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


@st.cache_data(ttl=120, show_spinner=False)
def load_latest_snapshot() -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    engine = get_engine()

    latest_ts_sql = text(
        """
        SELECT MAX(collected_at) AS latest_ts
        FROM traffic_speed_latest
        """
    )

    data_sql = text(
        """
        SELECT
            rs.link_id,
            COALESCE(rs.road_name, 'Unknown Road') AS road_name,
            rs.road_category,
            rs.start_lat,
            rs.start_lon,
            rs.end_lat,
            rs.end_lon,
            tsl.speed_band,
            tsl.minimum_speed,
            tsl.maximum_speed,
            tsl.collected_at
        FROM traffic_speed_latest AS tsl
        JOIN road_segments AS rs
          ON tsl.link_id = rs.link_id
        WHERE rs.start_lat IS NOT NULL
          AND rs.start_lon IS NOT NULL
          AND rs.end_lat IS NOT NULL
          AND rs.end_lon IS NOT NULL
        """
    )

    with engine.connect() as conn:
        latest_ts_df = pd.read_sql(latest_ts_sql, conn)
        latest_ts = latest_ts_df.loc[0, "latest_ts"] if not latest_ts_df.empty else None
        df = pd.read_sql(data_sql, conn)

    if df.empty:
        return pd.DataFrame(), pd.to_datetime(latest_ts, utc=True) if latest_ts is not None else None

    for col in [
        "link_id",
        "speed_band",
        "minimum_speed",
        "maximum_speed",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute estimated speed for tooltip only
    df["avg_speed"] = df[["minimum_speed", "maximum_speed"]].mean(axis=1, skipna=True)

    df["avg_speed_label"] = df["avg_speed"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )
    df["speed_band_label"] = df["speed_band"].apply(
        lambda x: str(int(x)) if pd.notna(x) else "N/A"
    )
    df["minimum_speed_label"] = df["minimum_speed"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
    )
    df["maximum_speed_label"] = df["maximum_speed"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
    )

    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")
    df["last_valid_update_label"] = df["collected_at"].apply(
        lambda x: x.tz_convert("Asia/Singapore").strftime("%d %b %Y, %I:%M %p SGT")
        if pd.notna(x)
        else "N/A"
    )

    df = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"]).copy()

    df["path"] = df.apply(
        lambda row: [
            [float(row["start_lon"]), float(row["start_lat"])],
            [float(row["end_lon"]), float(row["end_lat"])],
        ],
        axis=1,
    )

    # Use speed band for map coloring
    df["color"] = df["speed_band"].apply(speed_band_to_color)

    return df, pd.to_datetime(latest_ts, utc=True) if latest_ts is not None else None


# ---------- Map helpers ----------
def speed_band_to_color(band: float) -> list[int]:

    # UNAVAILABLE → very light grey (blend into map)
    if pd.isna(band):
        return [200, 200, 200, 120]  # light + semi-transparent

    band = int(band)

    if band <= 2:
        return [180, 30, 30, 220]   # red
    if band <= 4:
        return [255, 120, 0, 220]   # orange
    if band <= 6:
        return [255, 200, 0, 220]   # yellow
    return [80, 200, 120, 220]      # green


# ---------- UI ----------
def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-wrap">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <h1 class="hero-title">Smart City</h1>
                <div class="hero-subtitle">Real-time urban traffic intelligence for Singapore</div>
            </div>
            <div class="scroll-hint">Scroll down</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_map_section(df: pd.DataFrame, latest_ts: Optional[pd.Timestamp]) -> None:
    st.markdown('<section class="section-wrap">', unsafe_allow_html=True)

    left, right = st.columns([5.5, 1.2], gap="large")

    with left:
        if latest_ts is not None:
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.tz_localize("UTC")
            ts_display = latest_ts.tz_convert("Asia/Singapore").strftime("%d %b %Y, %I:%M %p SGT")
        else:
            ts_display = "No data available"

        st.markdown('<div class="map-title">Singapore Real-Time Road Speed</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="map-subtitle">Latest valid update in table: {ts_display}</div>',
            unsafe_allow_html=True,
        )

        if df.empty:
            st.warning(
                "No traffic data found in Supabase. Check whether traffic_speed_latest and road_segments exist and whether Airflow has inserted rows."
            )
        else:
            view_state = pdk.ViewState(
                latitude=1.3521,
                longitude=103.8198,
                zoom=11,
                pitch=0,
            )

            layer = pdk.Layer(
                "PathLayer",
                data=df,
                get_path="path",
                get_width=4,
                width_min_pixels=2,
                width_max_pixels=8,
                get_color="color",
                pickable=True,
                auto_highlight=True,
            )

            deck = pdk.Deck(
                map_style="light",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    "html": """
                        <b>{road_name}</b><br/>
                        Link ID: {link_id}<br/>
                        Road category: {road_category}<br/>
                        Speed band: {speed_band_label}<br/>
                        Minimum speed: {minimum_speed_label} km/h<br/>
                        Maximum speed: {maximum_speed_label} km/h<br/>
                        Estimated speed: {avg_speed_label} km/h<br/>
                        Last valid update: {last_valid_update_label}
                    """,
                    "style": {
                        "backgroundColor": "rgba(15,23,42,0.95)",
                        "color": "white",
                        "fontSize": "13px",
                    },
                },
            )
            st.pydeck_chart(deck, use_container_width=True, height=760)

    with right:
        st.markdown(
            """
            <div class="legend-box">
                <div class="legend-title">Speed Band</div>
                <div class="legend-flex">
                    <div class="legend-gradient"></div>
                    <div class="legend-labels">
                        <div>Band 7–8 (Fast, 60+ km/h)</div>
                        <div>Band 5–6 (Moderate, 40–59)</div>
                        <div>Band 3–4 (Slow, 20–39)</div>
                        <div>Band 1–2 (Severe, 0–19)</div>
                        <div>Unavailable</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not df.empty:
            valid_speed_count = int(df["avg_speed"].notna().sum())
            unavailable_count = int(df["avg_speed"].isna().sum())

            st.markdown(
                f"""
                <div class="metric-card">
                    <div><b>Segments loaded</b></div>
                    <div>{len(df):,}</div>
                </div>
                <div class="metric-card">
                    <div><b>Usable speed rows</b></div>
                    <div>{valid_speed_count:,}</div>
                </div>
                <div class="metric-card">
                    <div><b>Unavailable rows</b></div>
                    <div>{unavailable_count:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</section>", unsafe_allow_html=True)


# ---------- App ----------
def main() -> None:
    inject_css()
    render_hero()

    try:
        df, latest_ts = load_latest_snapshot()
        render_map_section(df, latest_ts)
    except Exception as exc:
        st.markdown('<section class="section-wrap">', unsafe_allow_html=True)
        st.error(
            "Could not load data from Supabase. Check your secrets, environment variables, and whether the tables traffic_speed_latest and road_segments exist with the expected columns."
        )
        st.exception(exc)
        st.markdown("</section>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()