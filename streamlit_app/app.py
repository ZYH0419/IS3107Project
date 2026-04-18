from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt
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
            background: linear-gradient(to top, #7f1d1d 0%, #b91c1c 18%, #f97316 40%, #facc15 68%, #86efac 100%);
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
def get_database_url():
    url = os.getenv("SUPABASE_DB_URI")
    if not url:
        raise RuntimeError("SUPABASE_DB_URI not found in environment variables")

    return url

@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_snapshot() -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    engine = get_engine()

    snapshot_time_sql = text(
        """
        SELECT MAX(collected_at) AS latest_ts
        FROM traffic_speed_latest
        """
    )

    with engine.connect() as conn:
        latest_ts_df = pd.read_sql(snapshot_time_sql, conn)
        latest_ts = latest_ts_df.loc[0, "latest_ts"] if not latest_ts_df.empty else None

        if pd.isna(latest_ts):
            return pd.DataFrame(), None

        query = text(
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
            WHERE tsl.collected_at = :latest_ts
              AND rs.start_lat IS NOT NULL
              AND rs.start_lon IS NOT NULL
              AND rs.end_lat IS NOT NULL
              AND rs.end_lon IS NOT NULL
            """
        )
        df = pd.read_sql(query, conn, params={"latest_ts": latest_ts})

    if df.empty:
        return df, pd.to_datetime(latest_ts, utc=True)

    df["avg_speed"] = (df["minimum_speed"].fillna(0) + df["maximum_speed"].fillna(0)) / 2.0
    df["avg_speed_label"] = df["avg_speed"].round(1)
    df["path"] = df.apply(
        lambda row: [
            [float(row["start_lon"]), float(row["start_lat"])],
            [float(row["end_lon"]), float(row["end_lat"])],
        ],
        axis=1,
    )
    df["color"] = df["avg_speed"].apply(speed_to_color)
    return df, pd.to_datetime(latest_ts, utc=True)


# ---------- Map helpers ----------
def speed_to_color(speed: float) -> list[int]:
    """Lower speed = more congested = redder."""
    if pd.isna(speed):
        return [148, 163, 184, 180]
    if speed <= 20:
        return [127, 29, 29, 220]
    if speed <= 35:
        return [185, 28, 28, 220]
    if speed <= 50:
        return [249, 115, 22, 220]
    if speed <= 65:
        return [250, 204, 21, 220]
    return [134, 239, 172, 220]


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

@st.cache_data(ttl=300, show_spinner=False)
def load_traffic_data(mode: str = "latest"):

    engine = get_engine()

    if mode == "latest":
        latest_ts_sql = text("""
            SELECT MAX(interval_start) AS latest_ts
            FROM traffic_speed_15min
        """)

        with engine.connect() as conn:
            latest_ts_df = pd.read_sql(latest_ts_sql, conn)
            latest_ts = latest_ts_df.loc[0, "latest_ts"]

            if pd.isna(latest_ts):
                return pd.DataFrame(), None

            query = text("""
                SELECT
                    t.interval_start,
                    t.link_id,
                    t.avg_speed_band,
                    t.avg_minimum_speed,
                    t.avg_maximum_speed,
                    r.road_name,
                    r.road_category,
                    r.start_lat,
                    r.start_lon,
                    r.end_lat,
                    r.end_lon
                FROM traffic_speed_15min t
                JOIN road_segments r
                  ON t.link_id = r.link_id
                WHERE t.interval_start = :ts
            """)

            df = pd.read_sql(query, conn, params={"ts": latest_ts})

        return df, pd.to_datetime(latest_ts, utc=True)

    elif mode == "history":
        query = """
        SELECT
            t.interval_start,
            t.link_id,
            t.avg_speed_band,
            t.avg_minimum_speed,
            t.avg_maximum_speed,
            t.samples,
            r.road_name,
            r.road_category
        FROM traffic_speed_15min t
        JOIN road_segments r
          ON t.link_id = r.link_id
        """

        df = pd.read_sql(query, engine)
        df["interval_start"] = pd.to_datetime(df["interval_start"])

        return df, None



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
        st.markdown(f'<div class="map-subtitle">Last refreshed: {ts_display}</div>', unsafe_allow_html=True)

        if df.empty:
            st.warning(
                "No traffic data found in Supabase. Check the connection string, whether Airflow has inserted rows into traffic_speed_latest, and whether road_segments has been loaded."
            )
        else:
            view_state = pdk.ViewState(latitude=1.3521, longitude=103.8198, zoom=11, pitch=0)
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
                    "html": "<b>{road_name}</b><br/>Link ID: {link_id}<br/>Road category: {road_category}<br/>Speed band: {speed_band}<br/>Estimated speed: {avg_speed_label} km/h",
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
                <div class="legend-title">Congestion Level</div>
                <div class="legend-flex">
                    <div class="legend-gradient"></div>
                    <div class="legend-labels">
                        <div>Fast (&gt; 65 km/h)</div>
                        <div>Moderate (50–65)</div>
                        <div>Slow (35–50)</div>
                        <div>Heavy (20–35)</div>
                        <div>Severe (&le; 20 km/h)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not df.empty:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div><b>Segments loaded</b></div>
                    <div>{len(df):,}</div>
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

        # ---------- DASHBOARD SECTION ----------
        st.markdown("## Traffic Data Insights")

        history_df, _ = load_traffic_data("history")

        if history_df.empty:
            st.info("No historical data available for EDA.")
            return

        # create derived feature
        history_df["hour"] = history_df["interval_start"].dt.hour

        # ---------- LAYOUT: 2 PANELS ----------
        col1, col2 = st.columns(2, gap="large")

        # ===== LEFT PANEL: Distribution =====
        with col1:
            st.markdown("### Speed Distribution")

            fig1, ax1 = plt.subplots()
            ax1.hist(history_df["avg_speed_band"].dropna(), bins=40)
            ax1.set_xlabel("Speed Band")
            ax1.set_ylabel("Frequency")
            ax1.set_facecolor("#0b1220")
            st.pyplot(fig1)

        # ===== RIGHT PANEL: Time Pattern =====
        with col2:
            st.markdown("### Hourly Traffic Pattern")

            hourly = history_df.groupby("hour")["avg_speed_band"].mean()

            fig2, ax2 = plt.subplots()
            ax2.plot(hourly.index, hourly.values)
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Avg Speed Band")
            ax2.set_xticks(range(0, 24, 2))
            ax2.grid(alpha=0.2)

            st.pyplot(fig2)

        # ---------- SUMMARY ROW ----------
        st.markdown("### Dataset Summary")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Total Records", len(history_df))

        with m2:
            st.metric("Avg Speed", round(history_df["avg_speed_band"].mean(), 2))

        with m3:
            st.metric("Time Range", f"{history_df['interval_start'].nunique()} intervals")

    except Exception as exc:
        st.markdown('<section class="section-wrap">', unsafe_allow_html=True)
        st.error(
            "Could not load data from Supabase. Check your secrets, environment variables, and whether the tables exist."
        )
        st.exception(exc)
        st.markdown("</section>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
