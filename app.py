import os
import pandas as pd
import streamlit as st
import pydeck as pdk
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Singapore Traffic Speed Bands", layout="wide")

# -----------------------------
# Database connection
# -----------------------------
DB_URL = os.environ["SUPABASE_DB_URI"]
engine = create_engine(DB_URL)

# -----------------------------
# Color mapping: light blue -> dark blue
# SpeedBand 1 = lightest, 8 = darkest
# -----------------------------
SPEED_BAND_COLORS = {
    1: [230, 245, 255],
    2: [200, 230, 255],
    3: [170, 215, 255],
    4: [130, 190, 255],
    5: [90, 160, 245],
    6: [50, 120, 230],
    7: [20, 80, 200],
    8: [0, 40, 130],
}

def get_color(speed_band: int):
    return SPEED_BAND_COLORS.get(int(speed_band), [180, 180, 180])

# -----------------------------
# Query latest snapshot
# -----------------------------
@st.cache_data(ttl=240)  # refresh cache every 4 minutes
def load_latest_snapshot():
    query = text("""
        WITH latest_time AS (
            SELECT MAX(collected_at) AS max_time
            FROM traffic_speed_snapshots
        )
        SELECT
            s.collected_at,
            s.link_id,
            s.speed_band,
            s.minimum_speed,
            s.maximum_speed,
            r.road_name,
            r.road_category,
            r.start_lon,
            r.start_lat,
            r.end_lon,
            r.end_lat
        FROM traffic_speed_snapshots s
        JOIN road_segments r
          ON s.link_id = r.link_id
        JOIN latest_time lt
          ON s.collected_at = lt.max_time
        WHERE r.start_lon IS NOT NULL
          AND r.start_lat IS NOT NULL
          AND r.end_lon IS NOT NULL
          AND r.end_lat IS NOT NULL
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df

df = load_latest_snapshot()

# -----------------------------
# Title / summary
# -----------------------------
st.title("Singapore Real-Time Traffic Speed Bands")
st.caption("Latest road-segment snapshot from LTA DataMall via Supabase")

if df.empty:
    st.warning("No data found in the latest snapshot.")
    st.stop()

latest_time = df["collected_at"].iloc[0]
st.write(f"**Latest snapshot time:** {latest_time}")

# -----------------------------
# Prepare line data for pydeck
# -----------------------------
df["start"] = df.apply(lambda row: [row["start_lon"], row["start_lat"]], axis=1)
df["end"] = df.apply(lambda row: [row["end_lon"], row["end_lat"]], axis=1)
df["color"] = df["speed_band"].apply(get_color)

# Optional width scaling
df["width"] = 3

# -----------------------------
# Sidebar filters (simple, optional)
# -----------------------------
st.sidebar.header("Filters")

road_categories = sorted(df["road_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Road Category",
    options=road_categories,
    default=road_categories
)

filtered_df = df[df["road_category"].isin(selected_categories)].copy()

# -----------------------------
# Map layer
# -----------------------------
layer = pdk.Layer(
    "LineLayer",
    data=filtered_df,
    get_source_position="start",
    get_target_position="end",
    get_color="color",
    get_width="width",
    width_min_pixels=2,
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=1.3521,
    longitude=103.8198,
    zoom=10.8,
    pitch=0,
)

tooltip = {
    "html": """
        <b>Road:</b> {road_name} <br/>
        <b>Link ID:</b> {link_id} <br/>
        <b>Road Category:</b> {road_category} <br/>
        <b>Speed Band:</b> {speed_band} <br/>
        <b>Speed Range:</b> {minimum_speed} - {maximum_speed} km/h
    """,
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light"
    ),
    use_container_width=True
)

# -----------------------------
# Legend
# -----------------------------
st.subheader("Speed Band Legend")

legend_cols = st.columns(8)
for i in range(1, 9):
    color = SPEED_BAND_COLORS[i]
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    with legend_cols[i - 1]:
        st.markdown(
            f"""
            <div style="
                background-color: {hex_color};
                padding: 18px;
                border-radius: 8px;
                text-align: center;
                color: {'white' if i >= 6 else 'black'};
                font-weight: bold;
            ">
                {i}
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("Light blue = lower speed band, dark blue = higher speed band")

# -----------------------------
# Raw table preview
# -----------------------------
with st.expander("Preview latest snapshot table"):
    st.dataframe(
        filtered_df[
            [
                "collected_at",
                "link_id",
                "road_name",
                "road_category",
                "speed_band",
                "minimum_speed",
                "maximum_speed",
            ]
        ].sort_values(by=["speed_band", "road_name"]),
        use_container_width=True
    )