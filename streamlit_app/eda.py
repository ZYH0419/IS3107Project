import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(engine):
    query = "SELECT * FROM traffic_speed_15min"
    df = pd.read_sql(query, engine)
    return df

def plot_speed_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df["speed"], bins=50)
    ax.set_title("Traffic Speed Distribution")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Frequency")
    return fig

def plot_hourly_pattern(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    hourly = df.groupby("hour")["speed"].mean()

    fig, ax = plt.subplots()
    ax.plot(hourly.index, hourly.values)
    ax.set_title("Average Speed by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Speed")
    return fig

def top_congested_roads(df):
    congestion = df.groupby("road_segment_id")["speed"].mean().sort_values()

    return congestion.head(10)

def missing_analysis(df):
    return df.isnull().mean()
