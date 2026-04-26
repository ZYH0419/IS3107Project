from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATASET_PATH = Path(__file__).resolve().parent.parent / "traffic_speed_bands_full.xlsx"
CHART_COLORS = {
    "severe": "#b91c1c",
    "slow": "#f97316",
    "moderate": "#facc15",
    "fast": "#86efac",
    "neutral": "#94a3b8",
    "accent": "#38bdf8",
}


def _to_snake_case(value: object) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _pick_column(
    columns: list[str],
    exact_names: tuple[str, ...] = (),
    keyword_groups: tuple[tuple[str, ...], ...] = (),
) -> Optional[str]:
    for exact_name in exact_names:
        if exact_name in columns:
            return exact_name

    best_column: Optional[str] = None
    best_score = 0

    for column in columns:
        score = 0
        for group in keyword_groups:
            if all(keyword in column for keyword in group):
                score += len(group)
        if score > best_score:
            best_score = score
            best_column = column

    return best_column if best_score > 0 else None


def _detect_schema(df: pd.DataFrame) -> dict[str, Optional[str]]:
    columns = list(df.columns)

    schema = {
        "timestamp": _pick_column(
            columns,
            exact_names=("collected_at", "timestamp", "datetime", "date_time", "recorded_at"),
            keyword_groups=(
                ("collect", "time"),
                ("update", "time"),
                ("record", "time"),
                ("date",),
                ("time",),
            ),
        ),
        "link_id": _pick_column(
            columns,
            exact_names=("link_id", "segment_id", "road_segment_id"),
            keyword_groups=(("link", "id"), ("segment", "id")),
        ),
        "road_name": _pick_column(
            columns,
            exact_names=("road_name", "street_name"),
            keyword_groups=(("road", "name"), ("street", "name"), ("road",)),
        ),
        "region": _pick_column(
            columns,
            exact_names=("region", "district", "planning_area", "zone", "area"),
            keyword_groups=(
                ("planning", "area"),
                ("district",),
                ("region",),
                ("zone",),
                ("area",),
            ),
        ),
        "road_category": _pick_column(
            columns,
            exact_names=("road_category", "category"),
            keyword_groups=(("road", "category"), ("category",)),
        ),
        "speed_band": _pick_column(
            columns,
            exact_names=("speed_band",),
            keyword_groups=(("speed", "band"),),
        ),
        "minimum_speed": _pick_column(
            columns,
            exact_names=("minimum_speed", "min_speed"),
            keyword_groups=(("minimum", "speed"), ("min", "speed")),
        ),
        "maximum_speed": _pick_column(
            columns,
            exact_names=("maximum_speed", "max_speed"),
            keyword_groups=(("maximum", "speed"), ("max", "speed")),
        ),
        "speed": _pick_column(
            columns,
            exact_names=("speed", "avg_speed", "average_speed"),
            keyword_groups=(("average", "speed"), ("avg", "speed"), ("speed",)),
        ),
    }

    return schema


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{float(value):,.1f}"
    return str(value)


def _render_metric_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div><b>{label}</b></div>
                    <div style="font-size: 1.35rem; margin-top: 0.3rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _apply_chart_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.92)",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def _safe_group_label(df: pd.DataFrame, schema: dict[str, Optional[str]]) -> tuple[str, pd.Series]:
    if schema["road_name"] and schema["road_name"] in df.columns:
        labels = df[schema["road_name"]].fillna("Unknown road").astype(str)
        return "Road", labels

    if schema["link_id"] and schema["link_id"] in df.columns:
        labels = "Link " + df[schema["link_id"]].fillna("Unknown").astype(str)
        return "Road segment", labels

    labels = pd.Series("Record", index=df.index, dtype="object")
    return "Record", labels


def _build_congestion_features(df: pd.DataFrame, schema: dict[str, Optional[str]]) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    note = "Congestion could not be computed because the dataset does not contain usable speed information."

    if schema["speed_band"] and schema["speed_band"] in df.columns:
        speed_band = pd.to_numeric(df[schema["speed_band"]], errors="coerce")
        speed_band = speed_band.where(speed_band.between(1, 8))
        df["speed_band_numeric"] = speed_band

        # Lower speed bands represent slower movement, so invert the scale so
        # a higher congestion_score means more severe congestion.
        df["congestion_score"] = 9 - speed_band
        note = "Congestion score is derived from Speed Band using `9 - speed_band`, so Band 1 is most congested and Band 8 is least congested."
        return df, note

    speed_series: Optional[pd.Series] = None
    if schema["speed"] and schema["speed"] in df.columns:
        speed_series = pd.to_numeric(df[schema["speed"]], errors="coerce")
    elif schema["minimum_speed"] and schema["maximum_speed"] and schema["minimum_speed"] in df.columns and schema["maximum_speed"] in df.columns:
        minimum_speed = pd.to_numeric(df[schema["minimum_speed"]], errors="coerce")
        maximum_speed = pd.to_numeric(df[schema["maximum_speed"]], errors="coerce")
        speed_series = pd.concat([minimum_speed, maximum_speed], axis=1).mean(axis=1, skipna=True)

    if speed_series is not None:
        observed_upper = speed_series.quantile(0.95)
        if pd.notna(observed_upper) and observed_upper > 0:
            df["avg_speed"] = speed_series
            normalized_speed = speed_series.clip(lower=0, upper=observed_upper) / observed_upper
            df["congestion_score"] = (1 - normalized_speed) * 8
            note = "Congestion score is approximated by inverting observed speed relative to the dataset's 95th percentile speed."

    return df, note


@st.cache_data(show_spinner=False)
def load_analysis_dataset() -> tuple[pd.DataFrame, dict[str, Optional[str]], str]:
    df = pd.read_excel(DATASET_PATH, engine="openpyxl")
    df = df.copy()
    df.columns = [_to_snake_case(column) for column in df.columns]

    schema = _detect_schema(df)

    numeric_candidates = {
        value
        for value in schema.values()
        if value is not None and value in df.columns
    }
    numeric_candidates.update(
        column
        for column in df.columns
        if any(token in column for token in ("lat", "lon", "speed", "band", "category", "count"))
    )
    for column in sorted(numeric_candidates):
        try:
            df[column] = pd.to_numeric(df[column])
        except (TypeError, ValueError):
            pass

    if schema["timestamp"] and schema["timestamp"] in df.columns:
        parsed_ts = pd.to_datetime(df[schema["timestamp"]], errors="coerce")
        if parsed_ts.notna().any():
            df["analysis_timestamp"] = parsed_ts
            df["analysis_date"] = parsed_ts.dt.date
            df["analysis_hour"] = parsed_ts.dt.hour
            df["analysis_weekday"] = parsed_ts.dt.day_name()
        else:
            schema["timestamp"] = None

    if schema["minimum_speed"] and schema["minimum_speed"] in df.columns:
        df[schema["minimum_speed"]] = pd.to_numeric(df[schema["minimum_speed"]], errors="coerce")
    if schema["maximum_speed"] and schema["maximum_speed"] in df.columns:
        df[schema["maximum_speed"]] = pd.to_numeric(df[schema["maximum_speed"]], errors="coerce")
    if schema["speed"] and schema["speed"] in df.columns:
        df[schema["speed"]] = pd.to_numeric(df[schema["speed"]], errors="coerce")

    if "avg_speed" not in df.columns:
        if schema["minimum_speed"] and schema["maximum_speed"] and schema["minimum_speed"] in df.columns and schema["maximum_speed"] in df.columns:
            df["avg_speed"] = df[[schema["minimum_speed"], schema["maximum_speed"]]].mean(axis=1, skipna=True)
        elif schema["speed"] and schema["speed"] in df.columns:
            df["avg_speed"] = df[schema["speed"]]

    if schema["minimum_speed"] and schema["maximum_speed"] and schema["minimum_speed"] in df.columns and schema["maximum_speed"] in df.columns:
        df["speed_range"] = pd.to_numeric(df[schema["maximum_speed"]], errors="coerce") - pd.to_numeric(df[schema["minimum_speed"]], errors="coerce")

    df, congestion_note = _build_congestion_features(df, schema)

    label_name, labels = _safe_group_label(df, schema)
    df["segment_label"] = labels
    df["segment_label_name"] = label_name
    df["record_count"] = 1

    if schema["region"] and schema["region"] in df.columns:
        df[schema["region"]] = df[schema["region"]].fillna("Unknown").astype(str)

    if schema["road_category"] and schema["road_category"] in df.columns:
        df[schema["road_category"]] = df[schema["road_category"]].fillna("Unknown").astype(str)

    if schema["road_name"] and schema["road_name"] in df.columns:
        df[schema["road_name"]] = df[schema["road_name"]].fillna("Unknown road").astype(str)

    if schema["link_id"] and schema["link_id"] in df.columns:
        df[schema["link_id"]] = df[schema["link_id"]].astype(str)

    return df, schema, congestion_note


def _apply_filters(df: pd.DataFrame, schema: dict[str, Optional[str]]) -> pd.DataFrame:
    filtered = df.copy()

    st.markdown("#### Filters")
    filter_cols = st.columns(2)

    if schema["road_category"] and schema["road_category"] in filtered.columns:
        category_options = sorted(filtered[schema["road_category"]].dropna().astype(str).unique())
        selected_categories = filter_cols[0].multiselect("Road category", options=category_options)
        if selected_categories:
            filtered = filtered[filtered[schema["road_category"]].isin(selected_categories)]
    else:
        filter_cols[0].caption("Road category filter unavailable.")

    if schema["speed_band"] and schema["speed_band"] in filtered.columns:
        band_series = pd.to_numeric(filtered[schema["speed_band"]], errors="coerce").dropna()
        band_options = sorted({int(value) for value in band_series if 1 <= value <= 8})
        selected_bands = filter_cols[1].multiselect("Speed band", options=band_options)
        if selected_bands:
            filtered = filtered[pd.to_numeric(filtered[schema["speed_band"]], errors="coerce").isin(selected_bands)]
    else:
        filter_cols[1].caption("Speed band filter unavailable.")

    if schema["road_name"] and schema["road_name"] in filtered.columns:
        road_query = st.text_input("Road name contains", placeholder="Filter by road name")
        if road_query.strip():
            filtered = filtered[
                filtered[schema["road_name"]].str.contains(road_query.strip(), case=False, na=False)
            ]

    return filtered


def _render_dataset_overview(df: pd.DataFrame, filtered: pd.DataFrame, schema: dict[str, Optional[str]]) -> None:
    st.markdown("#### Dataset Overview")

    date_range_label = "Not available"
    if "analysis_timestamp" in filtered.columns and filtered["analysis_timestamp"].notna().any():
        min_ts = filtered["analysis_timestamp"].min()
        max_ts = filtered["analysis_timestamp"].max()
        date_range_label = f"{min_ts:%d %b %Y} to {max_ts:%d %b %Y}"

    unique_entity_label = "Unique segments"
    unique_entity_value = len(filtered)
    if schema["road_name"] and schema["road_name"] in filtered.columns:
        unique_entity_label = "Unique roads"
        unique_entity_value = filtered[schema["road_name"]].nunique(dropna=True)
    elif schema["link_id"] and schema["link_id"] in filtered.columns:
        unique_entity_label = "Unique links"
        unique_entity_value = filtered[schema["link_id"]].nunique(dropna=True)

    _render_metric_cards(
        [
            ("Total rows", _format_number(len(df))),
            ("Filtered rows", _format_number(len(filtered))),
            ("Total columns", _format_number(df.shape[1])),
            (unique_entity_label, _format_number(unique_entity_value)),
            ("Date range", date_range_label),
        ]
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        missing_summary = filtered.isna().sum().sort_values(ascending=False)
        missing_summary = missing_summary[missing_summary > 0]
        if missing_summary.empty:
            st.success("No missing values remain in the current filtered view.")
        else:
            missing_df = missing_summary.reset_index()
            missing_df.columns = ["column", "missing_values"]
            missing_df["missing_pct"] = (missing_df["missing_values"] / len(filtered) * 100).round(2)
            st.dataframe(missing_df.head(10), width="stretch", hide_index=True)

    with right:
        preview_columns = [
            column
            for column in [
                schema.get("link_id"),
                schema.get("road_name"),
                schema.get("road_category"),
                schema.get("speed_band"),
                schema.get("minimum_speed"),
                schema.get("maximum_speed"),
            ]
            if column and column in filtered.columns
        ]
        preview_columns = list(dict.fromkeys(preview_columns))[:6]
        preview_df = filtered[preview_columns].head(8) if preview_columns else filtered.head(8)
        st.dataframe(preview_df, width="stretch", hide_index=True)


def _render_distribution_analysis(filtered: pd.DataFrame, schema: dict[str, Optional[str]]) -> None:
    st.markdown("#### Distribution Analysis")

    left, right = st.columns(2, gap="large")

    with left:
        if schema["speed_band"] and schema["speed_band"] in filtered.columns:
            speed_band_df = (
                pd.to_numeric(filtered[schema["speed_band"]], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .rename_axis("speed_band")
                .reset_index(name="records")
            )
            if not speed_band_df.empty:
                fig = px.bar(
                    speed_band_df,
                    x="speed_band",
                    y="records",
                    title="Speed Band Distribution",
                    color="speed_band",
                    color_continuous_scale=[
                        CHART_COLORS["severe"],
                        CHART_COLORS["slow"],
                        CHART_COLORS["moderate"],
                        CHART_COLORS["fast"],
                    ],
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_apply_chart_style(fig), width="stretch")
            else:
                st.info("No valid speed band values are available after filtering.")
        else:
            st.info("Speed band distribution is unavailable because the dataset has no speed band column.")

    with right:
        if "congestion_score" in filtered.columns and filtered["congestion_score"].notna().any():
            fig = px.histogram(
                filtered,
                x="congestion_score",
                nbins=16,
                title="Congestion Score Distribution",
                color_discrete_sequence=[CHART_COLORS["slow"]],
            )
            st.plotly_chart(_apply_chart_style(fig), width="stretch")
        else:
            st.info("Congestion score distribution is unavailable because no congestion score could be derived.")

    if (
        "analysis_hour" in filtered.columns
        and filtered["analysis_hour"].notna().any()
        and "congestion_score" in filtered.columns
        and filtered["congestion_score"].notna().any()
    ):
        time_left, time_right = st.columns(2, gap="large")

        with time_left:
            hourly = (
                filtered.dropna(subset=["analysis_hour"])
                .groupby("analysis_hour", as_index=False)["congestion_score"]
                .mean()
            )
            fig = px.line(
                hourly,
                x="analysis_hour",
                y="congestion_score",
                title="Average Congestion by Hour",
                markers=True,
            )
            fig.update_traces(line_color=CHART_COLORS["accent"])
            st.plotly_chart(_apply_chart_style(fig), width="stretch")

        with time_right:
            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            weekday = (
                filtered.dropna(subset=["analysis_weekday"])
                .groupby("analysis_weekday", as_index=False)["congestion_score"]
                .mean()
            )
            weekday["analysis_weekday"] = pd.Categorical(
                weekday["analysis_weekday"], categories=weekday_order, ordered=True
            )
            weekday = weekday.sort_values("analysis_weekday")
            fig = px.bar(
                weekday,
                x="analysis_weekday",
                y="congestion_score",
                title="Average Congestion by Weekday",
                color_discrete_sequence=[CHART_COLORS["moderate"]],
            )
            st.plotly_chart(_apply_chart_style(fig), width="stretch")




def _render_temporal_patterns(filtered: pd.DataFrame) -> None:

    if "analysis_timestamp" not in filtered.columns or filtered["analysis_timestamp"].notna().sum() == 0:
        return

    if "congestion_score" not in filtered.columns or filtered["congestion_score"].notna().sum() == 0:
        return

    trend_granularity = st.selectbox(
        "Trend granularity",
        options=["Daily", "Weekly", "Monthly"],
        index=0,
    )

    series = filtered.dropna(subset=["analysis_timestamp", "congestion_score"]).copy()
    if series.empty:
        st.info("There are no timestamped congestion rows available for trend analysis.")
        return

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    trend = (
        series.set_index("analysis_timestamp")
        .resample(freq_map[trend_granularity])["congestion_score"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="analysis_timestamp",
        y="congestion_score",
        title=f"Congestion Trend Over Time ({trend_granularity})",
        markers=True,
    )
    fig.update_traces(line_color=CHART_COLORS["accent"])
    st.plotly_chart(_apply_chart_style(fig, height=380), width="stretch")


def _render_spatial_patterns(filtered: pd.DataFrame, schema: dict[str, Optional[str]]) -> None:
    st.markdown("#### Spatial / Segment-Based Patterns")

    label_name = filtered["segment_label_name"].iloc[0] if not filtered.empty else "Segment"
    group_col = "segment_label"

    aggregations = {"record_count": "sum"}
    if "congestion_score" in filtered.columns:
        aggregations["congestion_score"] = "mean"
    if "avg_speed" in filtered.columns:
        aggregations["avg_speed"] = "mean"

    grouped = filtered.groupby(group_col, as_index=False).agg(aggregations)
    if grouped.empty:
        st.info("No grouped segment data is available for spatial analysis.")
        return

    left, right = st.columns(2, gap="large")

    with left:
        if "congestion_score" in grouped.columns:
            top_congested = grouped.nlargest(10, "congestion_score").sort_values("congestion_score")
            fig = px.bar(
                top_congested,
                x="congestion_score",
                y=group_col,
                orientation="h",
                title=f"Top 10 Most Congested {label_name}s",
                color="congestion_score",
                color_continuous_scale=[CHART_COLORS["moderate"], CHART_COLORS["severe"]],
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title=label_name)
            st.plotly_chart(_apply_chart_style(fig, height=420), width="stretch")
        else:
            st.info("Top congested segments cannot be shown because congestion score is unavailable.")

    with right:
        frequency = grouped.nlargest(10, "record_count").sort_values("record_count")
        fig = px.bar(
            frequency,
            x="record_count",
            y=group_col,
            orientation="h",
            title=f"{label_name} Frequency in Filtered Data",
            color_discrete_sequence=[CHART_COLORS["accent"]],
        )
        fig.update_layout(yaxis_title=label_name)
        st.plotly_chart(_apply_chart_style(fig, height=420), width="stretch")

    comparison_cols = st.columns(2, gap="large")

    with comparison_cols[0]:
        if "congestion_score" in grouped.columns:
            least_congested = grouped.nsmallest(10, "congestion_score").sort_values("congestion_score", ascending=False)
            fig = px.bar(
                least_congested,
                x="congestion_score",
                y=group_col,
                orientation="h",
                title=f"Top 10 Least Congested {label_name}s",
                color_discrete_sequence=[CHART_COLORS["fast"]],
            )
            fig.update_layout(yaxis_title=label_name)
            st.plotly_chart(_apply_chart_style(fig, height=420), width="stretch")
        else:
            st.info("Least congested segments cannot be shown because congestion score is unavailable.")

    with comparison_cols[1]:
        if schema["road_category"] and schema["road_category"] in filtered.columns and "congestion_score" in filtered.columns:
            category_df = (
                filtered.dropna(subset=["congestion_score"])
                .groupby(schema["road_category"], as_index=False)["congestion_score"]
                .mean()
                .sort_values("congestion_score", ascending=False)
            )
            fig = px.bar(
                category_df,
                x=schema["road_category"],
                y="congestion_score",
                title="Average Congestion by Road Category",
                color_discrete_sequence=[CHART_COLORS["slow"]],
            )
            fig.update_xaxes(title="Road category")
            st.plotly_chart(_apply_chart_style(fig, height=420), width="stretch")
        elif schema["region"] and schema["region"] in filtered.columns and "congestion_score" in filtered.columns:
            region_df = (
                filtered.dropna(subset=["congestion_score"])
                .groupby(schema["region"], as_index=False)["congestion_score"]
                .mean()
                .sort_values("congestion_score", ascending=False)
                .head(12)
            )
            fig = px.bar(
                region_df,
                x=schema["region"],
                y="congestion_score",
                title="Average Congestion by Region",
                color_discrete_sequence=[CHART_COLORS["slow"]],
            )
            st.plotly_chart(_apply_chart_style(fig, height=420), width="stretch")
        else:
            st.info("Grouped category or region comparison is unavailable for this schema.")


def _render_correlation_analysis(filtered: pd.DataFrame, schema: dict[str, Optional[str]]) -> None:
    st.markdown("#### Correlation Analysis")

    candidate_columns = [
        schema.get("road_category"),
        schema.get("speed_band"),
        schema.get("minimum_speed"),
        schema.get("maximum_speed"),
        schema.get("speed"),
        "avg_speed",
        "speed_range",
        "congestion_score",
    ]
    numeric_df = filtered[[column for column in candidate_columns if column and column in filtered.columns]].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        st.info("There are not enough numeric features in the filtered data to build a meaningful correlation matrix.")
        return

    corr = numeric_df.corr(numeric_only=True)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, CHART_COLORS["severe"]],
                [0.5, "#0f172a"],
                [1.0, CHART_COLORS["fast"]],
            ],
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    heatmap.update_layout(title="Numeric Feature Correlation Matrix")
    st.plotly_chart(_apply_chart_style(heatmap, height=460), width="stretch")


def _render_summary_insights(filtered: pd.DataFrame, schema: dict[str, Optional[str]]) -> None:
    insights: list[str] = []

    if "congestion_score" in filtered.columns and filtered["congestion_score"].notna().any():
        mean_congestion = filtered["congestion_score"].mean()
        insights.append(f"Average congestion score in the current view is `{mean_congestion:.2f}`.")

    if schema["speed_band"] and schema["speed_band"] in filtered.columns:
        speed_band_counts = pd.to_numeric(filtered[schema["speed_band"]], errors="coerce").dropna().astype(int)
        if not speed_band_counts.empty:
            dominant_band = speed_band_counts.mode().iloc[0]
            insights.append(f"The most common speed band in the filtered data is `Band {dominant_band}`.")

    if schema["road_name"] and schema["road_name"] in filtered.columns and "congestion_score" in filtered.columns:
        grouped = (
            filtered.dropna(subset=["congestion_score"])
            .groupby(schema["road_name"], as_index=False)["congestion_score"]
            .mean()
        )
        if not grouped.empty:
            top_road = grouped.sort_values("congestion_score", ascending=False).iloc[0]
            insights.append(
                f"`{top_road[schema['road_name']]}` currently has the highest average congestion score in the filtered set."
            )

    if insights:
        st.markdown("#### Key Findings")
        for insight in insights[:3]:
            st.markdown(f"- {insight}")


def render_data_analysis_section() -> None:
    st.markdown('<section class="section-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="map-title">Traffic Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="map-subtitle">Exploratory analysis, congestion correlations, and historical traffic patterns from the workbook dataset.</div>',
        unsafe_allow_html=True,
    )

    try:
        df, schema, congestion_note = load_analysis_dataset()
    except Exception as exc:
        st.error(
            "The historical analysis dataset could not be loaded. Make sure `traffic_speed_bands_full.xlsx` is present and that Excel support is installed."
        )
        st.caption(f"Technical detail: {exc}")
        st.markdown("</section>", unsafe_allow_html=True)
        return

    if df.empty:
        st.warning("The historical workbook was loaded but it does not contain any rows to analyze.")
        st.markdown("</section>", unsafe_allow_html=True)
        return

    with st.expander("How congestion is computed", expanded=False):
        st.write(congestion_note)

    filtered = _apply_filters(df, schema)
    if filtered.empty:
        st.warning("No rows match the current analysis filters. Adjust the filters to continue.")
        st.markdown("</section>", unsafe_allow_html=True)
        return

    _render_dataset_overview(df, filtered, schema)
    _render_summary_insights(filtered, schema)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_data,
        file_name="traffic_data_analysis_filtered.csv",
        mime="text/csv",
    )

    _render_distribution_analysis(filtered, schema)
    _render_temporal_patterns(filtered)
    _render_spatial_patterns(filtered, schema)
    _render_correlation_analysis(filtered, schema)

    min_speed_col = schema.get("minimum_speed")
    max_speed_col = schema.get("maximum_speed")


    st.markdown("</section>", unsafe_allow_html=True)
