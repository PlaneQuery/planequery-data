from __future__ import annotations

from dataclasses import dataclass

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl


DEFAULT_DOWNSAMPLE_SECONDS = 5
DEFAULT_GAP_BREAK_SECONDS = 60
DEFAULT_ALTITUDE_COLUMNS = (
    "baro_altitude_ft",
    "baroAltitude",
    "baro_altitude",
    "altitude",
)
DEFAULT_GRAPH_COLUMNS = (
    "baro_altitude_ft",
    "ground_speed_kt",
    "track_deg",
    "on_ground",
    "time_delta_s",
    "callsign_changed_from_prev",
    "above_ground_level_closest_airport",
)
BOOLEAN_GRAPH_COLUMNS = frozenset(("on_ground", "callsign_changed_from_prev"))


@dataclass(frozen=True)
class FlightAdsbVisualization:
    flights: pl.DataFrame
    altitude_figure: go.Figure
    map_figure: go.Figure | None = None


def downsample_adsb(adsb: pl.DataFrame, seconds: int = DEFAULT_DOWNSAMPLE_SECONDS) -> pl.DataFrame:
    if adsb.is_empty() or seconds <= 0:
        return adsb.sort("time") if "time" in adsb.columns else adsb

    return (
        adsb.with_columns((pl.col("time").dt.epoch("s") // seconds).alias("_bucket"))
        .unique(subset="_bucket", keep="first")
        .drop("_bucket")
        .sort("time")
    )


def insert_gap_breaks(
    df: pl.DataFrame,
    altitude_column: str,
    gap_seconds: int = DEFAULT_GAP_BREAK_SECONDS,
) -> tuple[list, list]:
    times = df["time"].to_list()
    altitudes = df[altitude_column].to_list()
    if not times:
        return times, altitudes

    out_times = [times[0]]
    out_altitudes = [altitudes[0]]
    for idx in range(1, len(times)):
        if (times[idx] - times[idx - 1]).total_seconds() > gap_seconds:
            out_times.append(None)
            out_altitudes.append(None)
        out_times.append(times[idx])
        out_altitudes.append(altitudes[idx])
    return out_times, out_altitudes


def visualize_flight_with_adsb(
    flights: pl.DataFrame,
    adsb: pl.DataFrame,
    *,
    icao: str | None = None,
    include_map: bool = False,
    downsample_seconds: int = DEFAULT_DOWNSAMPLE_SECONDS,
    gap_break_seconds: int = DEFAULT_GAP_BREAK_SECONDS,
    altitude_column: str | None = None,
) -> FlightAdsbVisualization:
    """Build altitude and optional map figures for one aircraft's flights and ADS-B points."""
    flights_i = _filter_single_icao(flights, icao, time_column="takeoff_time")
    adsb_i = _filter_single_icao(adsb, icao, time_column="time")
    icao_title = icao or _single_icao_value(flights_i, adsb_i) or "single aircraft"
    altitude_column = altitude_column or _first_existing_column(adsb_i, DEFAULT_ALTITUDE_COLUMNS)

    _require_columns(adsb_i, ["time", altitude_column], "adsb")
    _require_columns(flights_i, ["takeoff_time", "landing_time"], "flights")

    adsb_ds = downsample_adsb(adsb_i, downsample_seconds)
    graph_columns = _graph_columns(adsb_ds, altitude_column)
    adsb_graph = _prepare_adsb_for_graph(adsb_ds, graph_columns)
    row_titles = graph_columns
    altitude_figure = make_subplots(
        rows=len(row_titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=row_titles,
        row_heights=_row_heights(row_titles),
    )

    for row_idx, column in enumerate(graph_columns, start=1):
        graph_times, values = insert_gap_breaks(adsb_graph, column, gap_break_seconds)
        _add_graph_trace(
            altitude_figure,
            column=column,
            times=graph_times,
            values=values,
            row=row_idx,
        )

    shades = (
        "rgba(0,150,0,0.12)",
        "rgba(0,0,200,0.12)",
        "rgba(200,100,0,0.12)",
        "rgba(150,0,150,0.12)",
        "rgba(0,150,150,0.12)",
        "rgba(150,150,0,0.12)",
    )
    for idx, row in enumerate(flights_i.iter_rows(named=True)):
        altitude_figure.add_vrect(
            x0=row["takeoff_time"],
            x1=row["landing_time"],
            fillcolor=shades[idx % len(shades)],
            line_width=0,
            annotation_text=f"#{idx + 1}",
            annotation_position="top left",
            row="all",
            col=1,
        )
        altitude_figure.add_vline(
            x=row["takeoff_time"],
            line={"color": "green", "width": 1, "dash": "dash"},
            row="all",
            col=1,
        )
        altitude_figure.add_vline(
            x=row["landing_time"],
            line={"color": "red", "width": 1, "dash": "dash"},
            row="all",
            col=1,
        )

    altitude_figure.update_layout(
        height=350 + 180 * (len(graph_columns) - 1),
        margin={"l": 40, "r": 10, "t": 30, "b": 30},
        title=f"{icao_title} - {len(flights_i)} flight(s), {len(adsb_i)} pts (showing {len(adsb_ds)})",
    )
    altitude_figure.update_xaxes(title_text="time", row=len(row_titles), col=1)
    for row_idx, title in enumerate(row_titles, start=1):
        altitude_figure.update_yaxes(title_text=title, row=row_idx, col=1)
        if title in BOOLEAN_GRAPH_COLUMNS:
            altitude_figure.update_yaxes(
                range=[-0.15, 1.15],
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["false", "true"],
                showgrid=True,
                row=row_idx,
                col=1,
            )

    map_figure = _create_map_figure(adsb_ds, altitude_column) if include_map else None
    return FlightAdsbVisualization(flights=flights_i, altitude_figure=altitude_figure, map_figure=map_figure)


def _create_map_figure(adsb: pl.DataFrame, altitude_column: str) -> go.Figure | None:
    _require_columns(adsb, ["lat", "lon"], "adsb")
    adsb_points = adsb.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    if adsb_points.is_empty():
        return None

    map_figure = px.scatter_map(
        adsb_points.to_pandas(),
        lat="lat",
        lon="lon",
        color=altitude_column,
        zoom=3,
        height=400,
        color_continuous_scale="Viridis",
    )
    map_figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return map_figure


def _graph_columns(adsb: pl.DataFrame, altitude_column: str) -> list[str]:
    columns = [column for column in DEFAULT_GRAPH_COLUMNS if column in adsb.columns]
    if altitude_column not in columns:
        columns.insert(0, altitude_column)
    return columns


def _row_heights(columns: list[str]) -> list[float]:
    return [0.45 if column in BOOLEAN_GRAPH_COLUMNS else 1.0 for column in columns]


def _add_graph_trace(
    figure: go.Figure,
    *,
    column: str,
    times: list,
    values: list,
    row: int,
) -> None:
    if column in BOOLEAN_GRAPH_COLUMNS:
        figure.add_scatter(
            x=times,
            y=values,
            mode="lines+markers",
            line={"shape": "hv", "width": 2},
            marker={"size": 5, "symbol": "square"},
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.18)",
            name=column,
            customdata=_boolean_hover_values(values),
            hovertemplate=f"{column}: %{{customdata}}<br>time: %{{x}}<extra></extra>",
            connectgaps=False,
            row=row,
            col=1,
        )
        return

    figure.add_scatter(
        x=times,
        y=values,
        mode="lines+markers",
        marker={"size": 3},
        name=column,
        hovertemplate=f"{column}: %{{y}}<br>time: %{{x}}<extra></extra>",
        connectgaps=False,
        row=row,
        col=1,
    )


def _boolean_hover_values(values: list) -> list[str | None]:
    return [None if value is None else str(bool(value)).lower() for value in values]


def _prepare_adsb_for_graph(adsb: pl.DataFrame, graph_columns: list[str]) -> pl.DataFrame:
    boolean_columns = [column for column in graph_columns if column in BOOLEAN_GRAPH_COLUMNS]
    if not boolean_columns:
        return adsb

    return adsb.with_columns(pl.col(column).cast(pl.Int8).alias(column) for column in boolean_columns)


def _filter_single_icao(df: pl.DataFrame, icao: str | None, *, time_column: str) -> pl.DataFrame:
    if icao is not None and "icao" in df.columns:
        df = df.filter(pl.col("icao") == icao)
    if time_column in df.columns:
        df = df.sort(time_column)
    return df


def _single_icao_value(*dfs: pl.DataFrame) -> str | None:
    values = set()
    for df in dfs:
        if "icao" not in df.columns:
            continue
        values.update(df.get_column("icao").drop_nulls().unique().to_list())
    return next(iter(values)) if len(values) == 1 else None


def _first_existing_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"adsb must contain one altitude column from: {', '.join(candidates)}")


def _require_columns(df: pl.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")
