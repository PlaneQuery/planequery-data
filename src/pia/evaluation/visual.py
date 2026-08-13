"""Interactive visualization of the current PIA resolver.

Run this module from the repository root to reproduce the large real-flight
case used by ``test_real_flights_0``::

    python -m pia.evaluation.visual

The output is a self-contained HTML file. Candidate edges start hidden because
showing every candidate at once obscures the selected paths at real-world
flight counts; click their legend entry to display them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path

from pia.graph import FlightLinkCandidate

# ``data_engineering.utils`` reads OUTPUT_DIR at import time. Keep this module
# directly runnable in a fresh shell, as well as under the project's tests.
os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))

import plotly.graph_objects as go
import polars as pl

from data_engineering.flights.flight_type import (
    Flight,
    FlightIndex,
    flights_df_to_flights,
    get_flights,
    pia_or_american_ladd_icao_filter,
)
from data_engineering.ground_control import load_ground_control_mappings
from data_engineering.utils import OUTPUT_DIR
from pia.resolver import (
    build_candidates,
    get_flights_connected_to_icao,
    mapping_from_selected_candidates,
    solve_links,
)


DEFAULT_START_DATE = date(2026, 5, 14)
DEFAULT_END_DATE = date(2026, 6, 10)
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "tmp" / "resolve_pia_mappings.html"

PIA_COLOR = "#2563eb"
REAL_COLOR = "#64748b"
SELECTED_EDGE_COLOR = "#0f172a"
CANDIDATE_EDGE_COLOR = "#94a3b8"


@dataclass(slots=True)
class PiaResolution:
    mapping: dict[Flight, str]
    candidates: list[FlightLinkCandidate]
    selected_candidates: list[FlightLinkCandidate]


def _resolve_with_details(flights: list[Flight]) -> PiaResolution:
    """Run the same stages as ``resolve_pia_mappings``, retaining graph data."""

    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights)
    print(f"Number of PIA flights: {len(flight_index.get_pia_flights())}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates = solve_links(flight_index, candidates)
    print(f"Number of selected candidates: {len(selected_candidates)}")
    mapping = mapping_from_selected_candidates(selected_candidates, flight_index)
    print(f"Number of mappings: {len(mapping)}")
    return PiaResolution(mapping, candidates, selected_candidates)


def _mapped_identity(flight: Flight, mapping: dict[Flight, str]) -> str:
    if not flight.pia:
        return flight.icao
    return mapping.get(flight, f"unresolved: {flight.icao}")


def _flight_hover(flight: Flight, mapping: dict[Flight, str]) -> str:
    identity = _mapped_identity(flight, mapping)
    kind = "PIA" if flight.pia else "Real ICAO"
    mapped_line = (
        f"<br>Resolved identity: <b>{identity}</b>" if flight.pia else ""
    )
    callsign_line = f"<br>Callsign: {flight.callsign}" if flight.callsign else ""
    category_line = f"<br>Category: {flight.category}" if flight.category else ""
    return (
        f"<b>{kind} {flight.icao}</b>{mapped_line}"
        f"<br>{flight.takeoff_airport_ident} → {flight.landing_airport_ident}"
        f"<br>{flight.takeoff_time:%Y-%m-%d %H:%M} → "
        f"{flight.landing_time:%Y-%m-%d %H:%M}"
        f"{callsign_line}{category_line}"
        f"<br>Flight ID: {flight.flight_id}"
    )


def _edge_hover(
    candidate: FlightLinkCandidate,
    flights_by_id: dict[str, Flight],
    *,
    selected: bool,
) -> str:
    previous = flights_by_id[candidate.previous_flight_id]
    next_flight = flights_by_id[candidate.next_flight_id]
    gap = next_flight.takeoff_time - previous.landing_time
    total_minutes = int(gap.total_seconds() // 60)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remaining_minutes, 60)
    gap_parts = []
    if days:
        gap_parts.append(f"{days}d")
    if hours:
        gap_parts.append(f"{hours}h")
    gap_parts.append(f"{minutes}m")
    status = "Selected" if selected else "Candidate"
    return (
        f"<b>{status} continuity link</b>"
        f"<br>{previous.icao}: {previous.landing_airport_ident} "
        f"{previous.landing_time:%Y-%m-%d %H:%M}"
        f"<br>→ {next_flight.icao}: {next_flight.takeoff_airport_ident} "
        f"{next_flight.takeoff_time:%Y-%m-%d %H:%M}"
        f"<br>Ground gap: {' '.join(gap_parts)}"
        f"<br>Score: {candidate.score}"
    )


def _midpoint(a: datetime, b: datetime) -> datetime:
    return a + (b - a) / 2


def _continuity_lanes(
    flights: list[Flight],
    resolution: PiaResolution,
) -> dict[str, str]:
    """Assign each selected path to a stable, human-readable timeline row."""

    parent = {flight.flight_id: flight.flight_id for flight in flights}

    def find(flight_id: str) -> str:
        while parent[flight_id] != flight_id:
            parent[flight_id] = parent[parent[flight_id]]
            flight_id = parent[flight_id]
        return flight_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for candidate in resolution.selected_candidates:
        union(candidate.previous_flight_id, candidate.next_flight_id)

    components: dict[str, list[Flight]] = {}
    for flight in flights:
        components.setdefault(find(flight.flight_id), []).append(flight)

    lanes = {}
    duplicate_counts: dict[str, int] = {}
    ordered_components = sorted(
        components.values(),
        key=lambda component: min(flight.takeoff_time for flight in component),
    )
    for component in ordered_components:
        real_icaos = sorted({flight.icao for flight in component if not flight.pia})
        resolved_icaos = sorted(
            {
                resolution.mapping[flight]
                for flight in component
                if flight in resolution.mapping
            }
        )
        pia_icaos = sorted({flight.icao for flight in component if flight.pia})
        if real_icaos:
            base_label = " + ".join(real_icaos)
        elif resolved_icaos:
            base_label = " + ".join(resolved_icaos)
        elif pia_icaos:
            base_label = f"unresolved: {' + '.join(pia_icaos)}"
        else:
            base_label = "unresolved"

        duplicate_counts[base_label] = duplicate_counts.get(base_label, 0) + 1
        duplicate_number = duplicate_counts[base_label]
        label = (
            base_label
            if duplicate_number == 1
            else f"{base_label} (path {duplicate_number})"
        )
        for flight in component:
            lanes[flight.flight_id] = label
    return lanes


def _add_flight_traces(
    figure: go.Figure,
    flights: list[Flight],
    mapping: dict[Flight, str],
    lanes: dict[str, str],
) -> None:
    for pia, name, color in (
        (False, "Real ICAO flight", REAL_COLOR),
        (True, "PIA flight", PIA_COLOR),
    ):
        matching_flights = sorted(
            (flight for flight in flights if flight.pia is pia),
            key=lambda flight: flight.takeoff_time,
        )
        line_x: list[datetime | None] = []
        line_y: list[str | None] = []
        midpoint_x = []
        midpoint_y = []
        hover_text = []
        for flight in matching_flights:
            lane = lanes[flight.flight_id]
            line_x.extend([flight.takeoff_time, flight.landing_time, None])
            line_y.extend([lane, lane, None])
            midpoint_x.append(_midpoint(flight.takeoff_time, flight.landing_time))
            midpoint_y.append(lane)
            hover_text.append(_flight_hover(flight, mapping))

        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": color, "width": 12},
                name=name,
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        # A transparent midpoint gives every flight a generous, reliable hover
        # target even when its duration is short relative to the date range.
        figure.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                marker={"color": color, "size": 16, "opacity": 0.01},
                name=f"{name} details",
                legendgroup=name,
                showlegend=False,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )


def _add_edge_trace(
    figure: go.Figure,
    candidates: list[FlightLinkCandidate],
    flights_by_id: dict[str, Flight],
    mapping: dict[Flight, str],
    lanes: dict[str, str],
    *,
    selected: bool,
) -> None:
    line_x: list[datetime | None] = []
    line_y: list[str | None] = []
    midpoint_x = []
    midpoint_y = []
    hover_text = []
    for candidate in candidates:
        previous = flights_by_id[candidate.previous_flight_id]
        next_flight = flights_by_id[candidate.next_flight_id]
        previous_lane = lanes[previous.flight_id]
        next_lane = lanes[next_flight.flight_id]
        line_x.extend([previous.landing_time, next_flight.takeoff_time, None])
        line_y.extend([previous_lane, next_lane, None])
        midpoint_x.append(_midpoint(previous.landing_time, next_flight.takeoff_time))
        # Selected links should normally stay on one identity lane. For a
        # cross-lane candidate, attaching hover to the destination is clearer
        # than inventing a categorical y-coordinate between two lanes.
        midpoint_y.append(next_lane)
        hover_text.append(
            _edge_hover(candidate, flights_by_id, selected=selected)
        )

    name = "Selected continuity links" if selected else "Other candidate links"
    color = SELECTED_EDGE_COLOR if selected else CANDIDATE_EDGE_COLOR
    visibility: bool | str = True if selected else "legendonly"
    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={
                "color": color,
                "width": 2.2 if selected else 1,
                "dash": "solid" if selected else "dot",
            },
            opacity=0.9 if selected else 0.35,
            name=name,
            legendgroup=name,
            visible=visibility,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=midpoint_x,
            y=midpoint_y,
            mode="markers",
            marker={"color": color, "size": 12, "opacity": 0.01},
            name=f"{name} details",
            legendgroup=name,
            showlegend=False,
            visible=visibility,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )


def build_resolution_figure(
    flights: list[Flight],
    resolution: PiaResolution,
) -> go.Figure:
    """Build a scalable interactive timeline of a PIA resolution result."""

    flights_by_id = {flight.flight_id: flight for flight in flights}
    flight_lanes = _continuity_lanes(flights, resolution)
    selected_candidate_set = set(resolution.selected_candidates)
    other_candidates = [
        candidate
        for candidate in resolution.candidates
        if candidate not in selected_candidate_set
    ]
    lane_first_takeoff: dict[str, datetime] = {}
    lane_counts: dict[str, int] = {}
    for flight in flights:
        lane = flight_lanes[flight.flight_id]
        lane_first_takeoff[lane] = min(
            lane_first_takeoff.get(lane, flight.takeoff_time),
            flight.takeoff_time,
        )
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    lanes = sorted(lane_counts, key=lambda lane: (lane_first_takeoff[lane], lane))

    figure = go.Figure()
    # Draw edges before flights so each flight bar remains visually dominant.
    _add_edge_trace(
        figure,
        other_candidates,
        flights_by_id,
        resolution.mapping,
        flight_lanes,
        selected=False,
    )
    _add_edge_trace(
        figure,
        resolution.selected_candidates,
        flights_by_id,
        resolution.mapping,
        flight_lanes,
        selected=True,
    )
    _add_flight_traces(figure, flights, resolution.mapping, flight_lanes)

    tick_text = [f"{lane}  ·  {lane_counts[lane]} flights" for lane in lanes]
    figure.update_layout(
        title={
            "text": (
                "Resolved PIA flight continuity"
                f"<br><sup>{len(flights)} flights · "
                f"{len(resolution.selected_candidates)} selected links · "
                f"{len(resolution.candidates)} candidates · "
                f"{len(lanes)} identity lanes</sup>"
            ),
            "x": 0.01,
        },
        template="plotly_white",
        height=max(700, 250 + 46 * len(lanes)),
        margin={"l": 190, "r": 40, "t": 115, "b": 90},
        hovermode="closest",
        hoverlabel={"align": "left", "font_size": 13},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
            "groupclick": "togglegroup",
        },
    )
    figure.update_xaxes(
        title_text="Flight time (drag to zoom; double-click to reset)",
        rangeslider={"visible": True, "thickness": 0.06},
        showgrid=True,
        gridcolor="#e2e8f0",
    )
    figure.update_yaxes(
        title_text="Resolved aircraft identity",
        categoryorder="array",
        categoryarray=list(reversed(lanes)),
        tickmode="array",
        tickvals=lanes,
        ticktext=tick_text,
        automargin=True,
        showgrid=True,
        gridcolor="#f1f5f9",
    )
    return figure


def visualize_resolve_pia_mappings(
    flights: list[Flight],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[Flight, str], Path]:
    """Run the resolver and write its complete interactive flight graph."""

    resolution = _resolve_with_details(flights)
    figure = build_resolution_figure(flights, resolution)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".html":
        raise ValueError("The interactive visualization output must be an .html file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
    return resolution.mapping, output_path


def load_real_flight_case(
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
) -> list[Flight]:
    """Load the same bounded real-flight set as ``test_real_flights_0``."""

    ground_control = (
        load_ground_control_mappings()
        .filter((pl.col("date") > start_date) & (pl.col("date") < end_date))
        .sort("date")
    )
    pia_to_real_icao = dict(
        zip(ground_control["pia_icao"], ground_control["real_icao"])
    )
    relevant_icaos = set(pia_to_real_icao) | set(pia_to_real_icao.values())
    frame = (
        get_flights(start_date, end_date)
        .filter(pia_or_american_ladd_icao_filter())
        .filter(pl.col("icao").str.len_chars() == 6)
        .filter(pl.col("icao").is_in(relevant_icaos))
    )
    return flights_df_to_flights(frame)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and interactively visualize PIA flight mappings."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help=f"Flight window start (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help=f"Flight window end (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--icao",
        help=(
            "Render only flights in candidate-graph components connected to "
            "this ICAO (for example: aa699e)"
        ),
    )
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    flights = load_real_flight_case(args.start_date, args.end_date)
    if args.icao is not None:
        df_flights = get_flights(args.start_date, args.end_date)
        df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
        flights = flights_df_to_flights(df_flights)
        flight_index = FlightIndex(flights)
        candidates = build_candidates(flight_index)
        connected_flights = get_flights_connected_to_icao(
            candidates,
            flight_index,
            args.icao,
        )
        if not connected_flights:
            raise ValueError(
                f"ICAO {args.icao!r} has no flights in the selected date range"
            )
        flights = connected_flights
        print(
            f"Rendering {len(flights)} flights connected to ICAO "
            f"{args.icao.casefold()} in the candidate graph"
        )
    _, output_path = visualize_resolve_pia_mappings(flights, args.output)
    print(f"Interactive graph written to {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    main()
