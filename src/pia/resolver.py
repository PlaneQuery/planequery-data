from collections import defaultdict, deque
from datetime import timedelta
from itertools import combinations
import networkx as nx
import polars as pl
from ortools.sat.python import cp_model
from itertools import groupby
from datetime import date
from airports.airport_lookup import AirportLookup
from data_engineering.flights.flight_type import Flight, FlightIndex, flights_df_to_flights, pia_or_american_ladd_icao_filter, get_flights
from pia.graph import FlightLinkCandidate, create_graph_from_candidates
from pia.solver import resolve_links_with_margin

AIRPORT_CONTINUITY_TOLERANCE = 10
airport_lookup = AirportLookup()
def build_candidates(flight_index: FlightIndex) -> list[FlightLinkCandidate]:
    # sorted by icao, time
    # add scores to each edge. 
    candidates = []
    for icao, flights in flight_index.flights_by_icao.items():
        for i in range(len(flights)):
            flight = flights[i]
            if i < len(flights) - 1:
                next_flight = flights[i + 1]
                score = edge_score(flight, next_flight)
                candidate = FlightLinkCandidate(
                    flight.flight_id,
                    next_flight.flight_id,
                    score = score
                )
                # if flight.landing_airport_ident == next_flight.takeoff_airport_ident:
                candidates.append(candidate)
            else:
                next_flight = None
            assert flight.last_lon is not None # TODO: Fix this in type
            assert flight.last_lat is not None
            if next_flight is not None and flight.landing_airport_ident == next_flight.takeoff_airport_ident:
                continue

            if i != len(flights) - 1 and next_flight is not None:
                before_time = next_flight.takeoff_time
            else:
                before_time = None
            pia_flights = flight_index.pia_flights_after(flight.landing_time, flight.last_lat, flight.last_lon, flight.landing_airport_ident, before_time = before_time)
            for pia_flight in pia_flights:
                if pia_flight.category != flight.category:
                    continue
                score = edge_score(flight, pia_flight)
                candidate = FlightLinkCandidate(
                    flight.flight_id,
                    pia_flight.flight_id,
                    score = score
                )
                candidates.append(candidate)
            if flight.pia:
                ladd_flights = flight_index.ladd_flights_after(flight.landing_time, flight.last_lat, flight.last_lon, flight.landing_airport_ident, before_time = before_time)
                for ladd_flight in ladd_flights:
                    if ladd_flight.category != flight.category:
                        continue
                    score = edge_score(flight, ladd_flight)
                    candidate = FlightLinkCandidate(
                        flight.flight_id,
                        ladd_flight.flight_id,
                        score = score
                    )
                    candidates.append(candidate)


    return candidates

def filter_candidates(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
) -> list[FlightLinkCandidate]:
    """Remove disconnected multi-flight real ICAOs and their candidate edges."""
    graph = create_graph_from_candidates(candidates, flight_index)
    flight_ids_to_remove: set[str] = set()

    for flights in flight_index.flights_by_icao.values():
        if flights[0].pia or len(flights) == 1:
            continue

        first_flight_id = flights[0].flight_id
        last_flight_id = flights[-1].flight_id
        if not nx.has_path(graph, first_flight_id, last_flight_id):
            flight_ids_to_remove.update(flight.flight_id for flight in flights)

    filtered_candidates = [
        candidate
        for candidate in candidates
        if candidate.previous_flight_id not in flight_ids_to_remove
        and candidate.next_flight_id not in flight_ids_to_remove
    ]
    removed_candidate_count = len(candidates) - len(filtered_candidates)
    print(
        f"Filtered out {removed_candidate_count} candidates and "
        f"{len(flight_ids_to_remove)} flights"
    )

    flight_index.remove_flights(flight_ids_to_remove)
    return filtered_candidates


def edge_score(a: Flight, b: Flight) -> int:
    score = 0

    if a.icao == b.icao:
        score += 2000

    if a.landing_airport_ident == b.takeoff_airport_ident:
        score += 1500
    if airport_lookup.airport_ident_is_in_us_faa_airspace(a.takeoff_airport_ident) and not airport_lookup.airport_ident_is_in_us_faa_airspace(a.landing_airport_ident):
        if a.icao == b.icao:
            score += 2000
    if a.callsign and b.callsign and a.callsign == b.callsign:
        score += 250

    score = max(-2000, min(score, 6000))
    return score

def candidate_mapping_path_exists(
    graph: nx.Graph,
    flight_index: FlightIndex,
    pia_icao: str,
    real_icao: str,
) -> bool:
    """Return whether any candidate path supports a PIA-to-real ICAO mapping."""
    pia_flight_ids = (
        flight.flight_id
        for flight in flight_index.flights_by_icao.get(pia_icao, ())
    )
    real_flight_ids = {
        flight.flight_id
        for flight in flight_index.flights_by_icao.get(real_icao, ())
    }
    if not real_flight_ids:
        return False

    for pia_flight_id in pia_flight_ids:
        if pia_flight_id not in graph:
            continue
        component = nx.node_connected_component(graph, pia_flight_id)
        if not component.isdisjoint(real_flight_ids):
            return True
    return False


def check_candidate_mapping_paths(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
    pia_to_real_icao: dict[str, str],
) -> dict[str, bool]:
    """Check each known PIA mapping against the complete candidate graph."""
    graph = create_graph_from_candidates(candidates, flight_index)
    return {
        pia_icao: candidate_mapping_path_exists(
            graph,
            flight_index,
            pia_icao,
            real_icao,
        )
        for pia_icao, real_icao in pia_to_real_icao.items()
    }


def get_flights_connected_to_icao(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
    icao: str,
) -> list[Flight]:
    """Return all flights in candidate components containing ``icao``."""
    normalized_icao = icao.casefold()
    target_flight_ids = {
        flight.flight_id
        for indexed_icao, flights in flight_index.flights_by_icao.items()
        if indexed_icao.casefold() == normalized_icao
        for flight in flights
    }
    if not target_flight_ids:
        return []

    graph = create_graph_from_candidates(candidates, flight_index)
    connected_flight_ids: set[str] = set()
    for flight_id in target_flight_ids:
        connected_flight_ids.update(nx.node_connected_component(graph, flight_id))

    return sorted(
        (
            flight_index.get_flight_by_id(flight_id)
            for flight_id in connected_flight_ids
        ),
        key=lambda flight: flight.takeoff_time,
    )

def icao_mapping_from_flight_mapping(mapping: dict[Flight, str]) -> dict[str, str]:
    return {flight.icao: real_icao for flight, real_icao in mapping.items()}

def resolve_pia_mappings(
    flights: list[Flight],
    # TODO: known_pia_users: list[str],
    airport_continuity_tolerance: float = 10,
) -> tuple[dict[Flight, str], pl.DataFrame]:
    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights, None)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates, mapping, df_margins = resolve_links_with_margin(
        flight_index,
        candidates,
    )
    print(f"Number of selected candidates: {len(selected_candidates)}")
    print(f"Number of mappings: {len(mapping.keys())}")
    return mapping, df_margins


if __name__ == "__main__":
    start_date = date(2026, 1, 1)
    end_date = date(2026, 7, 1)
    df_flights = get_flights(start_date, end_date)
    df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    flights = flights_df_to_flights(df_flights)
    resolve_pia_mappings(flights)
