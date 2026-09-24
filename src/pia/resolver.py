from collections import defaultdict, deque
from datetime import timedelta
import time
from itertools import combinations
import networkx as nx
import polars as pl
import numpy as np
from ortools.sat.python import cp_model
from itertools import groupby
from datetime import date
from airports.runtime import airport_catalog_for_date
from data_engineering.filters import pia_or_american_ladd_icao
from flights.algorithm import get_flights_range
from flights.airport_reference import airport_from_reference
from flights.flight_type import Flight, FlightIndex, flights_df_to_flights
from pia.check_mappings import check_mappings
from pia.graph import FlightLinkCandidate, create_graph_from_candidates, create_graph_from_selected_candidates
from datetime import datetime
import heapq
from pia.solver import pia_icao_mapping_from_selected_candidates, resolve_links_with_margin

def build_candidates(flight_index: FlightIndex) -> list[FlightLinkCandidate]:
    # sorted by icao, time
    # add scores to each edge. 
    overlapping_pairs = find_overlapping_icao_pairs(flight_index)
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
                candidates.append(candidate)
            else:
                next_flight = None
            assert flight.last_lon is not None # TODO: Fix this in type
            assert flight.last_lat is not None

            if i != len(flights) - 1 and next_flight is not None:
                before_time = next_flight.takeoff_time
            else:
                before_time = None
            pia_flights = flight_index.pia_flights_after(
                flight.landing_time,
                flight.landing_airport_id,
                before_time=before_time,
            )
            for pia_flight in pia_flights:
                icao_pair = tuple(sorted((icao, pia_flight.icao)))
                if pia_flight.category != flight.category:
                    continue
                if icao_pair in overlapping_pairs:
                    continue
                score = edge_score(flight, pia_flight)
                candidate = FlightLinkCandidate(
                    flight.flight_id,
                    pia_flight.flight_id,
                    score = score
                )
                candidates.append(candidate)
            if flight.pia:
                ladd_flights = flight_index.ladd_flights_after(
                    flight.landing_time,
                    flight.landing_airport_id,
                    before_time=before_time,
                )
                for ladd_flight in ladd_flights:
                    icao_pair = tuple(sorted((icao, ladd_flight.icao)))
                    if icao_pair in overlapping_pairs:
                        continue
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

def find_overlapping_icao_pairs(flight_index: FlightIndex) -> set[tuple[str, str]]:
    """Return ICAO pairs whose flight histories overlap in time.

    Flights are processed with a sweep line, so the work is proportional to
    the number of flights that are simultaneously active rather than to every
    possible pair of flights. Real-to-real pairs do not need constraints
    because distinct real ICAOs already have distinct fixed solver labels.
    """
    pia_icaos = {flight.icao for flight in flight_index.get_pia_flights()}
    flights = sorted(
        (
            flight
            for icao_flights in flight_index.flights_by_icao.values()
            for flight in icao_flights
        ),
        key=lambda flight: flight.takeoff_time,
    )

    active_heap: list[tuple[datetime, str]] = []
    active_flights: dict[str, Flight] = {}
    overlapping_pairs: set[tuple[str,str]] = set()

    for flight in flights:
        while active_heap and active_heap[0][0] <= flight.takeoff_time:
            _, finished_flight_id = heapq.heappop(active_heap)
            active_flights.pop(finished_flight_id, None)

        for active_flight in active_flights.values():
            if active_flight.icao == flight.icao:
                continue
            if (
                active_flight.icao not in pia_icaos
                and flight.icao not in pia_icaos
            ):
                continue
            overlapping_pairs.add(
                tuple(sorted((active_flight.icao, flight.icao)))
            )

        active_flights[flight.flight_id] = flight
        heapq.heappush(
            active_heap,
            (flight.landing_time, flight.flight_id),
        )
    print(f"Found {len(overlapping_pairs)} overlapping ICAO pairs.")
    return overlapping_pairs

def edge_score(a: Flight, b: Flight) -> int:
    score = 0

    a_catalog = airport_catalog_for_date(a.takeoff_time.date())
    a_takeoff_airport = airport_from_reference(
        a_catalog,
        a.takeoff_airport_id,
    )
    a_landing_airport = airport_from_reference(
        a_catalog,
        a.landing_airport_id,
    )
    if a.icao == b.icao:
        score += 2000

    if a.landing_airport_id == b.takeoff_airport_id:
        score += 1500
    if (
        a_takeoff_airport is not None
        and a_landing_airport is not None
        and a_takeoff_airport.is_in_us_faa_airspace
        and not a_landing_airport.is_in_us_faa_airspace
    ):
        if a.icao == b.icao:
            score += 2000
    if a.callsign and b.callsign and a.callsign == b.callsign:
        score += 250

    score = max(-2000, min(score, 6000))
    return score

def candidate_mapping_path_exists(
    graph: nx.DiGraph,
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

    undirected_graph = graph.to_undirected(as_view=True)
    for pia_flight_id in pia_flight_ids:
        if pia_flight_id not in graph:
            continue
        component = nx.node_connected_component(
            undirected_graph,
            pia_flight_id,
        )
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
    undirected_graph = graph.to_undirected(as_view=True)
    connected_flight_ids: set[str] = set()
    for flight_id in target_flight_ids:
        connected_flight_ids.update(
            nx.node_connected_component(undirected_graph, flight_id)
        )

    return sorted(
        (
            flight_index.get_flight_by_id(flight_id)
            for flight_id in connected_flight_ids
        ),
        key=lambda flight: flight.takeoff_time,
    )

def resolve_pia_mappings(
    flights: list[Flight],
    known_pia_mappings: dict[str, str] = {},
    # TODO: known_pia_users: list[str],
) -> tuple[dict[Flight, str], pl.DataFrame, pl.DataFrame]:
    if len(known_pia_mappings) > 0:
        icaos_to_remove = set(known_pia_mappings.keys()) | set(known_pia_mappings.values())
        flights[:] = [
            flight for flight in flights
            if flight.icao not in icaos_to_remove
        ]
    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates, mapping, df_margins = resolve_links_with_margin(
        flight_index,
        candidates,
    )
    check_mappings(flight_index, mapping)
    graph = create_graph_from_selected_candidates(selected_candidates, flight_index)
    # selected_graph_stats(graph, flight_index)
    pia_icao_mapping = pia_icao_mapping_from_selected_candidates(
    selected_candidates,
    flight_index,
    )
    pia_mapping_stats_dict = create_mapping_stats(graph, flight_index, pia_icao_mapping)
    df_mapping_stats = pl.DataFrame(
        [
            {"pia_icao": pia_icao, **stats}
            for pia_icao, stats in pia_mapping_stats_dict.items()
        ]
    )
    print(f"Number of selected candidates: {len(selected_candidates)}")
    print(f"Number of mappings: {len(mapping.keys())}")
    return mapping, df_margins, df_mapping_stats

def selected_graph_stats(graph: nx.DiGraph, flight_index: FlightIndex):
    # I should only get 1 path that contains real_flights[0] and real_flights[-1] maybe 0? 
    for icao, flights in flight_index.flights_by_icao.items():
        if len(flights) < 2:
            continue
        first_flight = flights[0]
        last_flight = flights[-1]
        if first_flight.pia:
            continue
        paths = list(nx.all_simple_paths(graph, source=first_flight.flight_id, target=last_flight.flight_id))
        assert len(paths) == 1, f"Expected 1 path for {icao}, but found {len(paths)}"

def create_mapping_stats(graph: nx.DiGraph, flight_index: FlightIndex, mapping: dict):
    pia_icao_mapping_stats = dict()
    component_count = 0
    for nodes in nx.weakly_connected_components(graph):
        component_count += 1
        component = graph.subgraph(nodes)
        longest_path = nx.dag_longest_path(component)
        path_edges = [
        (u, v, component.edges[u, v])
        for u, v in nx.utils.pairwise(longest_path)
        ]
        score = 0
        for u, v, edge_data in path_edges:
            score += edge_data["score"]
        number_of_flights = len(longest_path)
        # calculate score on this.
        pia_icaos = set()
        for node in longest_path:
            flight = flight_index.get_flight_by_id(node)
            if flight.icao in mapping:
                pia_icaos.add(flight.icao)
        for pia_icao in pia_icaos:
            pia_icao_mapping_stats[pia_icao] = {
                "score": score,
                "number_of_flights": number_of_flights,
            }
    print(f"Number of connected components in selected graph: {component_count}")
    return pia_icao_mapping_stats



if __name__ == "__main__":
    start_date = date(2026, 6, 1)
    end_date = date(2026, 7, 1)
    df_flights = get_flights_range(start_date, end_date)
    df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
    df_flights = df_flights.filter(pia_or_american_ladd_icao())
    flights = flights_df_to_flights(df_flights)
    resolve_pia_mappings(flights)
