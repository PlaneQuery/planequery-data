
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
                if flight.landing_airport_ident == next_flight.takeoff_airport_ident:
                    # add edge from flight to next_flight
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
    gap = b.takeoff_time - a.landing_time
    gap_minutes = gap.total_seconds() / 60

    if gap_minutes < 15:
        score -= 300
    elif gap_minutes < 6 * 60:
        score += 1000
    elif gap_minutes < 24 * 60:
        score += 700
    elif gap_minutes < 3 * 24 * 60:
        score += 250
    elif gap_minutes < 7 * 24 * 60:
        score -= 100
    else:
        score -= 2000

    if a.icao == b.icao:
        score += 2000

    if a.landing_airport_ident == b.takeoff_airport_ident:
        score += 1500
    if airport_lookup.airport_ident_is_in_us_faa_airspace(a.takeoff_airport_ident) and not airport_lookup.airport_ident_is_in_us_faa_airspace(a.landing_airport_ident):
        if a.icao == b.icao:
            score += 2000
    if a.category and b.category:
        if a.category == b.category:
            score += 300
        else:
            score -= 300

    if a.callsign and b.callsign and a.callsign == b.callsign:
        score += 250

    score = max(-2000, min(score, 6000))
    return score

def solve_links(
    flight_index: FlightIndex,
    edges: list[FlightLinkCandidate],
    max_time_seconds: float = 60.0,
):
    model = cp_model.CpModel()
    edge_vars: list[cp_model.IntVar] = []

    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    nodes = set()
    # we could have a required node that all the real icao flights have to be used. 
    for i, edge in enumerate(edges):
        u = edge.previous_flight_id
        v = edge.next_flight_id
        nodes.add(u)
        nodes.add(v)
        x = model.new_bool_var(f'edge_{i}')
        incoming[v].append(x)
        outgoing[u].append(x)
        edge_vars.append(x)

    real_icaos = sorted({
        flight.icao
        for flight in flight_index.get_ladd_flights()
    })
    real_icao_labels = {icao: i for i, icao in enumerate(real_icaos)}
    pia_icao_labels: dict[str, cp_model.IntVar] = {}
    flight_labels: dict[str, cp_model.IntVar | int] = {}
    UNKNOWN_LABEL = len(real_icaos)
    for node in nodes:
        model.add(sum(incoming[node]) <= 1)
        model.add(sum(outgoing[node]) <= 1)

        f = flight_index.get_flight_by_id(node)
        if f.pia:
            icao = f.icao
            if icao not in pia_icao_labels:
                pia_icao_labels[icao] = model.new_int_var(0, UNKNOWN_LABEL, f'pia_to_real_{icao}')
            flight_labels[node] = pia_icao_labels[icao]
        else:
            icao = f.icao
            flight_labels[node] = real_icao_labels[icao]

    for edge, edge_var in zip(edges, edge_vars):
        u = edge.previous_flight_id
        v = edge.next_flight_id
        model.add(flight_labels[u] == flight_labels[v]).OnlyEnforceIf(edge_var)
 
    # objective: maximize the total score of the selected candidates
    model.maximize(
        sum(edge.score * edge_var for edge, edge_var in zip(edges, edge_vars))
    )
    
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = max_time_seconds
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_candidates = [
            candidate
            for candidate, edge_var in zip(edges, edge_vars)
            if solver.Value(edge_var) == 1
        ]
        return selected_candidates
    else:
        print('No solution found.')
        return []

def create_graph_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> nx.Graph:
    graph = nx.Graph()
    for candidate in selected_candidates:
        graph.add_edge(candidate.previous_flight_id, candidate.next_flight_id)
    print (f"Number of connected components: {nx.number_connected_components(graph)}")
    return graph

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

def mapping_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> dict[Flight, str]:
    graph = create_graph_from_selected_candidates(selected_candidates, flight_index)
    pia_to_real_icaos: dict[str, str] = dict()
    for component in nx.connected_components(graph):
        component_flights = []
        for flight_id in component:
            flight = flight_index.get_flight_by_id(flight_id)
            component_flights.append(flight)

        real_icao = None
        for f in component_flights:
            if f.ladd:
                real_icao = f.icao
                break
        if real_icao is None:
            continue

        for flight in component_flights:
            if flight.pia:
                if flight.pia in pia_to_real_icaos:
                    assert pia_to_real_icaos[flight.icao] == real_icao, (
                        f"Conflicting mappings for PIA ICAO {flight.icao}: "
                        f"{pia_to_real_icaos[flight.icao]} vs {real_icao}"
                    )
                pia_to_real_icaos[flight.icao] = real_icao

    resolved_pia_icaos = {}
    for pia_icao, real_icao in pia_to_real_icaos.items():
        resolved_pia_icaos[pia_icao] = real_icao

    return {
        flight: resolved_pia_icaos[flight.icao]
        for flight in flight_index.get_pia_flights()
        if flight.icao in resolved_pia_icaos
    }
def resolve_pia_mappings(
    flights: list[Flight],
    # TODO: known_pia_users: list[str],
    airport_continuity_tolerance: float = 10,
) -> dict[Flight, str]:
    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights, None)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    candidates = filter_candidates(candidates, flight_index)
    print(f"Number of candidates: {len(candidates)}")
    candidate_subgroups = split_candidates_by_component(candidates, flight_index)
    selected_candidates = []
    for candidates in candidate_subgroups:
        sel_candidates = solve_links(flight_index, candidates)
        selected_candidates.extend(sel_candidates)
    mapping = mapping_from_selected_candidates(selected_candidates, flight_index)
    print(f"Number of selected candidates: {len(selected_candidates)}")
    print(f"Number of mappings: {len(mapping.keys())}")
    return mapping

def split_candidates_by_component(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
) -> list[list[FlightLinkCandidate]]:
    """Group candidates by connected component of the candidate graph."""
    len_candidates = len(candidates)
    graph = create_graph_from_candidates(candidates, flight_index)
    components = list(nx.connected_components(graph))
    component_by_flight_id = {
        flight_id: component_index
        for component_index, component in enumerate(components)
        for flight_id in component
    }
    component_candidates: list[list[FlightLinkCandidate]] = [
        [] for _ in components
    ]

    for candidate in candidates:
        component_index = component_by_flight_id[candidate.previous_flight_id]
        component_candidates[component_index].append(candidate)

    total_num_candidates = sum(len(component) for component in component_candidates)
    assert total_num_candidates == len_candidates, (f"Total number of candidates in all components ({total_num_candidates}) ", f"does not match original number of candidates ({len_candidates})")
    print(f"Total number of candidates in all components: {total_num_candidates}")
    return component_candidates

if __name__ == "__main__":
    start_date = date(2026, 1, 1)
    end_date = date(2026, 7, 1)
    df_flights = get_flights(start_date, end_date)
    df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    flights = flights_df_to_flights(df_flights)
    resolve_pia_mappings(flights)
