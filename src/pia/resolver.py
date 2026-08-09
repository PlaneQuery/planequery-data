
from collections import defaultdict, deque
from datetime import timedelta
from itertools import combinations
from dataclasses import dataclass
import networkx as nx
from ortools.sat.python import cp_model
from itertools import groupby
from datetime import date
from airports.airport_lookup import AirportLookup
from data_engineering.flights.flight_type import Flight, FlightIndex, flights_df_to_flights, pia_or_american_ladd_icao_filter, get_flights

AIRPORT_CONTINUITY_TOLERANCE = 10
airport_lookup = AirportLookup()
@dataclass(frozen=True, slots=True)
class FlightLinkCandidate:
    id: int
    previous_flight_id: str
    next_flight_id: str
    score: int

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
                    len(candidates),
                    flight.flight_id,
                    next_flight.flight_id,
                    score = score
                )
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
                    len(candidates),
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
                        len(candidates),
                        flight.flight_id,
                        ladd_flight.flight_id,
                        score = score
                    )
                    candidates.append(candidate)

    return candidates

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
    candidates: list[FlightLinkCandidate],
    max_time_seconds: float = 60.0,
):
    model = cp_model.CpModel()
    candidate_vars: dict[int, cp_model.IntVar] = {}
    outgoing_vars: defaultdict[str, list[cp_model.IntVar]] = defaultdict(list)
    incoming_vars: defaultdict[str, list[cp_model.IntVar]] = defaultdict(list)

    for candidate in candidates:
        candidate_var = model.new_bool_var(f'candidate_{candidate.id}')
        candidate_vars[candidate.id] = candidate_var
        outgoing_vars[candidate.previous_flight_id].append(candidate_var)
        incoming_vars[candidate.next_flight_id].append(candidate_var)

    # A selected path may include a LADD ICAO more than once, but it may not
    # connect flights from two distinct LADD ICAOs. A PIA ICAO also has one
    # shared label across all of its flights, so it cannot join multiple LADD
    # paths. Selecting an edge requires equal labels at both endpoints.
    candidate_flight_ids = set(outgoing_vars) | set(incoming_vars)
    ladd_icaos = set()
    for flight_id in flight_index.get_flights_ids():
        flight = flight_index.get_flight_by_id(flight_id)
        if flight is not None and flight.ladd:
            ladd_icaos.add(flight.icao)
    ladd_icao_labels = {
        icao: index
        for index, icao in enumerate(sorted(ladd_icaos), start=1)
    }
    pia_icao_labels: dict[str, cp_model.IntVar] = {}
    flight_labels: dict[str, int | cp_model.IntVar] = {}
    for flight_id in candidate_flight_ids:
        flight = flight_index.get_flight_by_id(flight_id)

        # An ICAO is known to be LADD when at least one of its flights is
        # marked LADD.  Keep all non-PIA flights for that ICAO fixed to the
        # same label, even if an individual flight's LADD flag is missing.
        if flight.pia:
            pia_icao_label = pia_icao_labels.get(flight.icao)
            if pia_icao_label is None:
                pia_icao_label = model.new_int_var(
                    0,
                    len(ladd_icao_labels),
                    f"ladd_icao_label_{flight.icao}",
                )
                pia_icao_labels[flight.icao] = pia_icao_label
            flight_labels[flight_id] = pia_icao_label
        elif flight.icao in ladd_icao_labels:
            flight_labels[flight_id] = ladd_icao_labels[flight.icao]
        else:
            flight_labels[flight_id] = model.new_int_var(
                0,
                len(ladd_icao_labels),
                f"ladd_icao_label_{flight_id}",
            )

    for candidate in candidates:
        previous_label = flight_labels[candidate.previous_flight_id]
        next_label = flight_labels[candidate.next_flight_id]
        if isinstance(previous_label, int) and isinstance(next_label, int):
            if previous_label != next_label:
                model.add(candidate_vars[candidate.id] == 0)
        else:
            model.add(previous_label == next_label).only_enforce_if(
                candidate_vars[candidate.id]
            )
    
    # add constraints to ensure that each flight is linked to at most one next flight
    for next_candidates in outgoing_vars.values():
        model.add_at_most_one(next_candidates)
    
    # add constraints to ensure that each flight is linked to at most one previous flight
    for prev_candidates in incoming_vars.values():
        model.add_at_most_one(prev_candidates)
    
    # objective: maximize the total score of the selected candidates
    model.maximize(sum(candidate.score * candidate_vars[candidate.id] for candidate in candidates))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    print("Start Solver")
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_candidates = [candidate for candidate in candidates if solver.Value(candidate_vars[candidate.id]) == 1]
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

def create_graph_from_candidates(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
) -> nx.Graph:
    """Build the complete, undirected flight-continuity candidate graph.

    All indexed flights are included, even when they have no candidate edges.
    That makes an isolated flight distinguishable from a flight missing from the
    input when the graph is used to evaluate known PIA mappings.
    """
    graph = nx.Graph()
    graph.add_nodes_from(flight_index.get_flights_ids())
    graph.add_edges_from(
        (candidate.previous_flight_id, candidate.next_flight_id)
        for candidate in candidates
    )
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
    airport_continuity_tolerance: float = 10,
) -> dict[Flight, str]:
    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights, None)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates = solve_links(flight_index, candidates)
    create_graph_from_selected_candidates(selected_candidates, flight_index)
    print(f"Number of selected candidates: {len(selected_candidates)}")
    mapping = mapping_from_selected_candidates(selected_candidates, flight_index)
    print(f"Number of mappings: {len(mapping.keys())}")
    return mapping


if __name__ == "__main__":
    start_date = date(2026,5,20)
    end_date = date(2026,5,22)
    df_flights = get_flights(start_date, end_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    flights = flights_df_to_flights(df_flights)
    flight_index = FlightIndex(flights)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates = solve_links(flight_index, candidates)
    print(f"Number of selected candidates: {len(selected_candidates)}")
    mapping = mapping_from_selected_candidates(selected_candidates, flight_index)
    print(f"Number of mappings: {len(mapping.keys())}")
