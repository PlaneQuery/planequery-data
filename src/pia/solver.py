import networkx as nx
from collections import defaultdict, deque
from datetime import timedelta
from itertools import combinations
import networkx as nx
import polars as pl
from ortools.sat.python import cp_model
from itertools import groupby
from datetime import date
from airports.airport_lookup import AirportLookup
from ortools.sat.python import cp_model
from data_engineering.flights.flight_type import FlightIndex
from pia.graph import FlightLinkCandidate, create_graph_from_candidates, create_graph_from_selected_candidates


def _resolve_links(flight_index, candidates):
    candidate_subgroups = split_candidates_by_component(candidates, flight_index)
    selected_candidates = []
    for candidates in candidate_subgroups:
        sel_candidates = solve_links(flight_index, candidates)
        selected_candidates.extend(sel_candidates)
    return selected_candidates
def resolve_links_with_margin(flight_index, candidates):
    selected_candidates = _resolve_links(flight_index, candidates)
    icao_mapping = pia_icao_mapping_from_selected_candidates(selected_candidates, flight_index)
    for pia_icao, real_icao in icao_mapping.items():
        if pia_icao == real_icao:
            continue
        
    return selected_candidates


def pia_icao_mapping_from_selected_candidates(
    selected_candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
) -> dict[str, str]:
    """Return a mapping from each resolved PIA ICAO to its real ICAO."""
    graph = create_graph_from_selected_candidates(
        selected_candidates,
        flight_index,
    )
    pia_to_real_icaos: dict[str, str] = {}

    for component in nx.connected_components(graph):
        component_flights = [
            flight_index.get_flight_by_id(flight_id)
            for flight_id in component
        ]

        real_icao = next(
            (
                flight.icao
                for flight in component_flights
                if flight.ladd
            ),
            None,
        )
        if real_icao is None:
            continue

        for flight in component_flights:
            if flight.pia:
                pia_to_real_icaos[flight.icao] = real_icao

    return pia_to_real_icaos


def solve_links(
    flight_index: FlightIndex,
    edges: list[FlightLinkCandidate],
    forbidden_mapping: dict[str,str] | None = None # pia_icao -> real_icao
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

    real_icao_labels: dict[str, int] = dict()
    for i, icao in enumerate(real_icaos):
        real_icao_labels[icao] = i
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
                label =  model.new_int_var(0, UNKNOWN_LABEL, f'pia_to_real_{icao}')
                if forbidden_mapping and icao in forbidden_mapping:
                    forbidden_real_icao = forbidden_mapping[icao]
                    forbidden_label = real_icao_labels[forbidden_real_icao]
                    model.add(label != forbidden_label)
                pia_icao_labels[icao] = label
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
