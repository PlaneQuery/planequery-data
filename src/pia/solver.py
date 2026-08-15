import networkx as nx
import polars as pl
from collections import defaultdict
from ortools.sat.python import cp_model
from data_engineering.flights.flight_type import Flight, FlightIndex
from pia.graph import FlightLinkCandidate, create_graph_from_candidates, create_graph_from_selected_candidates
DEBUG = True
MINIMUM_SCORE_MARGIN = 1000
def resolve_links_with_margin(flight_index, candidates, use_margin_solve = False):
    candidate_components = [
        component
        for component in split_candidates_by_component(candidates, flight_index)
        if component
    ]
    selected_by_component: list[list[FlightLinkCandidate]] = []
    score_by_component: list[int] = []
    components_by_pia: dict[str, set[int]] = defaultdict(set)

    for component_index, component in enumerate(candidate_components):
        selected = solve_links(flight_index, component)
        selected_by_component.append(selected)
        score_by_component.append(sum(candidate.score for candidate in selected))

        for candidate in component:
            for flight_id in (
                candidate.previous_flight_id,
                candidate.next_flight_id,
            ):
                flight = flight_index.get_flight_by_id(flight_id)
                if flight.pia:
                    components_by_pia[flight.icao].add(component_index)

    selected_candidates = [
        candidate
        for selected in selected_by_component
        for candidate in selected
    ]
    print(f"Number of selected candidates: {len(selected_candidates)}")
    mapping = mapping_from_selected_candidates(selected_candidates, flight_index)
    if not use_margin_solve:
        return selected_candidates, mapping, pl.DataFrame()
    icao_mapping: dict[str, str] = {}
    flights_by_icao: dict[str, list[Flight]] = defaultdict(list)
    for flight, real_icao in mapping.items():
        if flight.icao in icao_mapping:
            assert icao_mapping[flight.icao] == real_icao, (
                f"Conflicting mappings for PIA ICAO {flight.icao}: "
                f"{icao_mapping[flight.icao]} vs {real_icao}"
            )
        else:
            icao_mapping[flight.icao] = real_icao
        flights_by_icao[flight.icao].append(flight)

    margin_by_pia_icao: dict[str, int | None] = {
        flight.icao: None
        for flight in flight_index.get_pia_flights()
    }
    margins = []
    for pia_icao, real_icao in list(icao_mapping.items()):
        assert pia_icao != real_icao, f"PIA ICAO {pia_icao} should not be the same as real ICAO {real_icao}"
        forbidden_mapping = {pia_icao: real_icao}
        affected_components = components_by_pia[pia_icao]
        original_affected_score = sum(
            score_by_component[index]
            for index in affected_components
        )
        alternative_affected_score = 0

        for index in affected_components:
            alternative = solve_links(
                flight_index,
                candidate_components[index],
                forbidden_mapping=forbidden_mapping,
            )
            alternative_affected_score += sum(
                candidate.score for candidate in alternative
            )

        margin = original_affected_score - alternative_affected_score
        margin_by_pia_icao[pia_icao] = margin
        margins.append(margin)
        if margin < MINIMUM_SCORE_MARGIN:
            flights = flights_by_icao[pia_icao]
            for flight in flights:
                mapping.pop(flight)
        print(
            f"PIA ICAO {pia_icao} -> Real ICAO {real_icao}: "
            f"Original affected score {original_affected_score}, "
            f"alternative affected score {alternative_affected_score}, "
            f"margin {margin}"
        )

    if margins:
        print(f"Margins: count={len(margins)}, max={max(margins)}, min={min(margins)}, average={sum(margins) / len(margins)}")

    if DEBUG:
        pia_icaos = sorted(margin_by_pia_icao)
        df_margins = pl.DataFrame(
            {
                "pia_icao": pl.Series(pia_icaos, dtype=pl.String),
                "predicted_icao": pl.Series(
                    [icao_mapping.get(pia_icao) for pia_icao in pia_icaos],
                    dtype=pl.String,
                ),
                "margin": pl.Series(
                    [margin_by_pia_icao[pia_icao] for pia_icao in pia_icaos],
                    dtype=pl.Int64,
                ),
            }
        )
    else:
        df_margins = pl.DataFrame()
    return selected_candidates, mapping, df_margins


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

        if len(flight_index.flights_by_icao[f.icao]) > 1:
            model.add(sum(incoming[node]) + sum(outgoing[node]) >= 1)
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
    print(f"Total number of componenets: {len(components)}")
    return component_candidates


def mapping_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> dict[Flight, str]:
    graph = create_graph_from_selected_candidates(selected_candidates, flight_index)
    pia_to_real_icaos: dict[str, str] = dict()
    components = nx.connected_components(graph)
    print(f"Number of connected components: {nx.number_connected_components(graph)}")
    for component in components:
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
                if flight.icao in pia_to_real_icaos:
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
