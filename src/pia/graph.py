from dataclasses import dataclass
from data_engineering.flights.flight_type import FlightIndex
import networkx as nx

@dataclass(frozen=True, slots=True)
class FlightLinkCandidate:
    previous_flight_id: str
    next_flight_id: str
    score: int

    @property
    def id(self) -> str:
        return f"{self.previous_flight_id}_{self.next_flight_id}"


def create_graph_from_candidates(
    candidates: list[FlightLinkCandidate],
    flight_index: FlightIndex,
) -> nx.DiGraph:
    """Build the complete, directed flight-continuity candidate graph.

    All indexed flights are included, even when they have no candidate edges.
    That makes an isolated flight distinguishable from a flight missing from the
    input when the graph is used to evaluate known PIA mappings.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(flight_index.get_flights_ids())
    graph.add_edges_from(
        (candidate.previous_flight_id, candidate.next_flight_id)
        for candidate in candidates
    )
    return graph

def create_graph_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> nx.DiGraph:
    graph = nx.DiGraph()
    for candidate in selected_candidates:
        graph.add_edge(candidate.previous_flight_id, candidate.next_flight_id, score=candidate.score)
    print (f"Number of connected components: {nx.number_weakly_connected_components(graph)}")
    return graph

def flight_overlaps(
    flight_a,
    flight_b,
) -> bool:
    """Return True if the two flights overlap in time."""
    return not (
        flight_a.landing_time <= flight_b.takeoff_time
        or flight_b.landing_time <= flight_a.takeoff_time
    )
def check_graph_for_overlap(graph: nx.DiGraph, flight_index: FlightIndex):
    for icao, flights in flight_index.flights_by_icao.items():
        if len(flights) < 2:
            continue
        first_flight = flights[0]
        last_flight = flights[-1]
        if first_flight.pia:
            continue
        paths = list(nx.all_simple_paths(graph, source=first_flight.flight_id, target=last_flight.flight_id))
        for path in paths:
            icaos = set()
            check_path = False
            for node in path:
                flight = flight_index.get_flight_by_id(node)
                if flight.pia:
                    check_path = True
                icaos.add(flight.icao)
            if not check_path:
                continue
            flights = []
            for icao in icaos:
                flights.extend(flight_index.flights_by_icao[icao])
            for i in range(len(path) - 1):
                flight_a = flight_index.get_flight_by_id(path[i])
                other_flights = flights.copy()
                other_flights.pop(i)
                for flight_b in other_flights:
                    overlaps = flight_overlaps(flight_a, flight_b)
                    assert overlaps == False, f"Flights {flight_a.flight_id} and {flight_b.flight_id} overlap in time but are connected in the graph."


