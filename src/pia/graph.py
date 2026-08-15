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

def create_graph_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> nx.Graph:
    graph = nx.Graph()
    for candidate in selected_candidates:
        graph.add_edge(candidate.previous_flight_id, candidate.next_flight_id)
    print (f"Number of connected components: {nx.number_connected_components(graph)}")
    return graph