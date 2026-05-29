
from itertools import combinations
import networkx as nx

from data_engineering.flights.flight_type import Flight, TestFlight

def _get_pia_nodes(g: nx.DiGraph):
    pia_nodes = [
        node
        for node, attrs in g.nodes(data=True)
        if attrs["flight"].pia
    ]
    return pia_nodes

def build_graph(flights: list[TestFlight]) -> nx.DiGraph:
    G = nx.DiGraph()
    for flight in flights:
        G.add_node(flight.flight_id, flight=flight)

    for flight_i, flight_j in combinations(flights, 2):
        is_connected = (
            flight_i.landing_airport_ident == flight_j.takeoff_airport_ident
            and flight_i.landing_time < flight_j.takeoff_time
            and ((flight_i.icao ==flight_j.icao) or flight_j.pia == True or flight_i.pia == True)
        )
        if is_connected:
            G.add_edge(flight_i.flight_id, flight_j.flight_id)

    return G

def nodes_without_pia_path(g: nx.DiGraph) -> set[str]:
    pia_nodes = [
        node
        for node, attrs in g.nodes(data=True)
        if attrs["flight"].pia
    ]
    nodes_with_pia_path = set(pia_nodes)
    for pia_node in pia_nodes:
        nodes_with_pia_path.update(nx.ancestors(g, pia_node))
        nodes_with_pia_path.update(nx.descendants(g, pia_node))
    return set(g.nodes) - nodes_with_pia_path

def remove_nodes_without_pia_path(g: nx.DiGraph) -> set[str]:
    nodes_to_remove = nodes_without_pia_path(g)
    g.remove_nodes_from(nodes_to_remove)
    return nodes_to_remove

def remove_pia_unconnected_nodes(g: nx.DiGraph):
    pia_nodes = _get_pia_nodes(g)
    for node in pia_nodes:
        if g.in_degree[node] ==0 and g.out_degree == 0:
            raise Exception(f"{node} has 0 degrees")

def create_icao_pia_occurance_map(g: nx.DiGraph):
    icao_occurance_map = dict()
    for node in list(g.nodes):
        if g.nodes[node]["flight"].pia:
            predecessors = list(g.predecessors(node))
            successors = list(g.successors(node)) 
            for node in predecessors + successors:
                flight = g.nodes[node]["flight"]
                if flight.pia == False:
                    icao = flight.icao
                    icao_occurance_map[icao] = icao_occurance_map.get(icao, 0) + 1
    return icao_occurance_map
def match_nodes_with_category(g: nx.DiGraph):
    for node in list(g.nodes):
        flight: TestFlight = g.nodes[node]["flight"]
        pia = flight.pia
        category = flight.category
        if pia and category != "":
            predecessors = list(g.predecessors(node))
            successors = list(g.successors(node)) 
            for node_p in predecessors:
                node_category = g.nodes[node_p]["flight"].category
                if node_category != category:
                    g.remove_edge(node_p, node)
            for node_s in successors:
                node_category = g.nodes[node_s]["flight"].category
                if node_category != category:
                    g.remove_edge(node, node_s)
def process_graph(g: nx.DiGraph):
    nodes_to_remove = remove_nodes_without_pia_path(g)
    remove_pia_unconnected_nodes(g)
    print(f"Removed {len(nodes_to_remove)} nodes")
    mapping_dict = identify_pia_nodes(g)
    print(f"Matched {len(mapping_dict.keys())}")
    match_nodes_with_category(g)
    icao_occurance_map = create_icao_pia_occurance_map(g)
    for icao in mapping_dict.values():
        icao_occurance_map[icao] = icao_occurance_map.get(icao, 0) + 10
    new_mapping_dict = identify_pia_nodes(g)
    print(f"Matched {len(new_mapping_dict.keys())}")
    mapping_dict = mapping_dict | new_mapping_dict
    return g, mapping_dict

# I can save the mapping till the end
#maybe dynamic programming
def compute_edge_scores_for_node(g: nx.DiGraph): # probably biggest thing is if the real icao later took off from the airport. Then it's like 90% + it was this. This combines with teh time score so real icao + 30 days unlikely. real icao + 6 hours super likely it took this edge.
    # for now treat score like probablitiy adds up to 1 all edges. 
    # score 0.0 (impossible) -> #1.0
    # things to factor in 
    # if there's more possible edges than it's more ambigious which one it used so therefore that edge score should matter less
    for u in g.nodes:
        flight = g.nodes[u]["flight"] # maybe for now just do score of the divided by number. I cany simplify by when I pick best path I only need to recalculate for the nodes that were involved
        successors = list(g.successors(u))
        for v in successors:
            flight_s: Flight = g.nodes[v]["flight"]
            time_gap = flight_s.takeoff_time - flight.landing_time
            # time_gap_score(time_gap)
    pass

def identify_pia_nodes(g):
    mapping_dict: dict[str, str] = dict()
    for node in list(g.nodes):
        flight: TestFlight = g.nodes[node]["flight"]
        pia = flight.pia
        if pia:
            predecessors = list(g.predecessors(node))
            successors = list(g.successors(node))
            incoming = len(predecessors)
            outgoing = len(successors)
            if outgoing == 1:
                successor = successors[0]
                sucessor_flight: TestFlight = g.nodes[successor]["flight"]
                sucessor_icao = sucessor_flight.icao
                mapping_dict[node] = sucessor_icao
                for predecessor in predecessors:
                    predecssor_flight = g.nodes[predecessor]["flight"]
                    predecessor_icao = predecssor_flight.icao
                    if predecessor_icao == sucessor_icao:
                        g.remove_node(predecessor)
                g.remove_node(node)
    return mapping_dict

import math

def log_normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    max_count = max(counts.values(), default=0)

    if max_count == 0:
        return {icao: 0.0 for icao in counts}

    return {
        icao: math.log1p(count) / math.log1p(max_count)
        for icao, count in counts.items()
    }

def time_gap_score(landing_time, next_takeoff_time) -> float: # This should be based on average time gap between ladd planes.
    gap_hours = (next_takeoff_time - landing_time).total_seconds() / 3600

    if gap_hours < 0:
        return 0.0          # impossible: takeoff before landing

    if gap_hours < 0.5:
        return 0.05         # under 30 min: very unlikely

    if gap_hours < 1:
        return 0.25         # 30-60 min: possible but weak

    if gap_hours < 6:
        return 1.0          # 1-6 hours: strong

    if gap_hours < 24:
        return 0.7          # same day: reasonable

    if gap_hours < 72:
        return 0.35         # 1-3 days: weaker

    if gap_hours < 168:
        return 0.1          # 3-7 days: very weak

    return 0.02             # over 7 days: extremely weak