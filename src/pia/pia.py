
from itertools import combinations
import networkx as nx

from data_engineering.flights.flight_type import Flight, TestFlight

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

def process_graph(g: nx.DiGraph):
    mapping_dict: dict[str, str] = dict() # map a node to an icao
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
                print(sucessor_icao)
                mapping_dict[node] = sucessor_icao
                for predecessor in predecessors:
                    predecssor_flight = g.nodes[predecessor]["flight"]
                    predecessor_icao = predecssor_flight.icao
                    if predecessor_icao == sucessor_icao:
                        print(predecessor_icao)
                        print(sucessor_icao)
                        g.remove_node(predecessor)
                g.remove_node(node)
    return g, mapping_dict
