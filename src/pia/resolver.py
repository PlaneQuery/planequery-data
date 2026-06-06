
from datetime import timedelta
from itertools import combinations
import networkx as nx
from ortools.sat.python import cp_model

from data_engineering.flights.flight_type import Flight, TestFlight

def _get_pia_nodes(g: nx.DiGraph):
    pia_nodes = [
        node
        for node, attrs in g.nodes(data=True)
        if attrs["flight"].pia
    ]
    return pia_nodes

def _get_real_nodes(g:nx.DiGraph):
    real_nodes = [
        node for node,attrs in g.nodes(data=True)
        if attrs["flight"].pia == False
    ]
    return real_nodes

def remove_pia_source_nodes(g: nx.DiGraph) -> nx.DiGraph:
    g = g.copy()
    while True:
        pia_source_nodes = [
            node
            for node, attrs in g.nodes(data=True)
            if attrs["flight"].pia and g.in_degree(node) == 0
        ]
        if not pia_source_nodes:
            return g
        print(f"{len(pia_source_nodes)} pia source nodes remove")
        # for node in pia_source_nodes:
        #     flight = g.nodes[node]["flight"]
        #     print(flight.icao, flight.takeoff_time, flight.takeoff_airport_ident)
        # raise Exception("pia source nodes present")
        g.remove_nodes_from(pia_source_nodes)


MAX_CONNECTION_GAP = timedelta(days=30)
def build_graph(flights: list[Flight]) -> nx.DiGraph:
    flights = sorted(flights, key=lambda f: (f.takeoff_time, f.icao))
    G = nx.DiGraph()
    for flight in flights:
        G.add_node(flight.flight_id, flight=flight)

    for flight_i, flight_j in combinations(flights, 2):
        is_connected = (
            flight_i.landing_airport_ident == flight_j.takeoff_airport_ident
            and flight_i.landing_time < flight_j.takeoff_time
            and flight_j.takeoff_time - flight_i.landing_time <= MAX_CONNECTION_GAP
            and ((flight_i.icao ==flight_j.icao) or flight_j.pia == True or flight_i.pia == True)
        )
        if is_connected:
            G.add_edge(flight_i.flight_id, flight_j.flight_id)

    return G

def print_graph(g: nx.DiGraph) -> None:
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("Nodes:")
    for node in sorted(g.nodes, key=lambda node: g.nodes[node]["flight"].takeoff_time):
        flight = g.nodes[node]["flight"]
        tags = []
        if flight.pia:
            tags.append("PIA")
        if flight.ladd:
            tags.append("LADD")
        tag_text = ",".join(tags) if tags else "REAL"
        print(
            f"  {node} [{tag_text}] "
            f"{flight.takeoff_time:%Y-%m-%d %H:%M} "
            f"{flight.takeoff_airport_ident}->{flight.landing_airport_ident} "
            f"icao={flight.icao} "
            f"category={flight.category or 'unknown'}"
        )

    print("Edges:")
    for source, target in sorted(
        g.edges,
        key=lambda edge: (
            g.nodes[edge[0]]["flight"].takeoff_time,
            g.nodes[edge[1]]["flight"].takeoff_time,
        ),
    ):
        source_flight = g.nodes[source]["flight"]
        target_flight = g.nodes[target]["flight"]
        gap = target_flight.takeoff_time - source_flight.landing_time
        print(
            f"  {source} -> {target} "
            f"via {source_flight.landing_airport_ident} "
            f"gap={gap}"
        )

def compute_edge_scores_for_node(g: nx.DiGraph): # probably biggest thing is if the real icao later took off from the airport. Then it's like 90% + it was this. This combines with teh time score so real icao + 30 days unlikely. real icao + 6 hours super likely it took this edge.
    # for now treat score like probablitiy adds up to 1 all edges. 
    # score 0.0 (impossible) -> #1.0
    # things to factor in 
    # if there's more possible edges than it's more ambigious which one it used so therefore that edge score should matter less
    # TODO: incorporate if callsign is the same.
    for u in g.nodes:
        flight = g.nodes[u]["flight"] # maybe for now just do score of the divided by number. I cany simplify by when I pick best path I only need to recalculate for the nodes that were involved
        successors = list(g.successors(u))
        for v in successors:
            flight_s: Flight = g.nodes[v]["flight"]
            time_gap = flight_s.takeoff_time - flight.landing_time
            # time_gap_score(time_gap)
            g.edges[u,v]["score"] = 1

def edge_score(a: Flight, b: Flight) -> int:
    score = 0
    gap = b.takeoff_time - a.landing_time
    gap_minutes = gap.total_seconds() / 60

    if gap_minutes < 30:
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
        score -= 500

    if a.icao == b.icao:
        score += 2000

    if a.category and b.category:
        if a.category == b.category:
            score += 300
        else:
            score -= 300

    if a.callsign and b.callsign and a.callsign == b.callsign:
        score += 250

    score = max(-2000, min(score, 1000))
    return score
def time_gap_score(time_delta: timedelta) -> float: # This should be based on average time gap between ladd planes.
    gap_hours = time_delta.total_seconds() / 3600

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

UNRESOLVED_PREFIX = "unresolved_pia:"
def get_canidate_icaos(g: nx.DiGraph):
    candidates = {}
    for node in list(g.nodes):
        flight: Flight = g.nodes[node]["flight"]
        if flight.pia:
            descendants = nx.descendants(g,node) # maybe I'll need to make this bounded.
            ancestors = nx.ancestors(g,node)
            nodes =  descendants | ancestors
            real_icaos = set()
            for node in nodes:
                f = g.nodes[node]["flight"]
                if f.pia == False:
                    real_icaos.add(f.icao)
            candidates[flight.flight_id] = sorted(list(real_icaos)) +  [f"{UNRESOLVED_PREFIX}{flight.icao}"]
        else:
            candidates[flight.flight_id] = [flight.icao]
    return candidates

def process_graph(g: nx.DiGraph, flights: list[Flight]):
    model = cp_model.CpModel()
    y: dict[tuple[str,str], cp_model.IntVar] = {}
    canidates_icaos_dict = get_canidate_icaos(g)
    print(f"Number of nodes: {len(g.nodes)}")
    for flight in flights:
        assignment_vars = []
        for real_icao in canidates_icaos_dict[flight.flight_id]:
            assignment_var = model.new_bool_var(f"y_{flight.flight_id}_{real_icao}")
            y[(flight.flight_id, real_icao)] = assignment_var
            assignment_vars.append(assignment_var)
        model.add(sum(assignment_vars) == 1)


    x: dict[tuple[str, str, str], cp_model.IntVar] = {}

    for i, j in g.edges: # basically a flight_id
        fi = g.nodes[i]["flight"]
        fj = g.nodes[j]["flight"]
        shared_candidate_icaos = set(canidates_icaos_dict[fi.flight_id]) & set(canidates_icaos_dict[fj.flight_id])
        for real_icao in sorted(shared_candidate_icaos):
            var = model.new_bool_var(f"x_{i}_{j}_{real_icao}")
            x[(i,j,real_icao)] = var

            model.add(var <= y[i,real_icao])
            model.add(var <= y[j,real_icao])


    for j in g.nodes:
        incoming_vars = []
        outgoing_vars = []
        for (i2,j2,real_icao), var in x.items():
            if j2 == j:
                incoming_vars.append(var)
            
            if i2 == j:
                outgoing_vars.append(var)
        if incoming_vars:
            model.add(sum(incoming_vars) <= 1)
        if outgoing_vars:
            model.add(sum(outgoing_vars) <= 1)
    
    # Optional
    start_vars: dict[str,cp_model.IntVar] = {}
    end_vars: dict[str,cp_model.IntVar] = {}

    for flight in flights:
        fid = flight.flight_id
        start_var = model.new_bool_var(f"start_{fid}")
        end_var = model.new_bool_var(f"end_{fid}")
        start_vars[fid] = start_var
        end_vars[fid] = end_var

        incoming_vars = [
            var
            for (i2, j2, real_icao), var in x.items()
            if j2 == fid
        ]

        outgoing_vars = [
            var
            for (i2, j2, real_icao), var in x.items()
            if i2 == fid
        ]

        model.add(start_var + sum(incoming_vars) == 1)
        model.add(end_var + sum(outgoing_vars) == 1)

    objective_terms = []

    for (i,j,real_icao), var in x.items():
        fi = g.nodes[i]["flight"]
        fj = g.nodes[j]["flight"]
        score = edge_score(fi, fj)
        objective_terms.append(score * var)
    START_PENALTY: int = 300
    END_PENALTY: int = 300
    for node in g.nodes:
        objective_terms.append(-START_PENALTY * start_vars[node])
        objective_terms.append(-END_PENALTY * end_vars[node])


    model.maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    print("start Solver")
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print_graph(g)
        raise RuntimeError("No feasible solution found")
    

    pia_mapping_dict = {}

    for flight in flights:
        if flight.pia:
            for mapped_icao in canidates_icaos_dict[flight.flight_id]:
                if solver.Value(y[(flight.flight_id, mapped_icao)]) == 1:
                    pia_mapping_dict[flight.flight_id] = mapped_icao
                    break
    
    return pia_mapping_dict, g
def resolve_pia_mappings(flights: list[Flight]):
    print(f"Number of flights: {len(flights)}")
    g = build_graph(flights)
    components = list(nx.weakly_connected_components(g))
    subgraphs = [
        g.subgraph(component_nodes)
        for component_nodes in components
    ]
    print(f"Number of subgraphs: {len(subgraphs)}")

    # 1. subdivide into  unconnected subgraphs to make computation easier/quicker. 
    # compute_edge_scores_for_node(g)
    mappings_dict = {}
    for g in subgraphs:
        if not _get_pia_nodes(g):
            continue
        subgraph_flights = [g.nodes[node]["flight"] for node in g.nodes]
        mappings, graph  = process_graph(g, subgraph_flights)
        mappings_dict.update(mappings)
    return mappings_dict, graph
