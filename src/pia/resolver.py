
from collections import defaultdict, deque
from datetime import timedelta
from itertools import combinations
from dataclasses import dataclass
import networkx as nx
from ortools.sat.python import cp_model
from itertools import groupby
from datetime import date
from data_engineering.flights.flight_type import Flight, FlightIndex, flights_df_to_flights, pia_or_american_ladd_icao_filter, get_flights

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
        next_flight = None
        for i in range(len(flights)):
            flight = flights[i]
            if i < len(flights) - 1:
                next_flight = flights[i + 1]
                candidate = FlightLinkCandidate(
                    len(candidates),
                    flight.flight_id,
                    next_flight.flight_id,
                    score = 1
                )
                # add edge from flight to next_flight
                candidates.append(candidate)
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
                candidate = FlightLinkCandidate(
                    len(candidates),
                    flight.flight_id,
                    pia_flight.flight_id,
                    score = 1
                )
                candidates.append(candidate)
            if flight.pia:
                ladd_flights = flight_index.ladd_flights_after(flight.landing_time, flight.last_lat, flight.last_lon, flight.landing_airport_ident, before_time = before_time)
                for ladd_flight in ladd_flights:
                    candidate = FlightLinkCandidate(
                        len(candidates),
                        flight.flight_id,
                        ladd_flight.flight_id,
                        score = 1
                    )
                    candidates.append(candidate)

    return candidates
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

def mapping_from_selected_candidates(selected_candidates: list[FlightLinkCandidate], flight_index: FlightIndex) -> dict[Flight, str]:
    mapping = {}
    for candidate in selected_candidates:
        previous_flight = flight_index.get_flight_by_id(candidate.previous_flight_id)
        next_flight = flight_index.get_flight_by_id(candidate.next_flight_id)
        assert previous_flight is not None
        assert next_flight is not None
        if previous_flight.pia:
            mapping[previous_flight] = next_flight.icao
        if next_flight.pia:
            mapping[next_flight] = previous_flight.icao
    return mapping
def resolve_pia_mappings(flights: list[Flight]) -> dict[Flight, str]:
    print(f"Number of flights: {len(flights)}")
    flight_index = FlightIndex(flights)
    pia_flights = flight_index.get_pia_flights()
    print(f"Number of PIA flights: {len(pia_flights)}")
    candidates = build_candidates(flight_index)
    print(f"Number of candidates: {len(candidates)}")
    selected_candidates = solve_links(flight_index, candidates)
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
