
from collections import deque
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

def build_candidates(flights: list[Flight]) -> list[FlightLinkCandidate]:
    # sorted by icao, time
    # add scores to each edge. 
    flight_index = FlightIndex(flights)
    candidates = []
    for icao, flights in flight_index.ladd_flights_by_icao.items():
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
            pia_flights = flight_index.pia_flights_after(flight.landing_time, flight.last_lon, flight.last_lat, flight.landing_airport_ident)
            for pia_flight in pia_flights:
                candidate = FlightLinkCandidate(
                    len(candidates),
                    flight.flight_id,
                    pia_flight.flight_id,
                    score = 1
                )
                candidates.append(candidate)

    return candidates
def solve_flights():
    pass

if __name__ == "__main__":
    start_date = date(2026,5,20)
    end_date = date(2026,5,22)
    df_flights = get_flights(start_date, end_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    flights = flights_df_to_flights(df_flights)
    candidates = build_candidates(flights)
    print(len(candidates))