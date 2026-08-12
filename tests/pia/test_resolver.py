
from datetime import date, datetime, time, timedelta
from data_engineering.flights.flight_type import Flight, FlightIndex, get_flights
from flights.flights_comparison import Match, df_flights_comparision, df_flights_comparison_stats, flights_comparison
import polars as pl

from pia.resolver import (
    FlightLinkCandidate,
    build_candidates,
    mapping_from_selected_candidates,
    solve_links,
)

start_time = datetime(2026, 1, 1, 0, 0, 0)
def c_tf(icao: str, takeoff_time: int, landing_time: int, takeoff_airport_ident: str, landing_airport_ident: str) -> Flight:
    if icao.startswith("pia"):
        ladd = False
        pia = True
    else:
        ladd = True
        pia = False
    takeoff_time_dt = start_time + timedelta(hours=takeoff_time)
    landing_time_dt = start_time + timedelta(hours=landing_time)
    flight = Flight(icao, takeoff_time_dt, landing_time_dt, takeoff_airport_ident, landing_airport_ident, pia=pia, ladd=ladd, first_lat = 0.0, first_lon=0.0, last_lat=0.0, last_lon=0.0, first_baro_altitude_ft=0, first_geom_altitude_ft=0, last_baro_altitude_ft=0, last_geom_altitude_ft=0) # TODO: make first/last points auto. Randomely select airports from airprots (with static seed.)
    return flight

# then we'll also provide a map like our ground_truth mappings/ground control does. 
test_mappings_0 = {
    "pia_a": "a"
}
test_flights_0 = [ # just a subgraph filter will get this
    c_tf("a", 0, 1, "a", "b"),
    c_tf("pia_a", 2, 3, "b", "c"),
    c_tf("pia_b", 8, 9, "b", "c"),

]
selected_candidates_0 = [ # How to comapre these. 
    FlightLinkCandidate(test_flights_0[0].flight_id, test_flights_0[1].flight_id, 1)
]


def test_solve_links_selects_valid_candidate():
    flight_index = FlightIndex(test_flights_0)

    selected = solve_links(flight_index, selected_candidates_0)

    assert selected == selected_candidates_0


def test_build_candidates_selects_correct_pia_mapping():
    flight_index = FlightIndex(test_flights_0)
    candidates = build_candidates(flight_index)

    selected = solve_links(flight_index, candidates)
    mapping = mapping_from_selected_candidates(selected, flight_index)

    assert [candidate.next_flight_id for candidate in selected] == [
        test_flights_0[1].flight_id
    ]
    assert {flight.icao: real_icao for flight, real_icao in mapping.items()} == test_mappings_0
