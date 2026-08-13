
from datetime import date, datetime, time, timedelta
from data_engineering.flights.flight_type import Flight, FlightIndex, get_flights
from flights.flights_comparison import Match, df_flights_comparision, df_flights_comparison_stats, flights_comparison
import polars as pl

from pia.graph import FlightLinkCandidate
from pia.resolver import (
    build_candidates,
    filter_candidates,
    mapping_from_selected_candidates,
    solve_links,
    split_candidates_by_component,
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


def test_solve_links_connects_consecutive_real_flights_on_one_path():
    real_flight_1 = c_tf("real", 0, 1, "a", "b")
    connecting_pia = c_tf("pia_connecting", 2, 3, "b", "c")
    decoy_pia = c_tf("pia_decoy", 2, 3, "b", "x")
    real_flight_2 = c_tf("real", 4, 5, "c", "d")
    flight_index = FlightIndex(
        [real_flight_1, connecting_pia, decoy_pia, real_flight_2]
    )
    connecting_candidates = [
        FlightLinkCandidate(
            real_flight_1.flight_id,
            connecting_pia.flight_id,
            10,
        ),
        FlightLinkCandidate(
            connecting_pia.flight_id,
            real_flight_2.flight_id,
            10,
        ),
    ]
    candidates = connecting_candidates + [
        FlightLinkCandidate(
            real_flight_1.flight_id,
            decoy_pia.flight_id,
            100,
        )
    ]

    selected = solve_links(flight_index, candidates)

    assert selected == connecting_candidates


def test_filter_candidates_removes_disconnected_multi_flight_real_icao():
    connected_flights = [
        c_tf("connected", 4, 5, "c", "d"),
        c_tf("connected", 6, 7, "d", "e"),
    ]
    disconnected_flights = [
        c_tf("disconnected", 10, 11, "d", "e"),
        c_tf("disconnected", 12, 13, "f", "g"),
    ]
    singleton = c_tf("singleton", 14, 15, "h", "i")
    flights = test_flights_0 + connected_flights + disconnected_flights + [singleton]
    flight_index = FlightIndex(flights)
    connected_candidate = FlightLinkCandidate(
        connected_flights[0].flight_id,
        connected_flights[1].flight_id,
        1,
    )
    candidates = selected_candidates_0 + [
        connected_candidate,
        FlightLinkCandidate(
            disconnected_flights[0].flight_id,
            test_flights_0[1].flight_id,
            1,
        ),
    ]

    filtered_candidates = filter_candidates(candidates, flight_index)

    assert filtered_candidates == selected_candidates_0 + [connected_candidate]
    assert "disconnected" not in flight_index.flights_by_icao
    assert {
        flight.flight_id for flight in flight_index.get_ladd_flights()
    } == {
        test_flights_0[0].flight_id,
        connected_flights[0].flight_id,
        connected_flights[1].flight_id,
        singleton.flight_id,
    }
    assert singleton.flight_id in flight_index.get_flights_ids()


def test_split_candidates_by_component_groups_original_candidates():
    first_component_flights = [
        c_tf("first_a", 0, 1, "a", "b"),
        c_tf("first_b", 2, 3, "b", "c"),
        c_tf("first_c", 4, 5, "c", "d"),
    ]
    second_component_flights = [
        c_tf("second_a", 6, 7, "e", "f"),
        c_tf("second_b", 8, 9, "f", "g"),
    ]
    isolated_flight = c_tf("isolated", 10, 11, "h", "i")
    flight_index = FlightIndex(
        first_component_flights + second_component_flights + [isolated_flight]
    )

    first_component_candidates = [
        FlightLinkCandidate(
            first_component_flights[0].flight_id,
            first_component_flights[1].flight_id,
            1,
        ),
        FlightLinkCandidate(
            first_component_flights[2].flight_id,
            first_component_flights[1].flight_id,
            2,
        ),
    ]
    duplicate_candidates = [
        FlightLinkCandidate(
            second_component_flights[0].flight_id,
            second_component_flights[1].flight_id,
            score,
        )
        for score in (3, 4)
    ]

    components = split_candidates_by_component(
        first_component_candidates + duplicate_candidates,
        flight_index,
    )

    non_empty_components = [component for component in components if component]
    assert len(components) == 3
    assert first_component_candidates in non_empty_components
    assert duplicate_candidates in non_empty_components
