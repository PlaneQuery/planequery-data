from datetime import datetime, timedelta

from data_engineering.flights.flight_type import Flight, FlightIndex
from pia.resolver import (
    FlightLinkCandidate,
    candidate_mapping_path_exists,
    check_candidate_mapping_paths,
    create_graph_from_candidates,
    get_flights_connected_to_icao,
)


def _flight(icao: str, hour: int, *, pia: bool = False, ladd: bool = False) -> Flight:
    takeoff_time = datetime(2026, 1, 1, hour)
    return Flight(
        icao=icao,
        takeoff_time=takeoff_time,
        landing_time=takeoff_time + timedelta(minutes=30),
        takeoff_airport_ident="KJFK",
        landing_airport_ident="KBOS",
        pia=pia,
        ladd=ladd,
    )


def test_candidate_graph_mapping_path_can_span_multiple_edges():
    real_flight = _flight("A00001", 1, ladd=True)
    pia_flight_1 = _flight("ABC123", 2, pia=True)
    pia_flight_2 = _flight("ABC123", 3, pia=True)
    unrelated_flight = _flight("A00002", 4, ladd=True)
    flight_index = FlightIndex(
        [real_flight, pia_flight_1, pia_flight_2, unrelated_flight]
    )
    candidates = [
        FlightLinkCandidate(0, real_flight.flight_id, pia_flight_1.flight_id, 10),
        FlightLinkCandidate(1, pia_flight_1.flight_id, pia_flight_2.flight_id, 10),
    ]

    graph = create_graph_from_candidates(candidates, flight_index)

    assert set(graph) == set(flight_index.get_flights_ids())
    assert candidate_mapping_path_exists(graph, flight_index, "ABC123", "A00001")
    assert not candidate_mapping_path_exists(graph, flight_index, "ABC123", "A00002")
    assert get_flights_connected_to_icao(
        candidates, flight_index, "abc123"
    ) == [real_flight, pia_flight_1, pia_flight_2]
    assert get_flights_connected_to_icao(candidates, flight_index, "missing") == []


def test_check_candidate_mapping_paths_reports_missing_icaos():
    real_flight = _flight("A00001", 1, ladd=True)
    pia_flight = _flight("ABC123", 2, pia=True)
    flight_index = FlightIndex([real_flight, pia_flight])
    candidates = [
        FlightLinkCandidate(0, real_flight.flight_id, pia_flight.flight_id, 10),
    ]

    results = check_candidate_mapping_paths(
        candidates,
        flight_index,
        {"ABC123": "A00001", "MISSING": "A00001", "ABC999": "A99999"},
    )

    assert results == {"ABC123": True, "MISSING": False, "ABC999": False}
