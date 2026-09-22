from datetime import datetime, timedelta

from flights.flight_type import Flight, FlightIndex
from pia.graph import FlightLinkCandidate
from pia.solver import (
    mapping_from_selected_candidates,
    resolve_links_with_margin,
    solve_links,
)


START_TIME = datetime(2026, 1, 1)


def pia_flight(icao: str, hour: int) -> Flight:
    return Flight(
        icao=icao,
        takeoff_time=START_TIME + timedelta(hours=hour),
        landing_time=START_TIME + timedelta(hours=hour + 1),
        takeoff_airport_ident="AAA",
        landing_airport_ident="BBB",
        pia=True,
    )


def seeded_candidate_graph():
    known_a = pia_flight("pia_known_a", 0)
    unknown = pia_flight("pia_unknown", 2)
    known_b = pia_flight("pia_known_b", 4)
    flight_index = FlightIndex([known_a, unknown, known_b])
    candidates = [
        FlightLinkCandidate(known_a.flight_id, unknown.flight_id, 10),
        FlightLinkCandidate(unknown.flight_id, known_b.flight_id, 9),
    ]
    known_mapping = {
        known_a.icao: "real_a",
        known_b.icao: "real_b",
    }
    return flight_index, candidates, known_mapping, unknown


def test_solve_links_uses_known_mapping_as_hard_constraints():
    flight_index, candidates, known_mapping, _ = seeded_candidate_graph()

    assert solve_links(flight_index, candidates) == candidates
    assert solve_links(
        flight_index,
        candidates,
        known_mapping=known_mapping,
    ) == [candidates[0]]


def test_known_mapping_resolves_unknown_pia_without_ladd_anchor():
    flight_index, candidates, known_mapping, unknown = seeded_candidate_graph()
    selected = solve_links(
        flight_index,
        candidates,
        known_mapping=known_mapping,
    )

    mapping = mapping_from_selected_candidates(
        selected,
        flight_index,
        known_mapping=known_mapping,
    )

    assert mapping[unknown] == "real_a"


def test_resolve_links_with_margin_threads_known_mapping():
    flight_index, candidates, known_mapping, unknown = seeded_candidate_graph()

    selected, mapping, _ = resolve_links_with_margin(
        flight_index,
        candidates,
        known_mapping=known_mapping,
    )

    assert selected == [candidates[0]]
    assert mapping[unknown] == "real_a"
    assert {
        flight.icao: real_icao
        for flight, real_icao in mapping.items()
        if flight.icao in known_mapping
    } == known_mapping
