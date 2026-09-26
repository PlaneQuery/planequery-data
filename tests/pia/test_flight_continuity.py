from datetime import datetime
from types import SimpleNamespace

from airports.models import AirportType
from flights.flight_type import Flight
from pia import flight_continuity
from pia.flight_continuity import AIRPORT_SIZE_RANK


def test_airport_size_rank_covers_every_airport_type():
    assert set(AIRPORT_SIZE_RANK) == set(AirportType)


def test_make_flights_continuous_skips_unresolved_airports(monkeypatch):
    monkeypatch.setattr(
        flight_continuity,
        "airport_catalog_for_date",
        lambda _: SimpleNamespace(id_index={}),
    )
    flights = [
        Flight(
            icao="abc123",
            takeoff_time=datetime(2026, 3, 1, 10),
            landing_time=datetime(2026, 3, 1, 11),
            takeoff_airport_id=None,
            takeoff_airport_ident=None,
            landing_airport_id=None,
            landing_airport_ident=None,
        ),
        Flight(
            icao="abc123",
            takeoff_time=datetime(2026, 3, 1, 12),
            landing_time=datetime(2026, 3, 1, 13),
            takeoff_airport_id=999,
            takeoff_airport_ident="UNKNOWN",
            landing_airport_id=999,
            landing_airport_ident="UNKNOWN",
        ),
    ]

    assert flight_continuity.make_flights_continuous(flights) == []
