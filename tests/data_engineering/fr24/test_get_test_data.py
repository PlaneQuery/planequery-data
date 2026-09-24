from dataclasses import dataclass
from datetime import date

import polars as pl

from airports import runtime
from data_engineering.fr24.get_test_data import fr24_raw_to_flights


@dataclass(frozen=True)
class FakeAirport:
    id: int
    ident: str


@dataclass(frozen=True)
class FakeCatalog:
    flight_date: date

    def resolve_code(self, code, *, code_priority):
        assert code_priority == (
            "icao_code",
            "gps_code",
            "ident",
            "iata_code",
            "local_code",
        )
        offset = 100 if self.flight_date == date(2026, 3, 1) else 200
        return FakeAirport(offset + len(code), f"{code}-{self.flight_date.day}")


def test_fr24_airports_are_resolved_once_per_flight_date(monkeypatch):
    requested_dates = []

    def catalog_for_date(flight_date):
        requested_dates.append(flight_date)
        return FakeCatalog(flight_date)

    monkeypatch.setattr(runtime, "airport_catalog_for_date", catalog_for_date)
    raw = pl.DataFrame(
        {
            "hex": ["ABC123", "DEF456", "AAA111"],
            "callsign": ["ONE", "TWO", "THREE"],
            "datetime_takeoff": [
                "2026-03-01T10:00:00Z",
                "2026-03-01T12:00:00Z",
                "2026-03-02T10:00:00Z",
            ],
            "datetime_landed": [
                "2026-03-01T11:00:00Z",
                "2026-03-01T13:00:00Z",
                "2026-03-02T11:00:00Z",
            ],
            "orig_icao": [" kjfk ", "KJFK", "KJFK"],
            "dest_icao_actual": ["klax", "KBOS", "KLAX"],
            "reg": ["N1", "N2", "N3"],
        }
    )

    flights = fr24_raw_to_flights(raw)

    assert requested_dates == [date(2026, 3, 1), date(2026, 3, 2)]
    assert flights.select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).rows() == [
        (104, "KJFK-1", 104, "KLAX-1"),
        (104, "KJFK-1", 104, "KBOS-1"),
        (204, "KJFK-2", 204, "KLAX-2"),
    ]
