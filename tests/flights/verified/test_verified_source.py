from datetime import date, datetime, timedelta

import polars as pl
import pytest

from flights.flight_source import NonADSBFlightSource
from flights.providers.bts import BTSFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME
from flights.verified import source
from flights.verified.source import VerifiedFlightSource


def _flights(callsign: str) -> pl.DataFrame:
    start = datetime(2026, 3, 1)
    return pl.DataFrame(
        {
            "icao": ["abc123"] * 4,
            "callsign": [callsign] * 4,
            "takeoff_time": [start + timedelta(hours=2 * i) for i in range(4)],
            "takeoff_airport_ident": ["KAAA", "KBBB", "KCCC", "KDDD"],
            "landing_airport_ident": ["KBBB", "KCCC", "KDDD", "KEEE"],
        }
    )


@pytest.mark.parametrize(
    ("source", "expected_name"),
    [
        (BTSFlightSource(), "bts-verified"),
        (SFDPSFlightSource(), "sfdps-verified"),
    ],
)
def test_verified_source_uses_selected_original_source(
    monkeypatch,
    source,
    expected_name,
):
    target_date = date(2026, 3, 1)
    airport_catalog = object()
    expected = _flights("TEST1")
    calls = []

    def get_source_flights(source_arg, date_arg):
        calls.append((source_arg, date_arg))
        return expected

    monkeypatch.setattr(
        source,
        "get_flights",
        get_source_flights,
    )
    monkeypatch.setattr(
        source,
        "_check_flights_in_us_faa_airspace",
        lambda df, catalog: ["abc123"],
    )
    monkeypatch.setattr(
        source,
        "read_adsb",
        lambda date_arg, *, icaos: pl.DataFrame(),
    )
    monkeypatch.setattr(
        source,
        "get_callsign_history",
        lambda df: pl.DataFrame(
            {"icao": ["abc123"] * 4, "callsign": ["TEST1"] * 4}
        ),
    )

    verified_source = VerifiedFlightSource(source)
    result = verified_source.create_flights_for_day(
        target_date,
        airport_catalog,
    )

    assert verified_source.name == expected_name
    assert verified_source.version == 1
    assert calls == [(source, target_date)]
    assert result.equals(expected)


def test_source_callsigns_must_match_adsb_history():
    flights = _flights("EXPECTED")
    history = pl.DataFrame(
        {
            "icao": ["abc123"] * 4,
            "callsign": ["ONE1", "TWO2", "THREE3", "FOUR4"],
        }
    )

    assert source._validate_callsign_history(flights, history) == []


def test_verified_source_rejects_unsupported_original_source():
    class UnsupportedSource(NonADSBFlightSource):
        def create_flights_for_day(self, dt, airport_catalog):
            raise NotImplementedError

    with pytest.raises(ValueError, match="only supports BTS and SFDPS"):
        VerifiedFlightSource(UnsupportedSource(name="unsupported"))


def test_all_verified_sources_are_registered():
    assert {
        "bts-verified",
        "sfdps-verified",
    } <= FLIGHT_SOURCES_BY_NAME.keys()
    assert "eurocontrol-verified" not in FLIGHT_SOURCES_BY_NAME
