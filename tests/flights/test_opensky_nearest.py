from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from airports.matcher import AirportMatcher
from airports.models import (
    Airport, AirportCatalogProvenance, AirportMatchProvenance, AirportType,
)
from data_engineering.adsb.read_adsb import ADSBSource
from flights import flight_source, opensky_nearest
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.flight_type import Flight, flights_to_flights_df
from flights.providers.opensky import OpenSkyTrinoFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME, OPENSKY_NEAREST_SOURCE


DAY = date(2026, 3, 1)
AIRPORT_COLUMNS = [
    "takeoff_airport_id", "takeoff_airport_ident",
    "landing_airport_id", "landing_airport_ident",
]


@pytest.fixture
def catalog():
    airports = [
        Airport(1, "WEST", AirportType.SMALL_AIRPORT, "West", 40, -75),
        Airport(2, "EAST", AirportType.SMALL_AIRPORT, "East", 40, -70),
    ]
    return SimpleNamespace(
        matcher=AirportMatcher(
            airports,
            {a.id: AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE for a in airports},
            {a.id: "test" for a in airports},
        ),
        provenance=AirportCatalogProvenance(
            DAY, "observed", DAY, "knowledge", DAY, "test",
        ),
    )


def observations(rows):
    return pl.DataFrame(rows, schema={
        "icao": pl.String, "time": pl.Datetime("ms"),
        "lat": pl.Float64, "lon": pl.Float64,
    }, orient="row")


def test_published_segments_batching_overnight_and_cache(monkeypatch, tmp_path, catalog):
    noon = datetime(2026, 3, 1, 12)
    start = datetime(2026, 3, 1, 23, 30)
    end = start + timedelta(hours=1)
    flights = flights_to_flights_df([
        Flight("abc123", start, end, "OLD", "OLD", 900, 901, callsign="NIGHT"),
        Flight("def456", start, end, "OLD", "OLD", 900, 901, callsign="MISSING"),
        Flight("abc123", noon, noon + timedelta(hours=1), "OLD", "OLD", 900, 901),
        Flight("fed321", noon, noon + timedelta(seconds=30), "OLD", "OLD", 900, 901),
    ])
    messages = observations([
        ("abc123", end, 40, -70),
        ("abc123", start, 40, -75),
        ("abc123", noon + timedelta(minutes=30), 40, -75),
        ("fed321", noon, 40, -70),
    ])
    published_reads = []

    def published_day(self, dt, airport_catalog):
        assert dt == DAY
        assert airport_catalog is catalog
        published_reads.append(dt)
        return flights

    reads = []

    def read_observations(*, start, end, source, icaos, columns):
        assert source is ADSBSource.OPENSKY
        assert columns == ["icao", "time", "lat", "lon"]
        reads.append((start, end, set(icaos)))
        return messages.filter(
            pl.col("icao").is_in(icaos)
            & (pl.col("time") >= start) & (pl.col("time") < end)
        )

    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(OpenSkyTrinoFlightSource, "create_flights_for_day", published_day)
    monkeypatch.setattr(opensky_nearest, "FLIGHT_BATCH_SIZE", 2)
    monkeypatch.setattr(opensky_nearest, "read_adsb", read_observations)
    source = OPENSKY_NEAREST_SOURCE
    result = flight_source.get_flights_for_catalog(source, DAY, catalog)
    cached = flight_source.get_flights_for_catalog(source, DAY, catalog)

    assert result.equals(cached)
    assert result.schema == flights.schema
    assert result.drop(AIRPORT_COLUMNS).equals(flights.drop(AIRPORT_COLUMNS))
    assert result.select(AIRPORT_COLUMNS).rows() == [
        (1, "WEST", 2, "EAST"),
        (None, None, None, None),  # Missing aircraft observations.
        (None, None, None, None),  # Only cruise observations for this segment.
        (2, "EAST", 2, "EAST"),  # Short flight's endpoint windows overlap.
    ]
    assert published_reads == [DAY]
    assert reads == [
        (start, end + timedelta(milliseconds=1), {"abc123", "def456"}),
        (noon, noon + timedelta(hours=1, milliseconds=1), {"abc123", "fed321"}),
    ]
    assert FLIGHT_SOURCES_BY_NAME["opensky-nearest"] is source
    path = flight_source.flight_path(source, DAY)
    assert path == (
        tmp_path / "data/flights/source=opensky-nearest/adsb_source=opensky"
        / "version=2/year=2026/month=03/day=01/data.parquet"
    )
    assert FlightManifest.read(flight_manifest_path(path)) == FlightManifest.from_provenance(
        catalog.provenance, adsb_source=ADSBSource.OPENSKY,
    )
    assert pl.read_parquet(
        flight_source.flight_path(OpenSkyTrinoFlightSource(), DAY)
    ).equals(flights)


def test_endpoint_windows_ignore_invalid_positions_and_other_segments(catalog):
    start = datetime(2026, 3, 1, 12)
    end = start + timedelta(hours=1)
    flights = flights_to_flights_df([
        Flight("abc123", start, end, "OLD", "OLD"),
    ])
    messages = observations([
        ("abc123", start - timedelta(milliseconds=1), 40, -70),
        ("abc123", start, float("nan"), -70),
        ("abc123", start + timedelta(seconds=1), 91, -70),
        ("abc123", start + timedelta(seconds=2), 40, float("inf")),
        ("abc123", start + timedelta(seconds=3), 40, 181),
        ("abc123", start + timedelta(minutes=1), 40, -75),
        ("abc123", end - timedelta(minutes=1), 40, -70),
        ("abc123", end, None, -75),
        ("abc123", end + timedelta(milliseconds=1), 40, -75),
        ("def456", start, 40, -70),
    ]).reverse()
    result = opensky_nearest.assign_nearest_airports(flights, messages, catalog)
    assert result.select(AIRPORT_COLUMNS).row(0) == (1, "WEST", 2, "EAST")

    # Moving both usable positions outside the inclusive one-minute windows
    # must not fall back to a cruise position or an original OpenSky label.
    messages = observations([
        ("abc123", start + timedelta(minutes=1, milliseconds=1), 40, -75),
        ("abc123", end - timedelta(minutes=1, milliseconds=1), 40, -70),
    ])
    result = opensky_nearest.assign_nearest_airports(flights, messages, catalog)
    assert result.select(AIRPORT_COLUMNS).row(0) == (None, None, None, None)


def test_empty_observations_keep_flights_with_null_airports(catalog):
    start = datetime(2026, 3, 1, 12)
    flights = flights_to_flights_df([Flight("abc123", start, start, "OLD", "OLD")])
    result = opensky_nearest.assign_nearest_airports(flights, observations([]), catalog)
    assert result.select(AIRPORT_COLUMNS).row(0) == (None, None, None, None)
    assert result.drop(AIRPORT_COLUMNS).equals(flights.drop(AIRPORT_COLUMNS))


def test_empty_day_needs_no_observations(monkeypatch, catalog):
    flights = flights_to_flights_df([])
    monkeypatch.setattr(opensky_nearest, "get_flights_for_catalog", lambda *args: flights)

    def unexpected(*args, **kwargs):
        pytest.fail("Empty flight days must not read observations")

    monkeypatch.setattr(opensky_nearest, "read_adsb", unexpected)
    assert OPENSKY_NEAREST_SOURCE.create_flights_for_day(DAY, catalog).equals(flights)
