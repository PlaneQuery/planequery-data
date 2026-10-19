from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from airports.models import Airport, AirportCatalogProvenance, AirportType
from data_engineering.adsb.canonical import ADSB_PARQUET_VERSIONS
from data_engineering.adsb.read_adsb import ADSBSource
from flights import flight_source
from flights.airport_nn import model, opensky
from flights.airport_nn.features import COLUMNS
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.flight_type import Flight, flights_to_flights_df
from flights.providers.opensky import OpenSkyTrinoFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME, OPENSKY_NN_SOURCE


DAY = date(2026, 3, 1)
AIRPORT_COLUMNS = [
    "takeoff_airport_id", "takeoff_airport_ident",
    "landing_airport_id", "landing_airport_ident",
]


@pytest.fixture
def catalog():
    airport = Airport(
        id=1, ident="TEST", type=AirportType.SMALL_AIRPORT,
        name="Test", lat=40, lon=-75, elevation_ft=100,
    )
    return SimpleNamespace(
        matcher=SimpleNamespace(
            airports_within_radius=lambda lat, lon, radius: [(airport, 0.0)],
        ),
        provenance=AirportCatalogProvenance(
            flight_date=DAY,
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=DAY,
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=DAY,
            policy_version="test",
        ),
    )


def test_opensky_nn_preserves_published_flights_and_infers_in_batches(
    monkeypatch, tmp_path, catalog,
):
    noon = datetime(2026, 3, 1, 12)
    start = datetime(2026, 3, 1, 23, 30)
    end = start + timedelta(hours=1)
    flights = flights_to_flights_df([
        Flight("abc123", start, end, "OLD", "OLD", 900, 901, callsign="NIGHT"),
        Flight("def456", start, end, "OLD", "OLD", 900, 901, callsign="MISSING"),
        Flight("abc123", noon, noon + timedelta(hours=1), "OLD", "OLD", 900, 901),
        Flight("fed321", noon, noon + timedelta(minutes=5), "OLD", "OLD", 900, 901),
    ])
    messages = pl.DataFrame({
        "icao": ["abc123", "abc123", "abc123", "fed321"],
        "time": [start, end, noon + timedelta(minutes=30), noon],
        "lat": [40.0] * 4,
        "lon": [-75.0] * 4,
        "baro_altitude_ft": [100.0] * 4,
        "ground_speed_kt": [20.0] * 4,
        "track_deg": [90.0] * 4,
        "baro_rate_fpm": [0.0] * 4,
        "on_ground": [True] * 4,
    }).with_columns(pl.col("time").cast(pl.Datetime("ms")))
    published_reads = []

    def published_day(self, dt, airport_catalog):
        assert dt == DAY
        assert airport_catalog is catalog
        published_reads.append(dt)
        return flights

    reads = []

    def read_observations(*, start, end, source, icaos, columns):
        assert source is ADSBSource.OPENSKY
        assert columns == COLUMNS
        reads.append((start, end, set(icaos)))
        return messages.filter(
            pl.col("icao").is_in(icaos)
            & (pl.col("time") >= start)
            & (pl.col("time") < end)
        ).select(columns)

    loads = []

    def load_model(path):
        loads.append(path)
        return model.AirportRanker(8).eval()

    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(OpenSkyTrinoFlightSource, "create_flights_for_day", published_day)
    monkeypatch.setattr(opensky, "FLIGHT_BATCH_SIZE", 2)
    monkeypatch.setattr(opensky, "read_adsb", read_observations)
    monkeypatch.setattr(model, "load_model", load_model)
    source = opensky.OpenSkyNNFlightSource(model_path=tmp_path / "model.pt")

    result = flight_source.get_flights_for_catalog(source, DAY, catalog)
    cached = flight_source.get_flights_for_catalog(source, DAY, catalog)

    assert result.equals(cached)
    assert result.schema == flights.schema
    assert result.drop(AIRPORT_COLUMNS).equals(flights.drop(AIRPORT_COLUMNS))
    assert result.select(AIRPORT_COLUMNS).rows() == [
        (1, "TEST", 1, "TEST"),
        (None, None, None, None),  # No observations for this aircraft.
        (None, None, None, None),  # Only cruise observations for this segment.
        (1, "TEST", 1, "TEST"),  # Endpoint windows overlap on a short flight.
    ]
    assert published_reads == [DAY]
    assert loads == [source.model_path]
    assert reads == [
        (start, end + timedelta(milliseconds=1), {"abc123", "def456"}),
        (noon, noon + timedelta(hours=1, milliseconds=1), {"abc123", "fed321"}),
    ]

    assert FLIGHT_SOURCES_BY_NAME["opensky-nn"] is OPENSKY_NN_SOURCE
    path = flight_source.flight_path(source, DAY)
    assert path == (
        tmp_path / "data/flights/source=opensky-nn/adsb_source=opensky"
        / "version=1/year=2026/month=03/day=01/data.parquet"
    )
    manifest = FlightManifest.read(flight_manifest_path(path))
    assert manifest == FlightManifest.from_provenance(
        catalog.provenance, adsb_source=ADSBSource.OPENSKY,
    )
    assert manifest.adsb_parquet_version == ADSB_PARQUET_VERSIONS[ADSBSource.OPENSKY]
    # Airport inference never overwrites the published OpenSky cache.
    assert pl.read_parquet(
        flight_source.flight_path(OpenSkyTrinoFlightSource(), DAY)
    ).equals(flights)


def test_empty_opensky_day_needs_no_observations_or_model(monkeypatch, catalog):
    flights = flights_to_flights_df([])
    monkeypatch.setattr(opensky, "get_flights_for_catalog", lambda *args: flights)

    def unexpected(*args, **kwargs):
        pytest.fail("Empty flight days must not read observations or load the model")

    monkeypatch.setattr(opensky, "read_adsb", unexpected)
    monkeypatch.setattr(model, "load_model", unexpected)
    result = opensky.OpenSkyNNFlightSource().create_flights_for_day(DAY, catalog)
    assert result.equals(flights)
