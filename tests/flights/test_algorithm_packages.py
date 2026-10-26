"""Integration contracts for the separately packaged segmentation baselines."""

import pickle
import subprocess
import sys
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from airports.matcher import AirportMatcher
from airports.models import (
    Airport,
    AirportCatalogProvenance,
    AirportMatchProvenance,
    AirportType,
)
from data_engineering.adsb.read_adsb import ADSBSource
from flights import cli, flight_source
from flights.evaluation import evaluation
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.sources import FLIGHT_SOURCES_BY_NAME


DAY = date(2026, 3, 1)
START = datetime(2026, 3, 1)
SOURCE_NAMES = ("openap", "plane-notify", "traffic")


@pytest.fixture
def catalog():
    airports = [
        Airport(1, "FIRST", AirportType.SMALL_AIRPORT, "First", 0.0, 0.0),
        Airport(2, "LAST", AirportType.SMALL_AIRPORT, "Last", 0.0, 1.0),
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


def observations():
    return pl.DataFrame([
        {
            "icao": icao,
            "time": START + timedelta(seconds=second),
            "lat": 0.0,
            "lon": second / 240,
            "callsign": "TEST",
            "registration": "N123",
            "baro_altitude_ft": 10000 if 60 <= second < 180 else 0,
            "geom_altitude_ft": None,
            "ground_speed_kt": 250.0 if 60 <= second < 180 else 0.0,
            "baro_rate_fpm": 0,
            "geom_rate_fpm": None,
            "on_ground": not 60 <= second < 180,
        }
        for icao in ("abc123", "def456")
        for second in range(0, 241, 30)
    ], schema_overrides={"geom_altitude_ft": pl.Int64, "geom_rate_fpm": pl.Int64})


@pytest.mark.parametrize("source_name", SOURCE_NAMES)
def test_source_identity_serialization_and_evaluation_dispatch(monkeypatch, source_name):
    source = FLIGHT_SOURCES_BY_NAME[source_name]
    assert isinstance(source, flight_source.ADSBFlightSource)
    assert source.__class__.__module__ == f"flights.{source_name.replace('-', '_')}"
    restored = pickle.loads(pickle.dumps(source))
    assert restored == source
    assert type(restored) is type(source)
    assert source.version == 1
    assert source.adsb_icao_batch_size() == 500
    assert source.reusable_flight_source() is None
    assert cli.parse_args(["2026-03-01", "2026-03-02", source_name]).source == source_name
    args = evaluation.parse_args([
        "--test-source", source_name, "--adsb-source", "opensky",
    ])
    assert args.test_source == [source_name]
    calls = []
    monkeypatch.setattr(evaluation, "get_flights", lambda *args: calls.append(args))
    evaluation.source_flights_reader(source_name, ADSBSource.OPENSKY)(DAY)
    assert calls == [(source, DAY, ADSBSource.OPENSKY)]


@pytest.mark.parametrize("source_name", SOURCE_NAMES)
@pytest.mark.parametrize("provider", list(ADSBSource))
@pytest.mark.parametrize("target_day", [False, True])
def test_complete_aircraft_batches_match_unbatched_and_reuse_cache(
    monkeypatch, tmp_path, catalog, source_name, provider, target_day,
):
    if source_name == "openap":
        pytest.importorskip("openap.phase")
    if source_name == "traffic":
        pytest.importorskip("traffic.core")
    source_type = type(FLIGHT_SOURCES_BY_NAME[source_name])
    source = source_type(target_day_adsb=target_day)
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(flight_source, "airport_catalog_for_date", lambda dt: catalog)
    monkeypatch.setattr(source_type, "adsb_icao_batch_size", lambda self: 1)
    frame = observations()
    expected = source.create_flights_for_day(DAY, frame, catalog)
    assert expected["icao"].to_list() == ["abc123", "def456"]
    reads = []

    def read(start, end, *, source, columns, lazy=False, icaos=None):
        reads.append((start, end, source, lazy, icaos))
        if lazy:
            assert columns == ["icao"]
            return frame.select("icao").lazy()
        assert len(icaos) == 1
        return frame.filter(pl.col("icao").is_in(icaos)).select(columns)

    monkeypatch.setattr(flight_source, "read_adsb", read)
    actual = flight_source.get_flights(source, DAY, provider)
    assert actual.equals(expected)
    context = timedelta(days=int(not target_day and provider is not ADSBSource.ADSBX))
    window = (START - context, START + timedelta(days=1) + context, provider)
    assert reads == [
        (*window, True, None),
        (*window, False, ["abc123"]),
        (*window, False, ["def456"]),
    ]
    path = flight_source.flight_path(source, DAY, provider)
    assert f"source={source_name}/adsb_source={provider.value}/" in str(path)
    assert ("target_day_adsb=true" in str(path)) is target_day
    assert "/version=1/" in str(path)
    assert FlightManifest.read(flight_manifest_path(path)) == source.flight_manifest(
        catalog.provenance, provider,
    )
    assert flight_source.get_flights(source, DAY, provider).equals(actual)
    assert len(reads) == 3


def test_registry_does_not_require_optional_openap_dependency():
    # Use a fresh interpreter so an already-imported detector cannot hide an
    # accidental eager dependency introduced by the package/registry imports.
    code = '''
import builtins
original_import = builtins.__import__
def without_openap(name, *args, **kwargs):
    if name == "openap" or name.startswith("openap."):
        raise ModuleNotFoundError("OpenAP intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_openap
from flights.sources import FLIGHT_SOURCES_BY_NAME
from flights.openap.segmentation import _phase_detector
assert "openap" in FLIGHT_SOURCES_BY_NAME
assert "plane-notify" in FLIGHT_SOURCES_BY_NAME
assert "sun" in FLIGHT_SOURCES_BY_NAME
try:
    _phase_detector()
except ImportError as exc:
    assert "pip install" in str(exc)
else:
    raise AssertionError("Missing dependency must fail only on detector use")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
