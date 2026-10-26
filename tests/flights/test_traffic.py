"""Upstream parity and canonical boundary contracts for the Traffic source."""

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.traffic import segmentation


DAY = date(2026, 3, 1)
START = datetime(2026, 3, 1)


def observations(seconds, callsigns=None, icao="abc123", start=START):
    return pl.DataFrame({
        "icao": [icao] * len(seconds),
        "time": [start + timedelta(seconds=s) for s in seconds],
        "callsign": callsigns if callsigns is not None else ["TEST"] * len(seconds),
        "registration": [None] * len(seconds),
        "lat": [float(i) for i in range(len(seconds))],
        "lon": [0.0] * len(seconds),
        "baro_altitude_ft": [1000] * len(seconds),
        "geom_altitude_ft": [None] * len(seconds),
    }, schema_overrides={"registration": pl.String, "callsign": pl.String,
                         "geom_altitude_ft": pl.Int64})


@pytest.fixture
def upstream():
    return pytest.importorskip("traffic.core").Traffic


def test_default_iterator_parity(upstream):
    # Exact ten-minute gap, >ten-minute gap, recurring callsign, missing
    # callsign, singleton, duplicate timestamp, and a second aircraft.
    frame = pl.concat([
        observations([0, 600, 1201, 1202, 2400]),
        observations([10, 20, 30, 40, 50, 60], ["A", "B", "A", "B", None, None]),
        observations([100, 100], ["DUP", "DUP"]),
        observations([0, 30], icao="def456"),
    ]).reverse()
    data = frame.rename({"icao": "icao24", "time": "timestamp"}).to_pandas()
    expected = sorted(
        (f.data.icao24.iloc[0], f.data.callsign.iloc[0],
         f.start.to_pydatetime(), f.stop.to_pydatetime())
        for f in upstream(data).iterate()
    )
    flights = segmentation.create_flights_from_adsb(frame, DAY)
    assert sorted(flights.select(
        "icao", "callsign", "takeoff_time", "landing_time",
    ).iter_rows()) == expected
    assert flights.schema == FLIGHT_POLARS_SCHEMA
    assert flights.height == 6
    assert flights.filter(pl.col("callsign") == "TEST")["landing_time"].to_list() == [
        START + timedelta(seconds=600), START + timedelta(seconds=1202),
        START + timedelta(seconds=30),
    ]


def test_ground_reports_and_callsign_gaps_are_not_reinterpreted(upstream):
    frame = observations([0, 30, 60, 90, 120]).with_columns(
        pl.Series("on_ground", [False, True, True, True, False]),
        pl.Series("baro_altitude_ft", [1000, None, None, None, 1000]),
    )
    flights = segmentation.create_flights_from_adsb(frame, DAY)
    assert flights.height == 1
    assert flights["landing_time"][0] == START + timedelta(seconds=120)
    # Null callsigns are excluded by upstream pandas groupby; empty strings
    # are a real group. Do not invent or forward-fill either callsign.
    frame = observations([0, 30, 60, 90], [None, None, "", ""])
    flights = segmentation.create_flights_from_adsb(frame, DAY)
    assert flights["takeoff_time"].to_list() == [START + timedelta(seconds=60)]


def test_midnight_and_utc_day_ownership(upstream):
    frame = pl.concat([
        observations([0, 60, 120], start=START - timedelta(seconds=60)),
        observations([0, 60, 120], start=START + timedelta(days=1, seconds=-60)),
    ])
    flights = segmentation.create_flights_from_adsb(frame, DAY)
    assert flights["takeoff_time"].to_list() == [START + timedelta(days=1, seconds=-60)]
    assert flights["landing_time"][0] == START + timedelta(days=1, seconds=60)
    zoned = observations([0, 60], start=datetime(
        2026, 2, 28, 19, tzinfo=timezone(timedelta(hours=-5)),
    ))
    assert segmentation.create_flights_from_adsb(zoned, DAY)["takeoff_time"][0] == START


def test_missing_endpoint_values_and_registration(upstream):
    frame = observations([0, 30, 60]).with_columns(
        pl.Series("lat", [None, 1.0, float("nan")]),
        pl.Series("registration", [None, "N123", "N456"]),
        pl.Series("baro_altitude_ft", [None, 1000, None], dtype=pl.Int64),
    )
    flight = segmentation.create_flights_from_adsb(frame, DAY).row(0, named=True)
    assert flight["registration"] == "N123"
    for field in ("first_lat", "last_lat", "first_baro_altitude_ft",
                  "last_baro_altitude_ft", "first_geom_altitude_ft"):
        assert flight[field] is None


@pytest.mark.parametrize("frame", [pl.DataFrame(), observations([0]).with_columns(
    pl.lit(None, dtype=pl.String).alias("icao"),
)])
def test_empty_or_unidentifiable_input_needs_no_optional_dependency(monkeypatch, frame):
    def unavailable():
        raise AssertionError("Empty input must not load Traffic")
    monkeypatch.setattr(segmentation, "_traffic_class", unavailable)
    assert segmentation.create_flights_from_adsb(frame, DAY).schema == FLIGHT_POLARS_SCHEMA


def test_singleton_and_null_callsign_groups_follow_upstream(upstream):
    for frame in (observations([0]), observations([0, 30], [None, None])):
        result = segmentation.create_flights_from_adsb(frame, DAY)
        assert result.is_empty()
        assert result.schema == FLIGHT_POLARS_SCHEMA


def test_version_mismatch_fails_before_using_different_algorithm(upstream, monkeypatch):
    monkeypatch.setattr(segmentation, "version", lambda name: "0.0")
    with pytest.raises(RuntimeError, match="flight-source version bump"):
        segmentation.create_flights_from_adsb(observations([0, 30]), DAY)


def test_registry_import_without_traffic():
    code = '''
import builtins
original_import = builtins.__import__
def without_traffic(name, *args, **kwargs):
    if name == "traffic" or name.startswith("traffic."):
        raise ModuleNotFoundError("Traffic intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_traffic
from flights.sources import FLIGHT_SOURCES_BY_NAME
from flights.traffic.segmentation import _traffic_class
assert "traffic" in FLIGHT_SOURCES_BY_NAME
try:
    _traffic_class()
except ImportError as exc:
    assert 'pip install -e ".[traffic]"' in str(exc)
else:
    raise AssertionError("Missing Traffic must fail only when used")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
