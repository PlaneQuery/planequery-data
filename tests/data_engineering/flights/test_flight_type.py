from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest

from flights.flight_type import (
    FLIGHT_POLARS_SCHEMA,
    Flight,
    FlightIndex,
    add_lat_lon_to_airport_columns,
    flights_df_to_flights,
    flights_to_flights_df,
)


def test_flights_df_to_flights_accepts_full_schema_columns():
    df = pl.DataFrame(
        {
            "icao": ["a00001"],
            "callsign": ["TEST1"],
            "registration": ["N1"],
            "takeoff_time": [datetime(2026, 1, 1, 1)],
            "takeoff_airport_id": [3622],
            "takeoff_airport_ident": ["KJFK"],
            "landing_time": [datetime(2026, 1, 1, 2)],
            "landing_airport_id": [3422],
            "landing_airport_ident": ["KBOS"],
            "first_message_time": [datetime(2026, 1, 1, 0, 55)],
            "first_lat": [40.0],
            "first_lon": [-73.0],
            "first_baro_altitude_ft": [50],
            "first_geom_altitude_ft": [60],
            "last_message_time": [datetime(2026, 1, 1, 2, 5)],
            "last_lat": [42.0],
            "last_lon": [-71.0],
            "last_baro_altitude_ft": [100],
            "last_geom_altitude_ft": [200],
            "pia": [False],
            "ladd": [True],
            "military": [False],
            "interesting": [False],
            "aircraft_type": ["C172"],
            "owner": [""],
            "aircraft_description": [""],
            "category": ["A1"],
        },
        schema=FLIGHT_POLARS_SCHEMA,
    )

    [flight] = flights_df_to_flights(df)

    assert flight.first_message_time == datetime(2026, 1, 1, 0, 55)
    assert flight.takeoff_airport_id == 3622
    assert flight.landing_airport_id == 3422
    assert flight.first_lat == 40.0
    assert flight.first_lon == -73.0
    assert flight.first_baro_altitude_ft == 50
    assert flight.first_geom_altitude_ft == 60
    assert flight.last_message_time == datetime(2026, 1, 1, 2, 5)
    assert flight.last_lat == 42.0
    assert flight.last_lon == -71.0
    assert flight.last_baro_altitude_ft == 100
    assert flight.last_geom_altitude_ft == 200


def test_flight_airport_ids_are_nullable():
    flight = Flight(
        icao="a00001",
        takeoff_time=datetime(2026, 1, 1, 1),
        landing_time=datetime(2026, 1, 1, 2),
        takeoff_airport_ident="KJFK",
        landing_airport_ident="KBOS",
    )

    frame = flights_to_flights_df([flight])

    assert frame.schema["takeoff_airport_id"] == pl.Int64
    assert frame.schema["landing_airport_id"] == pl.Int64
    assert frame.select("takeoff_airport_id", "landing_airport_id").row(0) == (
        None,
        None,
    )


def test_add_lat_lon_to_airport_columns_joins_by_stable_id():
    airport_catalog = SimpleNamespace(
        airports=(
            SimpleNamespace(id=3622, lat=40.639801, lon=-73.7789),
            SimpleNamespace(id=3422, lat=42.3643, lon=-71.005203),
        )
    )
    df = pl.DataFrame({
        "takeoff_airport_id": [3622, 9999],
        "takeoff_airport_ident": ["KJFK", "UNKNOWN"],
        "landing_airport_id": [3422, None],
        "landing_airport_ident": ["KBOS", "UNKNOWN"],
    }, schema_overrides={"landing_airport_id": pl.Int64})

    df = add_lat_lon_to_airport_columns(df, airport_catalog)

    assert df.to_dicts() == [
        {
            "takeoff_airport_id": 3622,
            "takeoff_airport_ident": "KJFK",
            "landing_airport_id": 3422,
            "landing_airport_ident": "KBOS",
            "takeoff_airport_lat": 40.639801,
            "takeoff_airport_lon": -73.7789,
            "landing_airport_lat": 42.3643,
            "landing_airport_lon": -71.005203,
        },
        {
            "takeoff_airport_id": 9999,
            "takeoff_airport_ident": "UNKNOWN",
            "landing_airport_id": None,
            "landing_airport_ident": "UNKNOWN",
            "takeoff_airport_lat": None,
            "takeoff_airport_lon": None,
            "landing_airport_lat": None,
            "landing_airport_lon": None,
        },
    ]


def test_flight_index_filters_pia_flights_by_time_and_location():
    start = datetime(2026, 1, 1, 12)
    flights = [
        Flight(
            "ladd", start, start, "KJFK", "KBOS", takeoff_airport_id=1, ladd=True
        ),
        Flight(
            "matching-airport",
            start,
            start,
            "KJFK",
            "KBOS",
            takeoff_airport_id=1,
            pia=True,
        ),
        Flight(
            "nearby-position",
            start,
            start,
            "KTEB",
            "KBOS",
            takeoff_airport_id=2,
            first_lat=40.7,
            first_lon=-74.0,
            pia=True,
        ),
        Flight(
            "lands-too-late",
            start,
            start.replace(hour=14),
            "KJFK",
            "KBOS",
            takeoff_airport_id=1,
            pia=True,
        ),
        Flight(
            "too-far",
            start,
            start,
            "KTEB",
            "KBOS",
            takeoff_airport_id=2,
            first_lat=42.0,
            first_lon=-71.0,
            pia=True,
        ),
        Flight(
            "too-early",
            start.replace(hour=11),
            start,
            "KJFK",
            "KBOS",
            takeoff_airport_id=1,
            pia=True,
        ),
    ]

    index = FlightIndex(flights)

    assert list(index.flights_by_icao) == [
        "ladd",
        "matching-airport",
        "nearby-position",
        "lands-too-late",
        "too-far",
        "too-early",
    ]
    assert index.get_flight_by_id("matching-airport_2026-01-01_12-00") is flights[1]
    with pytest.raises(KeyError):
        index.get_flight_by_id("missing_2026-01-01_12-00")
    assert index.get_flights_ids() == [flight.flight_id for flight in flights]
    assert [flight.icao for flight in index.get_pia_flights()] == [
        "too-early",
        "matching-airport",
        "nearby-position",
        "lands-too-late",
        "too-far",
    ]
    assert [flight.icao for flight in index.get_ladd_flights()] == ["ladd"]
    result = index.pia_flights_after(
        start,
        1,
        before_time=start.replace(hour=13),
    )

    assert [flight.icao for flight in result] == ["matching-airport"]
    assert [flight.icao for flight in index.ladd_flights_after(start, 1)] == [
        "ladd"
    ]
