from datetime import datetime

import polars as pl

from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, flights_df_to_flights


def test_flights_df_to_flights_accepts_full_schema_columns():
    df = pl.DataFrame(
        {
            "icao": ["a00001"],
            "callsign": ["TEST1"],
            "registration": ["N1"],
            "takeoff_time": [datetime(2026, 1, 1, 1)],
            "takeoff_airport_ident": ["KJFK"],
            "takeoff_airport_ident_distance_km": [1.5],
            "landing_time": [datetime(2026, 1, 1, 2)],
            "landing_airport_ident": ["KBOS"],
            "landing_airport_ident_distance_km": [2.5],
            "first_lat": [40.0],
            "first_lon": [-73.0],
            "first_baro_altitude_ft": [50],
            "first_geom_altitude_ft": [60],
            "last_lat": [42.0],
            "last_lon": [-71.0],
            "last_baro_altitude_ft": [100],
            "last_geom_altitude_ft": [200],
            "takeoff_airport_elevation_difference_ft": [10],
            "landing_airport_elevation_difference_ft": [20],
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

    assert flight.first_lat == 40.0
    assert flight.first_lon == -73.0
    assert flight.first_baro_altitude_ft == 50
    assert flight.first_geom_altitude_ft == 60
    assert flight.last_lat == 42.0
    assert flight.last_lon == -71.0
    assert flight.last_baro_altitude_ft == 100
    assert flight.last_geom_altitude_ft == 200
