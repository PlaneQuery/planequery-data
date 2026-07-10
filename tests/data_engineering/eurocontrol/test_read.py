from datetime import date, datetime

import polars as pl

import data_engineering.eurocontrol.read as eurocontrol_read
from data_engineering.eurocontrol.read import (
    create_eurocontrol_flights,
    eurocontrol_flights_parquet_path,
    eurocontrol_raw_flights_parquet_path,
    normalize_eurocontrol_flights,
    read_eurocontrol_flights,
)


class FakeAirport:
    def __init__(self, ident: str) -> None:
        self.ident = ident


class FakeAirportLookup:
    iata_to_ident_index = {"JFK": "KJFK"}
    gps_code_to_ident_index = {"EGLL": "EGLL", "KJFK": "KJFK", "KABC": "KABC"}
    local_code_to_ident_index = {"ABC": "KABC"}

    def get_Airport_from_airport_ident(self, airport_ident):
        if airport_ident in {"KJFK", "EGLL"}:
            return FakeAirport(airport_ident)
        return None


def _raw_eurocontrol_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "AC Registration": [" n-123ab ", "G-ABCD", "UNKNOWN"],
            "ADEP": ["JFK", "EGLL", "JFK"],
            "ADES": ["EGLL", "ABC", "EGLL"],
            "AC Type": ["B738", "A320", "GLF5"],
            "AC Operator": ["OP1", "OP2", "OP3"],
            "ACTUAL OFF BLOCK TIME": [
                datetime(2024, 6, 1, 1, 2),
                datetime(2024, 6, 2, 3, 4),
                datetime(2024, 6, 1, 9, 10),
            ],
            "ACTUAL ARRIVAL TIME": [
                datetime(2024, 6, 1, 5, 6),
                datetime(2024, 6, 2, 7, 8),
                datetime(2024, 6, 1, 11, 12),
            ],
        }
    )


def test_eurocontrol_raw_flights_parquet_path_uses_month_extract(monkeypatch, tmp_path):
    monkeypatch.setattr(eurocontrol_read, "OUTPUT_DIR", tmp_path)

    path = eurocontrol_raw_flights_parquet_path(date(2024, 6, 15))

    assert path == (
        tmp_path
        / "data"
        / "raw"
        / "eurocontrol"
        / "202406"
        / "intermediate"
        / "flights_20240601_20240630_time_columns.parquet"
    )


def test_eurocontrol_flights_parquet_path_uses_normalized_month_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(eurocontrol_read, "OUTPUT_DIR", tmp_path)

    path = eurocontrol_flights_parquet_path(date(2024, 6, 15))

    assert path == (
        tmp_path
        / "data"
        / "flights"
        / "eurocontrol"
        / "year=2024"
        / "month=06"
        / "data.parquet"
    )


def test_normalize_eurocontrol_flights_maps_to_flight_schema():
    registration_icao = pl.DataFrame(
        {
            "registration": ["N123AB", "GABCD"],
            "icao": ["a12345", "400abc"],
        }
    )

    df = normalize_eurocontrol_flights(
        _raw_eurocontrol_df(),
        registration_icao=registration_icao,
        airport_lookup=FakeAirportLookup(),
    )

    assert df.to_dicts() == [
        {
            "icao": "a12345",
            "callsign": "",
            "registration": "N-123AB",
            "takeoff_time": datetime(2024, 6, 1, 1, 2),
            "takeoff_airport_ident": "KJFK",
            "landing_time": datetime(2024, 6, 1, 5, 6),
            "landing_airport_ident": "EGLL",
            "first_message_time": None,
            "first_lat": None,
            "first_lon": None,
            "first_baro_altitude_ft": None,
            "first_geom_altitude_ft": None,
            "last_message_time": None,
            "last_lat": None,
            "last_lon": None,
            "last_baro_altitude_ft": None,
            "last_geom_altitude_ft": None,
            "pia": False,
            "ladd": False,
            "military": False,
            "interesting": False,
            "aircraft_type": "B738",
            "owner": "OP1",
            "aircraft_description": "",
            "category": "",
        },
        {
            "icao": "400abc",
            "callsign": "",
            "registration": "G-ABCD",
            "takeoff_time": datetime(2024, 6, 2, 3, 4),
            "takeoff_airport_ident": "EGLL",
            "landing_time": datetime(2024, 6, 2, 7, 8),
            "landing_airport_ident": "KABC",
            "first_message_time": None,
            "first_lat": None,
            "first_lon": None,
            "first_baro_altitude_ft": None,
            "first_geom_altitude_ft": None,
            "last_message_time": None,
            "last_lat": None,
            "last_lon": None,
            "last_baro_altitude_ft": None,
            "last_geom_altitude_ft": None,
            "pia": False,
            "ladd": False,
            "military": False,
            "interesting": False,
            "aircraft_type": "A320",
            "owner": "OP2",
            "aircraft_description": "",
            "category": "",
        },
        {
            "icao": None,
            "callsign": "",
            "registration": "UNKNOWN",
            "takeoff_time": datetime(2024, 6, 1, 9, 10),
            "takeoff_airport_ident": "KJFK",
            "landing_time": datetime(2024, 6, 1, 11, 12),
            "landing_airport_ident": "EGLL",
            "first_message_time": None,
            "first_lat": None,
            "first_lon": None,
            "first_baro_altitude_ft": None,
            "first_geom_altitude_ft": None,
            "last_message_time": None,
            "last_lat": None,
            "last_lon": None,
            "last_baro_altitude_ft": None,
            "last_geom_altitude_ft": None,
            "pia": False,
            "ladd": False,
            "military": False,
            "interesting": False,
            "aircraft_type": "GLF5",
            "owner": "OP3",
            "aircraft_description": "",
            "category": "",
        },
    ]


def test_create_eurocontrol_flights_writes_normalized_cache(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "data" / "flights" / "eurocontrol" / "year=2024" / "month=06" / "data.parquet"
    _raw_eurocontrol_df().write_parquet(raw_path)
    registration_icao = pl.DataFrame(
        {
            "registration": ["N123AB", "GABCD"],
            "icao": ["a12345", "400abc"],
        }
    )

    monkeypatch.setattr(
        eurocontrol_read,
        "normalize_eurocontrol_flights",
        lambda df: normalize_eurocontrol_flights(
            df,
            registration_icao=registration_icao,
            airport_lookup=FakeAirportLookup(),
        ),
    )

    df = create_eurocontrol_flights(
        date(2024, 6, 1),
        raw_path=raw_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert df.select("icao").to_series().to_list() == ["a12345", "400abc"]
    assert pl.read_parquet(output_path).select("icao").to_series().to_list() == ["a12345", "400abc"]


def test_read_eurocontrol_flights_uses_cache_and_filters_to_target_takeoff_date(monkeypatch, tmp_path):
    monkeypatch.setattr(eurocontrol_read, "OUTPUT_DIR", tmp_path)
    cache_path = eurocontrol_flights_parquet_path(date(2024, 6, 1))
    cache_path.parent.mkdir(parents=True)
    normalize_eurocontrol_flights(
        _raw_eurocontrol_df(),
        registration_icao=pl.DataFrame(
            {
                "registration": ["N123AB", "GABCD"],
                "icao": ["a12345", "400abc"],
            }
        ),
        airport_lookup=FakeAirportLookup(),
    ).filter(pl.col("icao").is_not_null()).write_parquet(cache_path)

    df = read_eurocontrol_flights(date(2024, 6, 1))

    assert df.select("icao").to_series().to_list() == ["a12345"]
