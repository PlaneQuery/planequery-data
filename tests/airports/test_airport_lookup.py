import os
from dataclasses import replace
from pathlib import Path

import pytest


os.environ["OUTPUT_DIR"] = str(Path(__file__).resolve().parents[2])

from airports.airport_lookup import AirportLookup


@pytest.fixture
def airport_lookup():
    AirportLookup._instance = None
    return AirportLookup()


@pytest.mark.parametrize("airport_ident", ["PHNL", "PANC", "TJSJ"])
def test_airport_ident_is_in_us_faa_island_airspace(airport_lookup, airport_ident):
    assert airport_lookup.airport_ident_is_in_us_faa_island_airspace(airport_ident)


@pytest.mark.parametrize("airport_ident", ["KJFK", "CYVR", "DOES_NOT_EXIST"])
def test_airport_ident_is_not_in_us_faa_island_airspace(airport_lookup, airport_ident):
    assert not airport_lookup.airport_ident_is_in_us_faa_island_airspace(airport_ident)


def test_airport_contains_all_csv_fields(airport_lookup):
    airport = airport_lookup.get_Airport_from_airport_ident("KJFK")

    assert airport is not None
    assert airport.id == 3622
    assert airport.ident == "KJFK"
    assert airport.type == "large_airport"
    assert airport.name == "John F. Kennedy International Airport"
    assert airport.latitude_deg == pytest.approx(40.639447)
    assert airport.longitude_deg == pytest.approx(-73.779317)
    assert airport.elevation_ft == 13
    assert airport.continent == "NA"
    assert airport.iso_country == "US"
    assert airport.iso_region == "US-NY"
    assert airport.municipality == "New York"
    assert airport.scheduled_service == "yes"
    assert airport.icao_code == "KJFK"
    assert airport.iata_code == "JFK"
    assert airport.gps_code == "KJFK"
    assert airport.local_code == "JFK"
    assert airport.home_link == "https://www.jfkairport.com/"
    assert airport.wikipedia_link == (
        "https://en.wikipedia.org/wiki/John_F._Kennedy_International_Airport"
    )
    assert airport.keywords == "Manhattan, New York City, NYC, Idlewild, IDL, KIDL"
    assert airport.lat == airport.latitude_deg
    assert airport.lon == airport.longitude_deg


def test_blank_airport_csv_fields_are_none(airport_lookup):
    airport = airport_lookup.get_Airport_from_airport_ident("00AA")

    assert airport is not None
    assert airport.icao_code is None
    assert airport.iata_code is None
    assert airport.home_link is None
    assert airport.wikipedia_link is None
    assert airport.keywords is None


def test_airport_equality_and_hash_use_ident(airport_lookup):
    airport = airport_lookup.get_Airport_from_airport_ident("KJFK")

    assert airport is not None
    same_ident_with_different_data = replace(
        airport,
        name="Different name",
        latitude_deg=0.0,
        distance_from_plane_point_in_km=100.0,
    )
    other_airport = airport_lookup.get_Airport_from_airport_ident("KLAX")

    assert airport == same_ident_with_different_data
    assert hash(airport) == hash(same_ident_with_different_data)
    assert len({airport, same_ident_with_different_data}) == 1
    assert airport != other_airport


def test_airport_distance_to_returns_kilometers(airport_lookup):
    airport = airport_lookup.get_Airport_from_airport_ident("KJFK")

    assert airport is not None
    assert airport.distance_to(airport.latitude_deg, airport.longitude_deg) == pytest.approx(0.0)
    assert airport.distance_to(33.9425, -118.408056) == pytest.approx(3974, abs=10)


@pytest.mark.parametrize(
    ("code", "expected_ident"),
    [
        ("7GA7", "K4A0"),
        ("ALM", "KALM"),
        ("IA32", "IA32"),
        ("IA8", "IA8"),
        ("LS84", "US-6197"),
        ("PA39", "PA39"),
    ],
)
def test_get_airport_from_code_resolves_sfdps_collisions(
    airport_lookup,
    code,
    expected_ident,
):
    airport = airport_lookup.get_airport_from_code(code)

    assert airport is not None
    assert airport.ident == expected_ident


def test_get_airport_from_code_uses_coordinates_as_final_tiebreaker(airport_lookup):
    expected = airport_lookup.get_Airport_from_airport_ident("US-4767")

    assert expected is not None
    airport = airport_lookup.get_airport_from_code(
        "IA32",
        lat=expected.latitude_deg,
        lon=expected.longitude_deg,
        code_priority=(),
    )
    assert airport == expected


def test_get_airport_from_code_returns_none_for_unknown_code(airport_lookup):
    assert airport_lookup.get_airport_from_code("DOES_NOT_EXIST") is None


def test_get_airport_from_code_requires_both_coordinates(airport_lookup):
    with pytest.raises(ValueError, match="lat and lon must either both be provided"):
        airport_lookup.get_airport_from_code("KJFK", lat=40.0)


def test_get_airport_from_code_raises_when_collision_remains(airport_lookup):
    with pytest.raises(ValueError, match="Multiple airports found for code 'O48'"):
        airport_lookup.get_airport_from_code("O48")

def test_get_airport_from_code_basic(airport_lookup):
    assert airport_lookup.get_airport_from_code("LAX").ident == "KLAX"
    assert airport_lookup.get_airport_from_code("BOS").ident == "KBOS"
