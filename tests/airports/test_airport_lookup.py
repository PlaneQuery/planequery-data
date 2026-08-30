import os
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
