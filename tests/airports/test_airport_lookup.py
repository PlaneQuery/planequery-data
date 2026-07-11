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
