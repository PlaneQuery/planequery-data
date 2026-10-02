"""OpenSky flights_data4 source."""

from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.opensky.read_trino_flights import read_opensky_flights
from flights.airport_reference import (
    ICAO_AIRPORT_CODE_PRIORITY,
    resolve_flight_airport_codes,
)
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


OPENSKY_AIRPORT_CODE_PRIORITY = ICAO_AIRPORT_CODE_PRIORITY


def create_opensky_flights(
    dt: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    flights = read_opensky_flights(dt)
    flights = resolve_flight_airport_codes(
        flights,
        airport_catalog,
        takeoff_code_column="takeoff_airport_ident",
        landing_code_column="landing_airport_ident",
        code_priority=OPENSKY_AIRPORT_CODE_PRIORITY,
        allow_unresolved=True,
    )
    return with_flight_schema_columns(flights)


@dataclass(frozen=True)
class OpenSkyTrinoFlightSource(NonADSBFlightSource):
    """Flights read directly from a downloaded OpenSky Trino partition."""

    name: str = "opensky"
    version: int = 3

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_opensky_flights(dt, airport_catalog)
