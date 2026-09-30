"""OpenSky flights_data4 source."""

from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.opensky.read_trino_flights import read_opensky_flights
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


@dataclass(frozen=True)
class OpenSkyTrinoFlightSource(NonADSBFlightSource):
    """Flights read directly from a downloaded OpenSky Trino partition."""

    name: str = "opensky"
    version: int = 2

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        del airport_catalog
        return with_flight_schema_columns(read_opensky_flights(dt))
