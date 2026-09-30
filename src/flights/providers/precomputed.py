from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


@dataclass(frozen=True)
class ADSBXPrecomputedFlightSource(NonADSBFlightSource):
    """The daily flight product published by ADS-B Exchange."""

    name: str = "adsbx"
    version: int = 1

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        del airport_catalog
        return with_flight_schema_columns(get_adsbx_flights_for_day(dt))
