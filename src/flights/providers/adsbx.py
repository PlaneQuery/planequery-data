from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from flights.airport_reference import (
    ICAO_AIRPORT_CODE_PRIORITY,
    resolve_flight_airport_codes,
)
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


ADSBX_AIRPORT_CODE_PRIORITY = ICAO_AIRPORT_CODE_PRIORITY


def create_adsbx_flights(
    dt: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    flights = get_adsbx_flights_for_day(dt)
    flights = resolve_flight_airport_codes(
        flights,
        airport_catalog,
        takeoff_code_column="takeoff_airport_ident",
        landing_code_column="landing_airport_ident",
        code_priority=ADSBX_AIRPORT_CODE_PRIORITY,
        allow_unresolved=True,
    )
    return with_flight_schema_columns(flights)


@dataclass(frozen=True)
class ADSBXPrecomputedFlightSource(NonADSBFlightSource):
    """The daily flight product published by ADS-B Exchange."""

    name: str = "adsbx"
    version: int = 2

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_adsbx_flights(dt, airport_catalog)
