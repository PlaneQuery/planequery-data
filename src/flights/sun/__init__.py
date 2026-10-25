"""Sun et al.'s ADS-B flight extraction with endpoint-nearest airports."""

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import ADSBFlightSource
from flights.segmentation import algorithm

from . import segmentation


@dataclass(frozen=True)
class SunFlightSource(ADSBFlightSource):
    name: str = "sun"
    version: int = 2

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self) -> tuple[str, ...]:
        return segmentation.ADSB_COLUMNS

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return algorithm._adsb_read_bounds(dt)

    def adsb_read_interval_for_source(
        self, dt: date, adsb_src: ADSBSource
    ) -> tuple[datetime, datetime]:
        return algorithm._adsb_read_bounds(dt, adsb_src)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return segmentation.create_flights_from_adsb(df_adsb, dt, airport_catalog)
