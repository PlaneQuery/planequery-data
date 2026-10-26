"""Traffic's default flight iterator in the standard source interface."""

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import ADSBFlightSource
from flights.opensky_reverse_eng_nearest import assign_nearest_airports
from flights.segmentation.algorithm import _adsb_read_bounds
from .segmentation import ADSB_COLUMNS, create_flights_from_adsb


@dataclass(frozen=True)
class TrafficFlightSource(ADSBFlightSource):
    name: str = "traffic"
    version: int = 1

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self) -> tuple[str, ...]:
        return ADSB_COLUMNS

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return _adsb_read_bounds(dt)

    def adsb_read_interval_for_source(
        self, dt: date, adsb_src: ADSBSource,
    ) -> tuple[datetime, datetime]:
        return _adsb_read_bounds(dt, adsb_src)

    def create_flights_for_day(
        self, dt: date, df_adsb: pl.DataFrame, airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return assign_nearest_airports(
            create_flights_from_adsb(df_adsb, dt), airport_catalog,
        )
