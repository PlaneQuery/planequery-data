from data_engineering.flights.flight_source import NonADSBFlightSource
from dataclasses import dataclass
from datetime import date
import polars as pl

@dataclass(frozen=True)
class BTSFlightSource(NonADSBFlightSource):
    name: str = "bts"
    version: int = 4
    uses_adsb_source: bool = False

    def create_flights_for_day(
        self,
        dt: date,
    ) -> pl.DataFrame:
        from data_engineering.bts.read_bts_ontime_to_flights import (
            create_flights_from_bts,
        )

        return create_flights_from_bts(dt)


