
import polars as pl
from functools import cached_property
from datetime import datetime
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights.flights_match_adsb import flights_match_adsb


class FlightsTestPackage:
    def __init__(
        self,
        flights: list[Flight],
        start_dt: datetime,
        end_dt: datetime,
        adsb_source: str = "adsblol",
    ) -> None:
        icao = flights[0].icao
        self.flights = flights
        self._lazy_adsb_rows: pl.LazyFrame = read_adsb(
            start_dt,
            end_dt,
            source=adsb_source,
            icaos=[icao],
            lazy=True,
        )

    @cached_property
    def adsb_rows(self) -> pl.DataFrame:
        return self._lazy_adsb_rows.collect()


def flights_match_real_adsb_with_dropout(test_package: FlightsTestPackage):
    flights = test_package.flights
    df_flights = flights_to_flights_df(flights)
    df_adsb = test_package.adsb_rows
    dt = datetime.combine(flights[0].takeoff_time.date(), datetime.min.time())
    assert flights_match_adsb(df_flights, df_adsb, dt) is True

    for i in range(len(flights)):
        new_flights = flights[:i] + flights[i + 1 :]
        df_flights_with_dropout = flights_to_flights_df(new_flights)
        assert not flights_match_adsb(df_flights_with_dropout, df_adsb, dt)
