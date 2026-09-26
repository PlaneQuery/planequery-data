from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.american_evals import continental_usa_to_continental_usa
from flights.flight_source import (
    ADSBFlightSource,
    get_flights,
    get_flights_for_catalog,
    write_flights,
)
from flights.flights_comparison import df_flights_comparison
from flights.flights_match_adsb import (
    ADSB_MATCHING_COLUMNS,
    get_matching_icaos_in_flights,
)
from flights.providers.bts import BTSFlightSource
from flights.providers.sfdps import SFDPSFlightSource


def _matching_reference_flights(
    dt: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    df_bts = get_flights_for_catalog(BTSFlightSource(), dt, airport_catalog)
    df_sfdps = get_flights_for_catalog(SFDPSFlightSource(), dt, airport_catalog)
    shared_icaos = set(df_bts["icao"].unique()) & set(df_sfdps["icao"].unique())

    df_bts = df_bts.filter(
        pl.col("icao").is_in(shared_icaos)
        & (pl.col("landing_time").dt.date() == dt)
    )
    df_sfdps = df_sfdps.filter(
        pl.col("icao").is_in(shared_icaos)
        & (pl.col("landing_time").dt.date() == dt)
    )
    df_bts = continental_usa_to_continental_usa(df_bts, airport_catalog)
    df_sfdps = continental_usa_to_continental_usa(df_sfdps, airport_catalog)

    start = datetime.combine(dt, datetime.min.time())
    comparison = df_flights_comparison(
        df_bts,
        df_sfdps,
        start,
        start + timedelta(days=1),
        airport_catalog,
    )
    matching = comparison.filter(
        (pl.col("match_status") == "both").all().over("icao")
    )
    return matching.with_columns(
        pl.col("category_df1").alias("category")
    ).select(df_bts.columns)


@dataclass(frozen=True)
class GroundTruthFlightSource(ADSBFlightSource):
    name: str = "ground-truth"
    version: int = 2

    def adsb_columns(self):
        return ADSB_MATCHING_COLUMNS

    def adsb_icaos(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ):
        return _matching_reference_flights(
            dt,
            airport_catalog,
        )["icao"].unique().to_list()

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        start = datetime.combine(dt, datetime.min.time())
        matching = _matching_reference_flights(dt, airport_catalog)
        return get_matching_icaos_in_flights(matching, df_adsb, start)


def get_ground_truth_flights_for_day(
    target_date: date,
    adsb_src: ADSBSource = ADSBSource.ADSBLOL,
    use_cache: bool = True,
) -> pl.DataFrame:
    return get_flights(
        GroundTruthFlightSource(),
        target_date,
        ADSBSource(adsb_src),
        use_cache=use_cache,
    )


if __name__ == "__main__":
    day = date(2026, 3, 1)
    print(write_flights(GroundTruthFlightSource(), day, ADSBSource.ADSBLOL))
