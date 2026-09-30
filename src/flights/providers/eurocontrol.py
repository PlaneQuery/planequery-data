from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.eurocontrol.preprocess import prepare_eurocontrol_month_parquet
from data_engineering.openairframes.read import get_openairframes_snapshot
from flights.airport_reference import resolve_flight_airport_codes
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


EUROCONTROL_AIRPORT_CODE_PRIORITY = (
    "icao_code",
    "ident",
    "gps_code",
    "iata_code",
    "local_code",
)


def create_eurocontrol_flights(
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    df = (
        pl.scan_parquet(prepare_eurocontrol_month_parquet(target_date))
        .filter(pl.col("takeoff_time").dt.date() == target_date)
        .collect()
    )
    df_openairframes = (
        get_openairframes_snapshot(at_date=target_date)
        .select("registration", "icao")
        .with_columns(
            pl.col("registration")
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace_all("-", "")
            .alias("registration_norm")
        )
        .with_columns(pl.len().over("registration_norm").alias("_match_count"))
        .filter(pl.col("_match_count") == 1)
        .drop("_match_count")
    )
    df = df.join(
        df_openairframes,
        left_on="registration",
        right_on="registration_norm",
        how="inner",
        validate="m:1",
    )

    df = resolve_flight_airport_codes(
        df,
        airport_catalog,
        takeoff_code_column="takeoff_airport_code",
        landing_code_column="landing_airport_code",
        code_priority=EUROCONTROL_AIRPORT_CODE_PRIORITY,
        allow_unresolved=True,
        prioritize_faa_airspace=False,
    )
    df = df.select(
        "icao",
        "registration",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
        "aircraft_type",
        "owner",
    )
    return with_flight_schema_columns(df)


@dataclass(frozen=True)
class EurocontrolFlightSource(NonADSBFlightSource):
    name: str = "eurocontrol"
    version: int = 4

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_eurocontrol_flights(dt, airport_catalog)
