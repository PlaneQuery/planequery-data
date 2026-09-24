from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.airport_lookup import AirportLookup
from data_engineering.eurocontrol.preprocess import prepare_eurocontrol_month_parquet
from data_engineering.openairframes.read import get_openairframes_snapshot
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


def create_eurocontrol_flights(target_date: date) -> pl.DataFrame:
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

    airport_lookup = AirportLookup()

    def get_airport_code(airport_id: str):
        airport = airport_lookup.get_airport_from_code(
            airport_id,
            prioritize_faa_airspace=False,
        )
        if airport is not None:
            return airport.ident
        return None

    df = df.with_columns(
        pl.col("takeoff_airport_code")
        .map_elements(get_airport_code, return_dtype=pl.String)
        .alias("takeoff_airport_ident"),
        pl.col("landing_airport_code")
        .map_elements(get_airport_code, return_dtype=pl.String)
        .alias("landing_airport_ident"),
    )
    df = df.filter(
        pl.col("takeoff_airport_ident").is_not_null()
        & pl.col("landing_airport_ident").is_not_null()
    )
    df = df.select(
        "icao",
        "registration",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
        "aircraft_type",
        "owner",
    )
    return with_flight_schema_columns(df)


@dataclass(frozen=True)
class EurocontrolFlightSource(NonADSBFlightSource):
    name: str = "eurocontrol"
    version: int = 1

    def create_flights_for_day(self, dt: date) -> pl.DataFrame:
        return create_eurocontrol_flights(dt)
