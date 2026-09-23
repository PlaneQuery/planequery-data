import calendar
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from airports.airport_lookup import AirportLookup
from data_engineering.openairframes.read import get_openairframes_snapshot
from data_engineering.utils import OUTPUT_DIR
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns

EUROCONTROL_COLUMNS = [
    "AC Registration",
    "ADEP",
    "ADES",
    "AC Type",
    "AC Operator",
    "ACTUAL OFF BLOCK TIME",
    "ACTUAL ARRIVAL TIME",
]


def eurocontrol_raw_flights_parquet_path(target_date: date) -> Path:
    _, last_day = calendar.monthrange(target_date.year, target_date.month)
    month_start = date(target_date.year, target_date.month, 1)
    month_end = date(target_date.year, target_date.month, last_day)
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "eurocontrol"
        / f"{target_date:%Y%m}"
        / "extract"
        / f"Flights_{month_start:%Y%m%d}_{month_end:%Y%m%d}.csv"
    )

def datetime_expr(column_name: str, alias: str) -> pl.Expr:
    return pl.col(column_name).str.strptime(pl.Datetime("ms"), strict=False).alias(alias)


def create_eurocontrol_flights(target_date: date) -> pl.DataFrame:
    df = pl.read_csv(eurocontrol_raw_flights_parquet_path(target_date))
    df = (
        df.select(EUROCONTROL_COLUMNS)
        .with_columns(
            pl.col("AC Type").alias("aircraft_type"),
            pl.col("AC Operator").alias("owner"),
            datetime_expr("ACTUAL OFF BLOCK TIME", "takeoff_time"),
            datetime_expr("ACTUAL ARRIVAL TIME", "landing_time"),
        )
    )
    df = df.filter(pl.col("takeoff_time").dt.date() == target_date)
    df_openairframes = (
    get_openairframes_snapshot(at_date=target_date)
    .select("registration", "icao")
    .with_columns(
        pl.col("registration")
        .str.replace("-", "")
        .alias("registration_norm")
    )
    .with_columns(
        pl.len().over("registration_norm").alias("_match_count")
    )
    .filter(pl.col("_match_count") == 1)
    .drop("_match_count")
    )
    df = (
        df.with_columns(pl.col("AC Registration").alias("registration"))
        .join(
            df_openairframes,
            left_on="registration",
            right_on="registration_norm",
            how="inner",
            validate="m:1",
        )
    )

    # TODO: deal with "ADEP" and "ADES"... airport identifeis
    airport_lookup = AirportLookup()
    def get_airport_code(airport_id: str):
        airport = airport_lookup.get_airport_from_code(airport_id, prioritize_faa_airspace=False)
        if airport is not None:
            return airport.ident
        return None
    df = df.with_columns(
        pl.col("ADEP").map_elements(get_airport_code, return_dtype=pl.String).alias("takeoff_airport_ident"),
        pl.col("ADES").map_elements(get_airport_code, return_dtype=pl.String).alias("landing_airport_ident"),
    )
    df = df.filter(pl.col("takeoff_airport_ident").is_not_null() & pl.col("landing_airport_ident").is_not_null())
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
