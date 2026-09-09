import calendar
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.openairframes.read import get_openairframes_snapshot
from data_engineering.utils import OUTPUT_DIR

EUROCONTROL_FLIGHTS_VERSION = "v1"

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

def eurocontrol_flights_parquet_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "eurocontrol"
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / "data.parquet"
    )

def datetime_expr(column_name: str, alias: str) -> pl.Expr:
    return pl.col(column_name).str.strptime(pl.Datetime("ms"), strict=False).alias(alias)


def get_eurocontrol_flights(target_date: date) -> pl.DataFrame:
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
    df_openairframes = get_openairframes_snapshot(at_date=target_date).select(
        "registration",
        "icao",
    )
    df_openairframes = df_openairframes.with_columns(pl.col("registration").str.replace("-", "").alias("registration_norm"))
    df = (
        df.with_columns(pl.col("AC Registration").alias("registration"))
        .join(
            df_openairframes,
            left_on="registration",
            right_on="registration_norm",
            how="inner",
            # validate="m:1",
        )
        .select(
            "icao",
            "registration",
            "takeoff_time",
            "landing_time",
            "ADEP",
            "ADES",
            "aircraft_type",
            "owner",
        )
    )

    return df
