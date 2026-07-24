from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from data_engineering.flights.flight_type import with_flight_schema_columns
from data_engineering.openairframes.read import (
    DEFAULT_OPENAIRFRAMES_ADSB_PATH,
    scan_latest_registration_icao_for_join,
)
from data_engineering.utils import OUTPUT_DIR

EUROCONTROL_FLIGHTS_VERSION = "v1"
REGISTRATION_JOIN_KEY = "registration_join_key"

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
        / "intermediate"
        / f"flights_{month_start:%Y%m%d}_{month_end:%Y%m%d}_time_columns.parquet"
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


def _registration_join_key_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "")
    )


def _registration_icao_for_join(
    registration_icao: pl.DataFrame | pl.LazyFrame | None = None,
    openairframes_path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.LazyFrame:
    if registration_icao is None:
        df_openairframes = scan_latest_registration_icao_for_join(
            path=openairframes_path,
            real_icao_column="icao",
        )
    elif isinstance(registration_icao, pl.DataFrame):
        df_openairframes = registration_icao.lazy()
    else:
        df_openairframes = registration_icao

    schema_names = df_openairframes.collect_schema().names()
    if REGISTRATION_JOIN_KEY not in schema_names:
        df_openairframes = df_openairframes.with_columns(
            _registration_join_key_expr("registration").alias(REGISTRATION_JOIN_KEY)
        )
        schema_names = df_openairframes.collect_schema().names()

    return (
        df_openairframes
        .filter(pl.col(REGISTRATION_JOIN_KEY) != "")
        .drop([col for col in ["registration"] if col in schema_names])
        .with_columns(pl.col("icao").str.strip_chars().str.to_lowercase())
        .unique(subset=[REGISTRATION_JOIN_KEY], keep="last", maintain_order=False)
    )


def _airport_code_to_ident(code: str | None, airport_lookup: Any | None = None) -> str | None:
    if code is None:
        return None

    code = code.strip().upper()
    if not code:
        return None

    if airport_lookup is None:
        from airports.airport_lookup import AirportLookup

        airport_lookup = AirportLookup()

    if len(code) == 3:
        return (
            airport_lookup.iata_to_ident_index.get(code)
            or airport_lookup.gps_code_to_ident_index.get(code)
            or airport_lookup.local_code_to_ident_index.get(code)
        )

    airport = airport_lookup.get_Airport_from_airport_ident(code)
    if airport is not None:
        return airport.ident

    return (
        airport_lookup.gps_code_to_ident_index.get(code)
        or airport_lookup.local_code_to_ident_index.get(code)
        or airport_lookup.iata_to_ident_index.get(code)
    )


def _datetime_expr(df: pl.DataFrame, column_name: str, alias: str) -> pl.Expr:
    if df.schema[column_name] == pl.String:
        return pl.col(column_name).str.strptime(pl.Datetime("ms"), strict=False).alias(alias)
    return pl.col(column_name).cast(pl.Datetime("ms")).alias(alias)


def normalize_eurocontrol_flights(
    df: pl.DataFrame,
    registration_icao: pl.DataFrame | pl.LazyFrame | None = None,
    airport_lookup: Any | None = None,
    openairframes_path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.DataFrame:
    if airport_lookup is None:
        from airports.airport_lookup import AirportLookup

        airport_lookup = AirportLookup()

    df_openairframes = _registration_icao_for_join(
        registration_icao=registration_icao,
        openairframes_path=openairframes_path,
    )
    df_flights = (
        df.select(EUROCONTROL_COLUMNS)
        .with_columns(
            _registration_join_key_expr("AC Registration").alias(REGISTRATION_JOIN_KEY),
            pl.col("AC Registration").str.strip_chars().str.to_uppercase().alias("registration"),
            pl.col("AC Type").alias("aircraft_type"),
            pl.col("AC Operator").alias("owner"),
            _datetime_expr(df, "ACTUAL OFF BLOCK TIME", "takeoff_time"),
            _datetime_expr(df, "ACTUAL ARRIVAL TIME", "landing_time"),
        )
        .lazy()
        .join(df_openairframes, on=REGISTRATION_JOIN_KEY, how="left")
        .drop(REGISTRATION_JOIN_KEY)
        .collect()
    )
    df_flights = df_flights.with_columns(
        pl.col("ADEP")
        .map_elements(
            lambda code: _airport_code_to_ident(code, airport_lookup=airport_lookup),
            return_dtype=pl.String,
        )
        .alias("takeoff_airport_ident"),
        pl.col("ADES")
        .map_elements(
            lambda code: _airport_code_to_ident(code, airport_lookup=airport_lookup),
            return_dtype=pl.String,
        )
        .alias("landing_airport_ident"),
    )
    df_flights = df_flights.drop(
        [
            col
            for col in ["AC Registration", "ADEP", "ADES", "AC Type", "AC Operator"]
            if col in df_flights.columns
        ]
    )
    return with_flight_schema_columns(df_flights)


def create_eurocontrol_flights(
    target_date: date,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    raw_path = raw_path or eurocontrol_raw_flights_parquet_path(target_date)
    output_path = Path(output_path or eurocontrol_flights_parquet_path(target_date))
    df = normalize_eurocontrol_flights(pl.read_parquet(raw_path)).filter(
        pl.col("icao").is_not_null()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df


def get_eurocontrol_flights_month(
    target_date: date,
    use_cache: bool = True,
    raw_path: str | Path | None = None,
) -> pl.DataFrame:
    path = eurocontrol_flights_parquet_path(target_date)
    if use_cache and path.exists():
        return pl.read_parquet(path)
    return create_eurocontrol_flights(target_date, raw_path=raw_path, output_path=path)


def read_eurocontrol_flights(
    target_date: date,
    raw_path: str | Path | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    df = get_eurocontrol_flights_month(
        target_date,
        use_cache=use_cache,
        raw_path=raw_path,
    )
    return df.filter(pl.col("takeoff_time").dt.date() == target_date)
