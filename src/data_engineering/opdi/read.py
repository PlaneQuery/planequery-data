from datetime import date
from pathlib import Path

import polars as pl

from flights.flight_type import (
    FLIGHT_POLARS_SCHEMA,
    with_flight_schema_columns,
)
from data_engineering.utils import OUTPUT_DIR

OPDI_FLIGHTS_VERSION = "v1"
OPDI_RAW_COLUMNS = [
    "icao24",
    "flt_id",
    "adep",
    "ades",
    "registration",
    "model",
    "typecode",
    "icao_aircraft_class",
    "icao_operator",
    "first_seen",
    "last_seen",
]


def opdi_raw_flights_parquet_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "opdi"
        / f"flight_list_{target_date:%Y%m}.parquet"
    )


def opdi_flights_parquet_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "opdi"
        / OPDI_FLIGHTS_VERSION
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / "data.parquet"
    )


def _datetime_expr(df: pl.DataFrame, column_name: str, alias: str) -> pl.Expr:
    if df.schema[column_name] == pl.String:
        return (
            pl.col(column_name)
            .str.to_datetime(strict=False)
            .dt.cast_time_unit("ms")
            .alias(alias)
        )
    return pl.col(column_name).cast(pl.Datetime("ms")).alias(alias)


def normalize_opdi_flights(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)

    df = (
        df.select(OPDI_RAW_COLUMNS)
        .rename(
            {
                "icao24": "icao",
                "flt_id": "callsign",
                "adep": "takeoff_airport_ident",
                "ades": "landing_airport_ident",
                "typecode": "aircraft_type",
                "model": "aircraft_description",
                "icao_aircraft_class": "category",
                "icao_operator": "owner",
            }
        )
        .with_columns(
            pl.col("icao").str.strip_chars().str.to_lowercase(),
            pl.col("callsign").str.strip_chars(),
            pl.col("registration").str.strip_chars(),
            pl.col("takeoff_airport_ident").str.strip_chars(),
            pl.col("landing_airport_ident").str.strip_chars(),
            _datetime_expr(df, "first_seen", "takeoff_time"),
            _datetime_expr(df, "last_seen", "landing_time"),
        )
        .with_columns(
            pl.col("takeoff_time").alias("first_message_time"),
            pl.col("landing_time").alias("last_message_time"),
        )
        .drop("first_seen", "last_seen")
    )
    return with_flight_schema_columns(df)


def create_opdi_flights_month(
    target_date: date,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    raw_path = Path(raw_path or opdi_raw_flights_parquet_path(target_date))
    output_path = Path(output_path or opdi_flights_parquet_path(target_date))
    df = normalize_opdi_flights(pl.read_parquet(raw_path)).filter(
        pl.col("icao").is_not_null()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df


def get_opdi_flights(
    target_date: date,
    use_cache: bool = True,
    raw_path: str | Path | None = None,
) -> pl.DataFrame:
    path = opdi_flights_parquet_path(target_date)
    if use_cache and path.exists():
        df = pl.read_parquet(path)
    else:
        df = create_opdi_flights_month(
            target_date,
            raw_path=raw_path,
            output_path=path,
        )
    return df.filter(pl.col("takeoff_time").dt.date() == target_date)
