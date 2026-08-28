from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.flights.flight_type import (
    FLIGHT_POLARS_SCHEMA,
    with_flight_schema_columns,
)
from data_engineering.utils import OUTPUT_DIR


DEFAULT_INPUT = Path(
    "/Users/jonahgoode/Documents/PlaneQuery/"
    "0fb63f25-7663-45e5-a7b1-aa22c1ecc4d0.csv"
)
MIKE_FLIGHTS_VERSION = "v1"
CSV_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.f UTC"


def mike_flights_parquet_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "mike"
        / MIKE_FLIGHTS_VERSION
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "flights.parquet"
    )


def _datetime_expr(df: pl.DataFrame, column_name: str) -> pl.Expr:
    if df.schema[column_name] == pl.String:
        return (
            pl.col(column_name)
            .str.to_datetime(
                CSV_TIMESTAMP_FORMAT,
                strict=False,
            )
            .dt.cast_time_unit("ms")
            .alias(column_name)
        )
    return pl.col(column_name).cast(pl.Datetime("ms")).alias(column_name)


def normalize_mike_flights(df: pl.DataFrame) -> pl.DataFrame:
    """Map Mike's CSV columns into the standard flight schema."""

    aliases = {
        "takeoff_ts": "takeoff_time",
        "landing_ts": "landing_time",
        "owner_operator": "owner",
    }
    renames = {
        source: destination
        for source, destination in aliases.items()
        if source in df.columns and destination not in df.columns
    }
    df = df.rename(renames)

    required_columns = {
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Mike flights CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df.with_columns(
        pl.col("icao").cast(pl.String).str.strip_chars().str.to_lowercase(),
        _datetime_expr(df, "takeoff_time"),
        _datetime_expr(df, "landing_time"),
    )
    return with_flight_schema_columns(df)


def read_mike_flights_csv(input_csv: str | Path | None = None) -> pl.DataFrame:
    input_path = Path(input_csv or DEFAULT_INPUT)
    if not input_path.exists():
        raise FileNotFoundError(f"Mike flights CSV does not exist: {input_path}")
    return normalize_mike_flights(pl.read_csv(input_path))


def create_mike_flights(
    input_csv: str | Path | None = None,
) -> pl.DataFrame:
    """Convert the input CSV and write one parquet partition per takeoff day."""

    df = read_mike_flights_csv(input_csv)
    df_with_takeoff = df.filter(pl.col("takeoff_time").is_not_null())
    skipped_rows = df.height - df_with_takeoff.height
    if skipped_rows:
        print(f"Skipping {skipped_rows} row(s) with no parseable takeoff_time")

    takeoff_dates = (
        df_with_takeoff
        .select(pl.col("takeoff_time").dt.date().unique().sort())
        .get_column("takeoff_time")
        .to_list()
    )
    for target_date in takeoff_dates:
        day_flights = df_with_takeoff.filter(
            pl.col("takeoff_time").dt.date() == target_date
        )
        output_path = mike_flights_parquet_path(target_date)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        day_flights.write_parquet(output_path)
        print(f"Wrote {day_flights.height} Mike flights to {output_path}")

    return df


def get_mike_flights_for_day(
    target_date: date,
    use_cache: bool = True,
    input_csv: str | Path | None = None,
) -> pl.DataFrame:
    """Read a daily Mike partition, creating it from the CSV when absent."""

    output_path = mike_flights_parquet_path(target_date)
    if use_cache and output_path.exists():
        return with_flight_schema_columns(pl.read_parquet(output_path))

    df = read_mike_flights_csv(input_csv).filter(
        pl.col("takeoff_time").dt.date() == target_date
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df
