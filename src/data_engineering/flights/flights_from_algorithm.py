"""Run the canonical flights algorithm and publish its output.

This module keeps planequery-data's S3 and Spark/Iceberg integration around
the implementation in ``flights_algorithm.main``.  The historical neural
flight model pipeline has been removed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

import flights_algorithm.main as flights_algorithm
from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.flights.flight_type import (
    ALGORITHM_ADSB_SOURCES,
    flights_algorithm_output_path,
    with_flight_schema_columns,
)
from data_engineering.utils import (
    upload_to_s3,
)


ICEBERG_FLIGHTS_COLUMNS = [
    "icao",
    "callsign",
    "registration",
    "takeoff_time",
    "takeoff_airport_ident",
    "landing_time",
    "landing_airport_ident",
    "pia",
    "ladd",
    "military",
    "interesting",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
]
BOOLEAN_COLUMNS = ["pia", "ladd", "military", "interesting"]


def _date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start_date: date, end_date: date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


def publish_flights_parquet(
    df_flights: pl.DataFrame,
    target_date: date,
    source: ADSBSource,
    no_airports_model: bool,
) -> Path:
    output_path = flights_algorithm_output_path(
        target_date,
        adsb_src=source,
        no_airports_model=no_airports_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_flight_schema_columns(df_flights).write_parquet(output_path)
    return output_path


def ingest_flights_to_iceberg(
    df_flights: pl.DataFrame,
    target_date: date,
    *,
    prod: bool,
) -> None:
    from data_engineering.spark import get_spark_session

    database = "planequery_lake_prod" if prod else "planequery_lake_dev"
    table = f"glue.{database}.flights"
    next_day = target_date + timedelta(days=1)
    spark = get_spark_session()
    try:
        existing_rows = spark.sql(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {table}
            WHERE takeoff_time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
              AND takeoff_time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
            """
        ).collect()[0]["row_count"]
        if existing_rows:
            raise RuntimeError(
                f"{table} already contains {existing_rows} rows for {target_date}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "flights.parquet"
            with_flight_schema_columns(df_flights).write_parquet(parquet_path)
            spark_df = (
                spark.read.parquet(str(parquet_path))
                .select(*ICEBERG_FLIGHTS_COLUMNS)
                .fillna(False, subset=BOOLEAN_COLUMNS)
            )
            spark_df.coalesce(2).writeTo(table).append()
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the flights algorithm and publish results to local parquet, S3, and/or Iceberg."
    )
    parser.add_argument("start_date", type=_date_arg)
    parser.add_argument("end_date", type=_date_arg)
    parser.add_argument(
        "--source",
        type=ADSBSource,
        choices=ALGORITHM_ADSB_SOURCES,
        default=ADSBSource.ADSBLOL,
    )
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no-airports-model", action="store_true")
    parser.add_argument("--write-to-s3", action="store_true")
    parser.add_argument("--ingest-to-iceberg", action="store_true")
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    if args.end_date <= args.start_date:
        parser.error("end_date must be after start_date")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    array_index = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX")
    dates = list(_date_range(args.start_date, args.end_date))
    if array_index is not None:
        dates = [dates[int(array_index)]]

    for target_date in dates:
        df_flights = flights_algorithm.main(
            target_date=target_date,
            source=args.source,
            max_workers=args.max_workers,
            no_airports_model=args.no_airports_model,
        )
        output_path = publish_flights_parquet(
            df_flights,
            target_date,
            args.source,
            args.no_airports_model,
        )
        print(f"Published {len(df_flights)} flights to {output_path}")
        if args.write_to_s3:
            upload_to_s3(output_path)
        if args.ingest_to_iceberg:
            ingest_flights_to_iceberg(df_flights, target_date, prod=args.prod)


if __name__ == "__main__":
    main()
