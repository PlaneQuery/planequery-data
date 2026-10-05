"""Run the flights algorithm and publish its output to external stores."""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.utils import upload_to_s3
from flights import algorithm
from flights.flight_manifest import (
    FlightManifest,
    flight_manifest_path,
)
from flights.flight_source import flight_path
from flights.flight_type import with_flight_schema_columns
from planequery_data.utils import date_range


ICEBERG_FLIGHTS_COLUMNS = [
    "icao",
    "callsign",
    "registration",
    "takeoff_time",
    "takeoff_airport_id",
    "takeoff_airport_ident",
    "landing_time",
    "landing_airport_id",
    "landing_airport_ident",
    "pia",
    "ladd",
    "military",
    "interesting",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
    "airport_catalog_observed_commit_sha",
    "airport_catalog_knowledge_cutoff_date",
    "airport_catalog_knowledge_commit_sha",
    "airport_catalog_available_date",
    "airport_catalog_policy_version",
]
BOOLEAN_COLUMNS = ["pia", "ladd", "military", "interesting"]


def _date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def publish_flights_parquet(
    df_flights: pl.DataFrame,
    target_date: date,
    source: ADSBSource,
    airport_catalog: AirportCatalogView,
) -> Path:
    output_path = flight_path(
        algorithm.AlgorithmNearestFlightSource(),
        target_date,
        source,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_flight_schema_columns(df_flights).write_parquet(output_path)
    FlightManifest.from_provenance(
        airport_catalog.provenance,
        adsb_source=source,
    ).write(flight_manifest_path(output_path))
    return output_path


def ingest_flights_to_iceberg(
    df_flights: pl.DataFrame,
    target_date: date,
    flight_manifest: FlightManifest,
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
            with_flight_schema_columns(df_flights).with_columns(
                pl.lit(
                    flight_manifest.observed_commit_sha
                ).alias("airport_catalog_observed_commit_sha"),
                pl.lit(
                    flight_manifest.knowledge_cutoff_date
                ).alias("airport_catalog_knowledge_cutoff_date"),
                pl.lit(
                    flight_manifest.knowledge_commit_sha
                ).alias("airport_catalog_knowledge_commit_sha"),
                pl.lit(
                    flight_manifest.catalog_available_date
                ).alias("airport_catalog_available_date"),
                pl.lit(
                    flight_manifest.policy_version
                ).alias("airport_catalog_policy_version"),
            ).write_parquet(parquet_path)
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
        choices=tuple(ADSBSource),
        default=ADSBSource.ADSBLOL,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=algorithm.default_max_workers(),
    )
    parser.add_argument("--write-to-s3", action="store_true")
    parser.add_argument("--ingest-to-iceberg", action="store_true")
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    if args.end_date <= args.start_date:
        parser.error("end_date must be after start_date")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    array_index = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX")
    dates = list(date_range(args.start_date, args.end_date))
    if array_index is not None:
        dates = [dates[int(array_index)]]

    for target_date in dates:
        airport_catalog = airport_catalog_for_date(target_date)
        df_flights = algorithm.create_flights_for_day(
            target_date=target_date,
            airport_catalog=airport_catalog,
            adsb_src=args.source,
            max_workers=args.max_workers,
        )
        output_path = publish_flights_parquet(
            df_flights,
            target_date,
            args.source,
            airport_catalog,
        )
        print(f"Published {len(df_flights)} flights to {output_path}")
        if args.write_to_s3:
            upload_to_s3(output_path)
            upload_to_s3(flight_manifest_path(output_path))
        if args.ingest_to_iceberg:
            ingest_flights_to_iceberg(
                df_flights,
                target_date,
                FlightManifest.read(
                    flight_manifest_path(output_path)
                ),
                prod=args.prod,
            )


if __name__ == "__main__":
    main()
