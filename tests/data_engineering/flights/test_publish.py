from dataclasses import dataclass
from datetime import date

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
import data_engineering.spark as spark_module
from data_engineering.flights import publish
from flights.airport_catalog_manifest import (
    AirportCatalogManifest,
    airport_catalog_manifest_path,
)
from flights import flight_source
from flights.flight_type import FLIGHT_POLARS_SCHEMA


@dataclass(frozen=True)
class FakeAirportCatalog:
    provenance: AirportCatalogProvenance


class FakeSpark:
    def __init__(self, existing_rows: int = 0):
        self.existing_rows = existing_rows
        self.read = self
        self.selected_columns = None
        self.table = None
        self.stopped = False

    def sql(self, query):
        self.query = query
        return self

    def collect(self):
        return [{"row_count": self.existing_rows}]

    def parquet(self, path):
        self.parquet_path = path
        self.parquet_schema = pl.read_parquet(path).schema
        return self

    def select(self, *columns):
        self.selected_columns = columns
        return self

    def fillna(self, value, subset):
        self.fill_value = value
        self.fill_subset = subset
        return self

    def coalesce(self, partitions):
        self.partitions = partitions
        return self

    def writeTo(self, table):
        self.table = table
        return self

    def append(self):
        self.appended = True

    def stop(self):
        self.stopped = True


def test_publish_flights_parquet_uses_canonical_path(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    airport_catalog = FakeAirportCatalog(
        AirportCatalogProvenance(
            flight_date=date(2026, 3, 1),
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test",
        )
    )
    flights = pl.DataFrame({"icao": ["abc123"]})

    path = publish.publish_flights_parquet(
        flights,
        date(2026, 3, 1),
        publish.ADSBSource.ADSBLOL,
        airport_catalog,
    )

    assert path == (
        tmp_path
        / "data/flights/source=algorithm-nearest/adsb_source=adsblol/version=6"
        / "year=2026/month=03/day=01/data.parquet"
    )
    assert pl.read_parquet(path).schema == FLIGHT_POLARS_SCHEMA
    assert pl.read_parquet(path).select(
        "takeoff_airport_id", "landing_airport_id"
    ).row(0) == (None, None)
    assert AirportCatalogManifest.read(
        airport_catalog_manifest_path(path)
    ) == AirportCatalogManifest.from_provenance(airport_catalog.provenance)


def test_ingest_flights_to_iceberg(monkeypatch):
    spark = FakeSpark()
    monkeypatch.setattr(spark_module, "get_spark_session", lambda: spark)

    publish.ingest_flights_to_iceberg(
        pl.DataFrame({"icao": ["abc123"]}),
        date(2026, 3, 1),
        AirportCatalogManifest(
            flight_date=date(2026, 3, 1),
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test",
        ),
        prod=True,
    )

    assert spark.table == "glue.planequery_lake_prod.flights"
    assert spark.selected_columns == tuple(publish.ICEBERG_FLIGHTS_COLUMNS)
    assert spark.fill_subset == publish.BOOLEAN_COLUMNS
    assert spark.partitions == 2
    assert spark.appended
    assert spark.stopped
    assert spark.parquet_schema["airport_catalog_observed_commit_sha"] == pl.String
    assert spark.parquet_schema["airport_catalog_knowledge_cutoff_date"] == pl.Date
    assert spark.parquet_schema["airport_catalog_knowledge_commit_sha"] == pl.String
    assert spark.parquet_schema["airport_catalog_available_date"] == pl.Date
    assert spark.parquet_schema["airport_catalog_policy_version"] == pl.String


def test_ingest_flights_to_iceberg_rejects_existing_day(monkeypatch):
    spark = FakeSpark(existing_rows=1)
    monkeypatch.setattr(spark_module, "get_spark_session", lambda: spark)

    with pytest.raises(RuntimeError, match="already contains 1 rows"):
        publish.ingest_flights_to_iceberg(
            pl.DataFrame({"icao": ["abc123"]}),
            date(2026, 3, 1),
            AirportCatalogManifest(
                flight_date=date(2026, 3, 1),
                observed_commit_sha="observed-commit",
                knowledge_cutoff_date=date(2026, 4, 1),
                knowledge_commit_sha="knowledge-commit",
                catalog_available_date=date(2026, 4, 1),
                policy_version="test",
            ),
            prod=False,
        )

    assert spark.stopped
