from datetime import date

import polars as pl
import pytest

import data_engineering.spark as spark_module
from data_engineering.flights import publish
from flights import flight_source
from flights.flight_type import FLIGHT_POLARS_SCHEMA


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
    flights = pl.DataFrame({"icao": ["abc123"]})

    path = publish.publish_flights_parquet(
        flights,
        date(2026, 3, 1),
        publish.ADSBSource.ADSBLOL,
    )

    assert path == (
        tmp_path
        / "data/flights/source=algorithm-nearest/adsb_source=adsblol/version=3"
        / "year=2026/month=03/day=01/data.parquet"
    )
    assert pl.read_parquet(path).schema == FLIGHT_POLARS_SCHEMA


def test_ingest_flights_to_iceberg(monkeypatch):
    spark = FakeSpark()
    monkeypatch.setattr(spark_module, "get_spark_session", lambda: spark)

    publish.ingest_flights_to_iceberg(
        pl.DataFrame({"icao": ["abc123"]}),
        date(2026, 3, 1),
        prod=True,
    )

    assert spark.table == "glue.planequery_lake_prod.flights"
    assert spark.selected_columns == tuple(publish.ICEBERG_FLIGHTS_COLUMNS)
    assert spark.fill_subset == publish.BOOLEAN_COLUMNS
    assert spark.partitions == 2
    assert spark.appended
    assert spark.stopped


def test_ingest_flights_to_iceberg_rejects_existing_day(monkeypatch):
    spark = FakeSpark(existing_rows=1)
    monkeypatch.setattr(spark_module, "get_spark_session", lambda: spark)

    with pytest.raises(RuntimeError, match="already contains 1 rows"):
        publish.ingest_flights_to_iceberg(
            pl.DataFrame({"icao": ["abc123"]}),
            date(2026, 3, 1),
            prod=False,
        )

    assert spark.stopped
