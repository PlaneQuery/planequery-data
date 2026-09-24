from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from flights.airport_reference import resolve_flight_airport_codes
from flights.flight_type import with_flight_schema_columns
from flights.flight_source import NonADSBFlightSource, file_artifact_cache_key


DEFAULT_INPUT = Path(
    "/Users/jonahgoode/Documents/PlaneQuery/"
    "0fb63f25-7663-45e5-a7b1-aa22c1ecc4d0.csv"
)
CSV_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.f UTC"


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
    target_date: date,
    airport_catalog: AirportCatalogView,
    input_csv: str | Path | None = None,
) -> pl.DataFrame:
    flights = read_mike_flights_csv(input_csv).filter(
        pl.col("takeoff_time").dt.date() == target_date
    )
    return resolve_flight_airport_codes(
        flights,
        airport_catalog,
        takeoff_code_column="takeoff_airport_ident",
        landing_code_column="landing_airport_ident",
        code_priority=("ident",),
    )


@dataclass(frozen=True)
class MikeFlightSource(NonADSBFlightSource):
    name: str = "mike"
    version: int = 2
    input_csv: str | Path | None = None

    def artifact_cache_key(self) -> str:
        return file_artifact_cache_key("input", self.input_csv or DEFAULT_INPUT)

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_mike_flights(dt, airport_catalog, self.input_csv)
