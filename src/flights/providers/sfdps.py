import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import scan_adsb
from data_engineering.swim.read_sfdps import read_sfdps, sort_sfdps_messages
from data_engineering.swim.utils import MISSING_SFDPS_DAYS
from flights.airport_reference import (
    ICAO_AIRPORT_CODE_PRIORITY,
    resolve_flight_airport_codes,
)
from flights.flight_source import NonADSBFlightSource, get_flights
from flights.flight_type import with_flight_schema_columns


ADSB_ICAO_INFO_COLUMNS = [
    "registration",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
    "pia",
    "ladd",
    "military",
    "interesting",
]
SFDPS_PREFERRED_AIRCRAFT_COLUMNS = ("registration", "aircraft_type")

SFDPS_AIRPORT_CODE_PRIORITY = ICAO_AIRPORT_CODE_PRIORITY
# Allow small reporting-clock differences, but not future actual arrivals.
ACTUAL_LANDING_CLOCK_SKEW = timedelta(minutes=2)

GUFI_FLIGHT_SCHEMA = {
    "gufi": pl.String,
    "timestamp": pl.Datetime("us"),
    "callsign": pl.String,
    "aircraft_address": pl.String,
    "registration": pl.String,
    "aircraft_type": pl.String,
    "flightStatus": pl.String,
    "takeoff_time": pl.Datetime("us"),
    "takeoff_airport_icao": pl.String,
    "landing_time": pl.Datetime("us"),
    "landing_airport_icao": pl.String,
    "estimated_landing_time": pl.Datetime("us"),
    "estimated_takeoff_time": pl.Datetime("us"),
}


def _missing_sfdps_days() -> set[date]:
    return set(MISSING_SFDPS_DAYS)


def resolve_sfdps_airports(
    df_flights: pl.DataFrame,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Resolve SFDPS point codes, mapping unresolved codes to null."""
    return resolve_flight_airport_codes(
        df_flights,
        airport_catalog,
        takeoff_code_column="takeoff_airport_ident",
        landing_code_column="landing_airport_ident",
        code_priority=SFDPS_AIRPORT_CODE_PRIORITY,
        allow_unresolved=True,
    )

def _get_most_common_non_null_value(df: pl.DataFrame, column: str):
    """Return the most common non-null value, preferring the latest value on ties."""
    non_null_values = df[column].drop_nulls()
    if non_null_values.is_empty():
        return None
    value_counts = non_null_values.value_counts()
    max_count = value_counts["count"].max()
    most_common_values = set(
        value_counts.filter(pl.col("count") == max_count)[column].to_list()
    )
    return next(
        value for value in reversed(non_null_values.to_list())
        if value in most_common_values
    )

def gufi_state_machine(df: pl.DataFrame) -> pl.DataFrame:
    landing_time = None
    takeoff_time = None
    landing_airport = None
    takeoff_airport = None
    estimated_landing_time = None
    estimated_takeoff_time = None
    flight_status = None
    aircraft_address = _get_most_common_non_null_value(df, "aircraft_address")
    aircraft_registration = _get_most_common_non_null_value(df, "aircraft_registration")
    aircraft_type = _get_most_common_non_null_value(df, "aircraft_type")
    aircraft_identification = _get_most_common_non_null_value(df, "aircraft_identification")

    for _, row in enumerate(df.iter_rows(named=True)):
        if row["takeoff_time"] is not None:
            takeoff_time = row["takeoff_time"]
        if row["takeoff_airport_icao"] is not None:
            takeoff_airport = row["takeoff_airport_icao"]
        if (
            row["landing_time"] is not None
            and row["timestamp"] is not None
            and row["landing_time"] <= row["timestamp"] + ACTUAL_LANDING_CLOCK_SKEW
        ):
            landing_time = row["landing_time"]
        if row["landing_airport_icao"] is not None:
            landing_airport = row["landing_airport_icao"]
        if row["estimated_landing_time"] is not None:
            estimated_landing_time = row["estimated_landing_time"]
        if row["estimated_takeoff_time"] is not None:
            estimated_takeoff_time = row["estimated_takeoff_time"]

        if row["flightStatus"] == "CANCELLED":
            # never left ground
            pass

        if row["flightStatus"] == "ACTIVE":
            pass
        if row["flightStatus"] == "DROPPED":
            pass
        flight_status = row["flightStatus"]
        if takeoff_time is not None and landing_time is not None:
            return pl.DataFrame(
                {
                    "gufi": [row["gufi"]],
                    "timestamp": [row["timestamp"]],
                    "callsign": [aircraft_identification],
                    "aircraft_address": [aircraft_address],
                    "registration": [aircraft_registration],
                    "aircraft_type": [aircraft_type],
                    "flightStatus": [flight_status],
                    "takeoff_time": [takeoff_time],
                    "takeoff_airport_icao": [takeoff_airport],
                    "landing_time": [landing_time],
                    "landing_airport_icao": [landing_airport],
                    "estimated_landing_time": [estimated_landing_time],
                    "estimated_takeoff_time": [estimated_takeoff_time],
                },
                schema=GUFI_FLIGHT_SCHEMA,
            )
    if flight_status == "DROPPED":
        pass

    return pl.DataFrame(schema=GUFI_FLIGHT_SCHEMA)


def sfdps_to_flights(current_day: datetime) -> pl.DataFrame:
    # Arrival messages can arrive after midnight. Combine each GUFI's history
    # before reconstructing flights so departure state survives the boundary.
    df = pl.concat([
        read_sfdps(current_day),
        read_sfdps(current_day + timedelta(days=1)),
    ]).pipe(sort_sfdps_messages)
    df = df.group_by("gufi").map_groups(gufi_state_machine)
    df = (
        df.rename(
            {
                "takeoff_airport_icao": "takeoff_airport_ident",
                "landing_airport_icao": "landing_airport_ident",
            }
        )
        .with_columns(
            pl.col("aircraft_address").str.strip_chars().str.to_lowercase().alias("icao"),
            pl.lit(False).alias("pia"),
            pl.lit(False).alias("ladd"),
            pl.lit(False).alias("military"),
            pl.lit(False).alias("interesting"),
            pl.lit("").alias("owner"),
            pl.lit("").alias("aircraft_description"),
            pl.lit("").alias("category"),
        )
    )
    return with_flight_schema_columns(df)


def filter_sfdps_flights(df: pl.DataFrame, current_date: datetime) -> pl.DataFrame:
    df = df.filter(pl.col("icao").is_not_null() & (pl.col("icao") != ""))
    df = df.filter(pl.col("takeoff_time").dt.date() == current_date.date())
    return df


def add_adsb_icao_info(df_flights: pl.DataFrame, current_day: datetime) -> pl.DataFrame:
    if df_flights.is_empty():
        return df_flights
    icaos = df_flights["icao"].drop_nulls().unique().to_list()
    df_icao_info = (
        scan_adsb(
            current_day.date(),
            columns=["icao", "time", *ADSB_ICAO_INFO_COLUMNS],
            icaos=icaos,
        )
        .sort(["icao", "time"])
        .group_by("icao")
        .agg(
            *[
                pl.col(column).last()
                for column in ADSB_ICAO_INFO_COLUMNS
                if column != "category"
            ],
            # Trace snapshots report category intermittently; a trailing blank
            # must not erase the aircraft's last known category.
            pl.col("category")
            .filter(pl.col("category").str.strip_chars().fill_null("") != "")
            .last(),
        )
        .select(["icao", *ADSB_ICAO_INFO_COLUMNS])
        .collect(engine="streaming")
    )
    sfdps_columns = {
        column: f"_sfdps_{column}"
        for column in SFDPS_PREFERRED_AIRCRAFT_COLUMNS
    }
    df_flights = (
        df_flights.rename(sfdps_columns)
        .drop(
            [
                column
                for column in ADSB_ICAO_INFO_COLUMNS
                if column not in SFDPS_PREFERRED_AIRCRAFT_COLUMNS
            ]
        )
        .join(df_icao_info, on="icao", how="left")
        .with_columns(
            *[
                pl.when(
                    pl.col(sfdps_column).str.strip_chars().fill_null("") != ""
                )
                .then(pl.col(sfdps_column))
                .otherwise(pl.col(column))
                .fill_null("")
                .alias(column)
                for column, sfdps_column in sfdps_columns.items()
            ],
            pl.col("owner").fill_null(""),
            pl.col("aircraft_description").fill_null(""),
            pl.col("category").fill_null(""),
            pl.col("pia").fill_null(False),
            pl.col("ladd").fill_null(False),
            pl.col("military").fill_null(False),
            pl.col("interesting").fill_null(False),
        )
        .drop(*sfdps_columns.values())
    )
    return with_flight_schema_columns(df_flights)


def create_sfdps_flights_day(
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    assert target_date not in _missing_sfdps_days()
    assert target_date + timedelta(days=1) not in _missing_sfdps_days(), (
        "SFDPS flight reconstruction requires the following day's logs"
    )
    assert target_date >= date(2026, 1, 1)
    day_dt = datetime(target_date.year, target_date.month, target_date.day)
    df_flights = sfdps_to_flights(day_dt)
    df_flights = filter_sfdps_flights(df_flights, day_dt)
    df_flights = add_adsb_icao_info(df_flights, day_dt)
    df_flights = resolve_sfdps_airports(
        df_flights,
        airport_catalog,
    )
    df_flights = df_flights.sort(["icao", "takeoff_time"])
    return df_flights


@dataclass(frozen=True)
class SFDPSFlightSource(NonADSBFlightSource):
    name: str = "sfdps"
    version: int = 14

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_sfdps_flights_day(dt, airport_catalog)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/load SFDPS flights for one UTC day."
    )
    parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute instead of reading cached parquet when present.",
    )
    args = parser.parse_args()

    target_day = date.fromisoformat(args.date)
    df_flights = get_flights(
        SFDPSFlightSource(),
        target_day,
        use_cache=not args.no_cache,
    )
    print(df_flights)


if __name__ == "__main__":
    main()
