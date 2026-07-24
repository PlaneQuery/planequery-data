import argparse
import os
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import scan_adsb
from data_engineering.flights.flight_type import with_flight_schema_columns
from data_engineering.swim.read_sfdps import read_sfdps
from data_engineering.swim.utils import MISSING_SFDPS_DAYS
from data_engineering.utils import OUTPUT_DIR


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

REGISTRATION_CALLSIGN_PATTERN = (
    r"^(?:[A-Z]{1,3}-[A-Z0-9]{2,5}|"
    r"N(?:[1-9][0-9]{0,4}|"
    r"[1-9][0-9]{0,3}[A-HJ-NP-Z]|"
    r"[1-9][0-9]{0,2}[A-HJ-NP-Z]{2}))$"
)


def _read_sfdps_for_day(current_day: datetime) -> pl.DataFrame:
    return read_sfdps(current_day)


def _missing_sfdps_days() -> set[date]:
    return set(MISSING_SFDPS_DAYS)


def gufi_state_machine(df: pl.DataFrame) -> pl.DataFrame:
    landing_time = None
    takeoff_time = None
    landing_airport = None
    takeoff_airport = None
    estimated_landing_time = None
    estimated_takeoff_time = None
    flight_status = None

    for _, row in enumerate(df.iter_rows(named=True)):
        if row["takeoff_time"] is not None:
            takeoff_time = row["takeoff_time"]
        if row["takeoff_airport_icao"] is not None:
            takeoff_airport = row["takeoff_airport_icao"]
        if row["landing_time"] is not None:
            landing_time = row["landing_time"]
        if row["landing_airport_icao"] is not None:
            landing_airport = row["landing_airport_icao"]
        if row["estimated_landing_time"] is not None:
            estimated_landing_time = row["estimated_landing_time"]
        if row["estimated_takeoff_time"] is not None:
            estimated_takeoff_time = row["estimated_takeoff_time"]

        if row["flightStatus"] == "CANCELLED":
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
                    "callsign": [row["callsign"]],
                    "flightStatus": [flight_status],
                    "takeoff_time": [takeoff_time],
                    "takeoff_airport_icao": [takeoff_airport],
                    "landing_time": [landing_time],
                    "landing_airport_icao": [landing_airport],
                    "estimated_landing_time": [estimated_landing_time],
                    "estimated_takeoff_time": [estimated_takeoff_time],
                },
                schema={
                    "gufi": pl.String,
                    "timestamp": pl.Datetime("us"),
                    "callsign": pl.String,
                    "flightStatus": pl.String,
                    "takeoff_time": pl.Datetime("us"),
                    "takeoff_airport_icao": pl.String,
                    "landing_time": pl.Datetime("us"),
                    "landing_airport_icao": pl.String,
                    "estimated_landing_time": pl.Datetime("us"),
                    "estimated_takeoff_time": pl.Datetime("us"),
                },
            )
    if flight_status == "DROPPED":
        pass

    return df.clear()


def sfdps_to_flights(current_day: datetime) -> pl.DataFrame:
    df = _read_sfdps_for_day(current_day)
    df = df.group_by("gufi").map_groups(gufi_state_machine)
    df = (
        df.rename(
            {
                "takeoff_airport_icao": "takeoff_airport_ident",
                "landing_airport_icao": "landing_airport_ident",
            }
        )
        .with_columns(
            pl.lit(None).cast(pl.String).alias("icao"),
            pl.when(pl.col("callsign").str.contains(REGISTRATION_CALLSIGN_PATTERN))
            .then(pl.col("callsign"))
            .otherwise(pl.lit(None))
            .alias("registration"),
            pl.lit(False).alias("pia"),
            pl.lit(False).alias("ladd"),
            pl.lit(False).alias("military"),
            pl.lit(False).alias("interesting"),
            pl.lit("").alias("aircraft_type"),
            pl.lit("").alias("owner"),
            pl.lit("").alias("aircraft_description"),
            pl.lit("").alias("category"),
        )
    )
    return with_flight_schema_columns(df)


def filter_sfdps_flights(df: pl.DataFrame, current_date: datetime) -> pl.DataFrame:
    day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    df = df.filter(
        (pl.col("takeoff_time").dt.date() == current_date.date())
        & (pl.col("landing_time").dt.date() == current_date.date())
        & (pl.col("takeoff_time") != day_start)
    )
    df = df.with_columns((pl.col("landing_time") - pl.col("takeoff_time")).alias("flight_duration"))
    df = df.filter(pl.col("flight_duration") > pl.duration(minutes=20))
    df = df.drop("flight_duration")
    return df


def compress_callsign_segments(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("callsign") != "")
    df = df.with_columns((pl.int_range(0, pl.len()) == pl.len() - 1).alias("is_last"))
    df = (
        df.with_columns((pl.col("callsign") != pl.col("callsign").shift(1)).alias("_new_callsign"))
        .fill_null(True)
    )
    df = df.with_columns(
        pl.when(pl.col("_new_callsign").shift(-1).eq(True) | pl.col("is_last"))
        .then(pl.col("time"))
        .alias("end_time")
        .backward_fill()
    )
    df = df.filter(pl.col("_new_callsign"))
    df = df.rename({"time": "start_time"})
    df = df.drop("_new_callsign", "is_last")
    return df


def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()


def _adsb_registration_map(current_day: datetime, registrations: list[str]) -> pl.DataFrame:
    return _collect_streaming(
        scan_adsb(current_day.date(), columns=["icao", "registration", "pia", "ladd", "time"])
        .filter((pl.col("registration") != "") & pl.col("registration").is_in(registrations))
        .group_by(["icao", "registration"])
        .agg(
            pl.len().alias("len"),
            pl.col("pia").any().alias("pia"),
            pl.col("ladd").any().alias("ladd"),
        )
        .sort(["registration", "len", "icao"], descending=[False, True, False])
        .group_by("registration", maintain_order=True)
        .first()
        .select(["icao", "registration", "pia", "ladd"])
    )


def _adsb_callsign_segments(current_day: datetime, callsigns: list[str]) -> pl.DataFrame:
    return _collect_streaming(
        scan_adsb(current_day.date(), columns=["icao", "callsign", "pia", "ladd", "time"])
        .filter((pl.col("callsign") != "") & pl.col("callsign").is_in(callsigns))
        .sort(["icao", "time", "callsign"])
        .with_columns(
            (
                (pl.col("callsign") != pl.col("callsign").shift(1).over("icao"))
                | pl.col("callsign").shift(1).over("icao").is_null()
            ).alias("_new_callsign")
        )
        .with_columns(pl.col("_new_callsign").cum_sum().over("icao").alias("_segment_id"))
        .group_by(["icao", "callsign", "_segment_id"])
        .agg(
            pl.col("time").min().alias("start_time"),
            pl.col("time").max().alias("end_time"),
            pl.col("pia").any().alias("pia"),
            pl.col("ladd").any().alias("ladd"),
        )
        .select(["icao", "callsign", "start_time", "end_time", "pia", "ladd"])
    )


def add_adsb_icao_info(df_flights: pl.DataFrame, current_day: datetime) -> pl.DataFrame:
    icaos = df_flights["icao"].drop_nulls().unique().to_list()
    df_icao_info = _collect_streaming(
        scan_adsb(current_day.date(), columns=["icao", "time", *ADSB_ICAO_INFO_COLUMNS])
        .filter(pl.col("icao").is_in(icaos))
        .sort(["icao", "time"])
        .group_by("icao")
        .last()
        .select(["icao", *ADSB_ICAO_INFO_COLUMNS])
    )
    df_flights = (
        df_flights.drop(ADSB_ICAO_INFO_COLUMNS)
        .join(df_icao_info, on="icao", how="left")
        .with_columns(
            pl.col("registration").fill_null(""),
            pl.col("aircraft_type").fill_null(""),
            pl.col("owner").fill_null(""),
            pl.col("aircraft_description").fill_null(""),
            pl.col("category").fill_null(""),
            pl.col("pia").fill_null(False),
            pl.col("ladd").fill_null(False),
            pl.col("military").fill_null(False),
            pl.col("interesting").fill_null(False),
        )
    )
    return with_flight_schema_columns(df_flights)


def sfdps_flights_derive_icao(df_sfdps_flights: pl.DataFrame, current_day: datetime) -> pl.DataFrame:
    df_sfdps_flights_with_registration = df_sfdps_flights.filter(pl.col("registration").is_not_null())
    df_sfdps_flights_without_registration = df_sfdps_flights.filter(pl.col("registration").is_null())
    registrations = df_sfdps_flights_with_registration["registration"].drop_nulls().unique().to_list()
    callsigns = (
        df_sfdps_flights_without_registration.filter(pl.col("callsign") != "")
        ["callsign"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    df_adsb_registration_map = _adsb_registration_map(current_day, registrations)
    df_sfdps_flights_0 = (
        df_sfdps_flights_with_registration
        .join(
            df_adsb_registration_map.select("registration", "icao"),
            on="registration",
            how="inner",
            suffix="_adsb",
        )
        .with_columns(
            pl.col("icao_adsb").alias("icao"),
        )
        .drop("icao_adsb")
    )
    df_adsb_map = _adsb_callsign_segments(current_day, callsigns)
    matched_callsigns = set(df_adsb_map["callsign"].unique().to_list())
    missing_callsign_count = len(set(callsigns) - matched_callsigns)
    print(f"SFDPS callsigns with no ADS-B match: {missing_callsign_count}/{len(callsigns)}")
    _30min = pl.duration(minutes=30)

    df_sfdps_flights_1 = (
        df_sfdps_flights_without_registration
        .with_row_index("_row_idx")
        .join(df_adsb_map.select("icao", "callsign", "start_time", "end_time"), on="callsign", how="left")
        .with_columns(
            (
                (pl.col("takeoff_time") >= pl.col("start_time"))
                & (pl.col("takeoff_time") <= pl.col("end_time"))
            ).alias("_takeoff_in_range"),
            (
                (pl.col("landing_time") >= pl.col("start_time"))
                & (pl.col("landing_time") <= pl.col("end_time"))
            ).alias("_landing_in_range"),
        )
        .filter(
            pl.col("_takeoff_in_range")
            | pl.col("_landing_in_range")
            | ((pl.col("takeoff_time") - pl.col("start_time")).abs() <= _30min)
            | ((pl.col("takeoff_time") - pl.col("end_time")).abs() <= _30min)
            | ((pl.col("landing_time") - pl.col("start_time")).abs() <= _30min)
            | ((pl.col("landing_time") - pl.col("end_time")).abs() <= _30min)
        )
        .with_columns(pl.col("icao_right").n_unique().over("_row_idx").alias("_icao_count"))
        .filter(pl.col("_icao_count") == 1)
        .with_columns(
            pl.when(pl.col("_takeoff_in_range"))
            .then(0)
            .when(pl.col("_landing_in_range"))
            .then(1)
            .otherwise(2)
            .alias("_priority"),
            (
                (pl.col("takeoff_time") - pl.col("start_time")).abs()
                + (pl.col("landing_time") - pl.col("end_time")).abs()
            ).alias("_time_dist"),
        )
        .sort(["_priority", "_time_dist"], nulls_last=True)
        .group_by("_row_idx", maintain_order=True)
        .first()
        .with_columns(
            pl.col("icao").fill_null(pl.col("icao_right")),
        )
        .drop(
            "_row_idx",
            "icao_right",
            "start_time",
            "end_time",
            "_takeoff_in_range",
            "_landing_in_range",
            "_priority",
            "_time_dist",
            "_icao_count",
        )
    )
    df_flights = pl.concat([df_sfdps_flights_0, df_sfdps_flights_1])
    return add_adsb_icao_info(df_flights, current_day)


def sfdps_flights_parquet_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "sfdps"
        / "v1"
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "part-0.parquet"
    )


def create_sfdps_flights_day(target_date: date) -> pl.DataFrame:
    day_dt = datetime(target_date.year, target_date.month, target_date.day)
    df_flights = sfdps_to_flights(day_dt)
    df_flights = filter_sfdps_flights(df_flights, day_dt)
    df_flights = sfdps_flights_derive_icao(df_flights, day_dt)
    df_flights = df_flights.sort(["icao", "takeoff_time"])
    return df_flights


def get_sfdps_flights_day(target_date: date, use_cache: bool = True) -> pl.DataFrame:
    missing_sfdps_days = _missing_sfdps_days()
    if target_date in missing_sfdps_days or target_date < date(2026, 1, 1):
        raise Exception("trying to load missing data")

    parquet_path = sfdps_flights_parquet_path(target_date)
    if use_cache and parquet_path.exists():
        return pl.read_parquet(parquet_path)

    df_flights = create_sfdps_flights_day(target_date)
    os.makedirs(parquet_path.parent, exist_ok=True)
    df_flights.write_parquet(parquet_path)
    return df_flights


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/load SFDPS flights for one UTC day.")
    parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute instead of reading cached parquet when present.",
    )
    args = parser.parse_args()

    target_day = date.fromisoformat(args.date)
    df_flights = get_sfdps_flights_day(target_day, use_cache=not args.no_cache)
    print(df_flights)


if __name__ == "__main__":
    main()
