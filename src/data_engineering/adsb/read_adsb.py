"""Read provider-independent canonical ADS-B Parquet."""

import os
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Literal, overload

import polars as pl

from data_engineering.adsb.canonical import (
    ADSBSource,
    adsb_day_dir,
    adsb_day_is_complete,
    coerce_adsb_source,
    empty_adsb_scan,
    scan_adsb_parquet,
)
from data_engineering.filters import pia_or_american_ladd_icao


DEFAULT_COLUMNS_SET = [
    "time",
    "icao",
    "registration",
    "callsign",
    "lat",
    "lon",
    "baro_altitude_ft",
    "on_ground",
    "ground_speed_kt",
    "track_deg",
    "aircraft_type",
    "aircraft_description",
    "owner",
    "category",
    "pia",
    "ladd",
    "military",
    "interesting",
]


def _column_list(
    columns: Sequence[str],
    additional_columns: Sequence[str] | None = None,
) -> list[str]:
    return list(dict.fromkeys([*columns, *(additional_columns or [])]))


def _date_range_inclusive(start_day: date, end_day: date) -> list[date]:
    return [
        start_day + timedelta(days=offset)
        for offset in range((end_day - start_day).days + 1)
    ]


def _date_range_for_interval(start_dt: datetime, end_dt: datetime) -> list[date]:
    if end_dt <= start_dt:
        return []
    end_day = (end_dt - timedelta(microseconds=1)).date()
    return _date_range_inclusive(start_dt.date(), end_day)


def _restore_partition_from_s3(source: ADSBSource, day: date) -> None:
    if os.getenv("PLANEQUERY_READ_FROM_S3") != "1":
        return
    from data_engineering.utils import download_prefix_from_s3

    partition_dir = adsb_day_dir(source, day)
    downloaded = download_prefix_from_s3(partition_dir)
    if downloaded:
        print(f"Downloaded {downloaded} files from S3 into {partition_dir}")


def _create_provider_parquet(source: ADSBSource, day: date) -> None:
    if source is ADSBSource.ADSBLOL:
        from data_engineering.adsb.download_adsb_to_parquet import (
            create_parquet_for_day,
        )

        create_parquet_for_day(day, keep_folders=True)
        return

    if source is ADSBSource.ADSBX:
        if day.day != 1:
            raise FileNotFoundError(
                "ADSBX sample readsb-hist data is only available for "
                f"the first day of each month; no source data is expected for {day}."
            )
        from data_engineering.adsbx.readsb import (
            ensure_readsb_day_bucketed_parquet,
        )

        ensure_readsb_day_bucketed_parquet(day)
        return

    if source is ADSBSource.OPENSKY:
        from data_engineering.opensky.read_opensky_trino_states import (
            write_opensky_state_vectors_cache_for_day,
        )

        write_opensky_state_vectors_cache_for_day(day)
        return

    raise AssertionError(f"Unhandled ADS-B source: {source!r}")


def ensure_adsb_day(source: ADSBSource, day: date) -> bool:
    """Ensure one canonical provider partition exists; report local creation."""
    source = coerce_adsb_source(source)
    if adsb_day_is_complete(source, day):
        return False
    _restore_partition_from_s3(source, day)
    if adsb_day_is_complete(source, day):
        return False

    _create_provider_parquet(source, day)
    if not adsb_day_is_complete(source, day):
        raise RuntimeError(
            f"{source.value} conversion did not complete canonical partition "
            f"{adsb_day_dir(source, day)}"
        )
    return True


def ensure_adsb_source_data(
    start: date | datetime,
    end: datetime | None = None,
    *,
    source: ADSBSource = ADSBSource.ADSBLOL,
) -> list[date]:
    """Ensure canonical Parquet for every day touched by an interval."""
    source = coerce_adsb_source(source)
    if isinstance(start, datetime):
        start_dt = start
        end_dt = end if end is not None else start_dt + timedelta(days=1)
    else:
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = end if end is not None else start_dt + timedelta(days=1)

    return [
        day
        for day in _date_range_for_interval(start_dt, end_dt)
        if ensure_adsb_day(source, day)
    ]


def _filter_adsb_scan(
    lf: pl.LazyFrame,
    columns: Sequence[str],
    *,
    pia_or_american_ladd_only: bool,
) -> pl.LazyFrame:
    if pia_or_american_ladd_only:
        lf = lf.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        lf = lf.with_columns(
            pl.col("callsign").str.replace_all(" ", "").alias("callsign")
        )
    return lf.select(columns)


def scan_adsb(
    current_day: date,
    columns: Sequence[str] = DEFAULT_COLUMNS_SET,
    additional_columns: Sequence[str] | None = None,
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
) -> pl.LazyFrame:
    """Ensure and scan one canonical ADS-B day."""
    source = coerce_adsb_source(source)
    ensure_adsb_day(source, current_day)
    requested_columns = _column_list(columns, additional_columns)
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = _column_list(
        requested_columns,
        ["time", "icao", *filter_columns],
    )
    lf = scan_adsb_parquet(
        source,
        [current_day],
        scan_columns,
        icaos=icaos,
    ).filter(pl.col("time").dt.date() == current_day)
    return _filter_adsb_scan(
        lf,
        requested_columns,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )


@overload
def read_adsb(
    start: date | datetime,
    end: datetime | None = None,
    columns: Sequence[str] = DEFAULT_COLUMNS_SET,
    additional_columns: Sequence[str] | None = None,
    *,
    icaos: list[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


@overload
def read_adsb(
    start: date | datetime,
    end: datetime | None = None,
    columns: Sequence[str] = DEFAULT_COLUMNS_SET,
    additional_columns: Sequence[str] | None = None,
    *,
    icaos: list[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    lazy: Literal[True],
) -> pl.LazyFrame: ...


def read_adsb(
    start: date | datetime,
    end: datetime | None = None,
    columns: Sequence[str] = DEFAULT_COLUMNS_SET,
    additional_columns: Sequence[str] | None = None,
    *,
    icaos: list[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read a ``[start, end)`` interval from canonical provider Parquet."""
    source = coerce_adsb_source(source)
    if isinstance(start, datetime):
        start_dt = start
        end_dt = end if end is not None else start_dt + timedelta(days=1)
    else:
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = end if end is not None else start_dt + timedelta(days=1)

    requested_columns = _column_list(columns, additional_columns)
    if icaos is not None and not icaos:
        empty = empty_adsb_scan(requested_columns)
        return empty if lazy else empty.collect()

    days = _date_range_for_interval(start_dt, end_dt)
    for day in days:
        ensure_adsb_day(source, day)

    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = _column_list(
        requested_columns,
        ["time", "icao", *filter_columns],
    )
    lf = scan_adsb_parquet(source, days, scan_columns, icaos=icaos)
    lf = lf.filter(
        (pl.col("time") >= start_dt) & (pl.col("time") < end_dt)
    )
    lf = _filter_adsb_scan(
        lf,
        requested_columns,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    return lf if lazy else lf.collect()


def read_icaos_from_adsb(
    current_day: date,
    icaos: list[str],
    columns: Sequence[str] = DEFAULT_COLUMNS_SET,
) -> pl.DataFrame:
    return read_adsb(current_day, columns=columns, icaos=icaos)


def get_icaos_in_adsb(
    current_day: date,
    source: ADSBSource = ADSBSource.ADSBLOL,
) -> list[str]:
    """List distinct ICAOs seen in one canonical provider partition."""
    source = coerce_adsb_source(source)
    lf = scan_adsb(current_day, columns=["icao"], source=source)
    if source is ADSBSource.ADSBLOL:
        lf = lf.filter(exclude_invalid_icaos())
    return (
        lf.drop_nulls()
        .unique()
        .collect(engine="streaming")
        .get_column("icao")
        .to_list()
    )


def exclude_invalid_icaos(col: str = "icao") -> pl.Expr:
    return (
        (pl.col(col) >= "010000")
        & (pl.col(col) != "ffffff")
        & ~pl.col(col).str.starts_with("~")
    )


def filter_ladd_icaos(col: str = "is_ladd") -> pl.Expr:
    return pl.col(col).any().over("icao")


def filter_fixed_wing_icaos(col: str = "emitter_category") -> pl.Expr:
    return pl.col(col).is_in(["A1", "A2", "A3", "A4", "A5"]).any().over("icao")


def most_common_category(col: str = "emitter_category") -> pl.Expr:
    return (
        pl.col(col)
        .filter(pl.col(col) != "")
        .mode()
        .first()
        .over("icao")
    )


if __name__ == "__main__":
    print(read_adsb(datetime(2025, 8, 1)))
