from collections.abc import Sequence
from datetime import date, datetime, timedelta
from enum import StrEnum
import os
import shutil
from pathlib import Path
from typing import Literal, overload

import polars as pl

from data_engineering.filters import pia_or_american_ladd_icao
from data_engineering.adsb.partitions import (
    icao_bucket as _icao_bucket,
    query_icao_buckets as _query_buckets,
)
from data_engineering.utils import OUTPUT_DIR


class ADSBSource(StrEnum):
    ADSBLOL = "adsblol"
    ADSBX = "adsbx"
    OPENSKY = "opensky"
ADSBLOL_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"
ADSBX_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsb-exchange/traces_parquet"
EMPTY_COLUMN_DTYPES = {
    "time": pl.Datetime("ms"),
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "aircraft_type": pl.String,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "baro_altitude_ft": pl.Int32,
    "geom_altitude_ft": pl.Int32,
    "baro_rate_fpm": pl.Int32,
    "ground_speed_kt": pl.Float32,
    "track_deg": pl.Float32,
    "on_ground": pl.Boolean,
    "pia": pl.Boolean,
    "ladd": pl.Boolean,
    "military": pl.Boolean,
    "interesting": pl.Boolean,
    "owner": pl.String,
    "aircraft_description": pl.String,
    "category": pl.String,
}


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


def _day_partition_dir(root: Path, current_day: date) -> Path:
    return root / f"year={current_day.year}" / f"month={current_day.month:02d}" / f"day={current_day.day:02d}"


def _parquet_day_dir(current_day: date) -> Path:
    return _day_partition_dir(ADSBLOL_PARQUET_ROOT, current_day)


def _adsbx_parquet_day_dir(current_day: date) -> Path:
    return _day_partition_dir(ADSBX_PARQUET_ROOT, current_day)


def _adsbx_parquet_path(current_day: date) -> str:
    return str(_adsbx_parquet_day_dir(current_day) / "icao_bucket=*" / "data.parquet")


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return any(partition_dir.glob("icao_bucket=*/data.parquet"))


def _parquet_day_has_data(current_day: date) -> bool:
    return _parquet_partition_has_data(_parquet_day_dir(current_day))


def _restore_partition_from_s3(partition_dir: Path) -> None:
    if os.getenv("PLANEQUERY_READ_FROM_S3") != "1":
        return
    from data_engineering.utils import download_prefix_from_s3

    downloaded = download_prefix_from_s3(partition_dir)
    if downloaded:
        print(f"Downloaded {downloaded} parquet files from S3 into {partition_dir}")


def create_adsbx_parquet_for_day(current_day: date):
    partition_dir = _adsbx_parquet_day_dir(current_day)
    if not _parquet_partition_has_data(partition_dir):
        _restore_partition_from_s3(partition_dir)
    if _parquet_partition_has_data(partition_dir):
        return
    if current_day.day != 1:
        raise FileNotFoundError(
            "ADSBExchange sample readsb-hist data is only available for the "
            f"first day of each month; no source data is expected for {current_day}."
        )

    from data_engineering.adsbx.readsb import ensure_readsb_day_bucketed_parquet

    ensure_readsb_day_bucketed_parquet(
        current_day,
        output_root=ADSBX_PARQUET_ROOT,
    )


def _parquet_read_paths(
    current_day: date,
    icao_buckets: Sequence[int] | None = None,
) -> str | list[str]:
    day_dir = _parquet_day_dir(current_day)
    if _parquet_partition_has_data(day_dir):
        if icao_buckets is None:
            return str(day_dir / "icao_bucket=*" / "data.parquet")
        paths = [
            day_dir / f"icao_bucket={bucket}" / "data.parquet"
            for bucket in sorted(set(icao_buckets))
        ]
        return [str(path) for path in paths if path.exists()]

    return []


def _column_list(
    columns: Sequence[str],
    additional_columns: Sequence[str] | None = None,
) -> list[str]:
    return list(dict.fromkeys(list(columns) + list(additional_columns or [])))


def _empty_scan(columns: Sequence[str]) -> pl.LazyFrame:
    return pl.DataFrame(
        schema={column: EMPTY_COLUMN_DTYPES.get(column, pl.Null) for column in columns}
    ).lazy()


def create_adsb_parquet_for_day(
    current_day: date,
    *,
    pia_or_american_ladd_only: bool = False,
) -> bool:
    """Ensure a parquet partition exists and report whether it was created locally."""
    if not _parquet_day_has_data(current_day):
        _restore_partition_from_s3(_parquet_day_dir(current_day))
    if _parquet_day_has_data(current_day):
        return False

    from data_engineering.adsb.download_adsb_to_parquet import create_parquet_for_day
    create_parquet_for_day(
        current_day,
        keep_folders=True,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    return _parquet_day_has_data(current_day)


def remove_adsb_parquet_days(days: Sequence[date]) -> None:
    """Remove locally-created adsb.lol parquet partitions after their final use."""
    for day in days:
        partition_dir = _parquet_day_dir(day)
        if partition_dir.exists():
            print(f"Removing temporary parquet partition: {partition_dir}")
            shutil.rmtree(partition_dir)


def scan_adsb(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
):
    create_adsb_parquet_for_day(
        current_day,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    columns = _column_list(columns, additional_columns)
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = list(dict.fromkeys(columns + ["time", "icao", *filter_columns]))
    parquet_paths = _parquet_read_paths(
        current_day,
        _query_buckets(icaos=icaos),
    )
    if not parquet_paths:
        return _empty_scan(columns)

    # Some day partitions can contain newly added columns (for example
    # `aircraft-year`) while older files do not. Ignore extra columns so
    # mixed-schema partitions remain readable.
    lf = pl.scan_parquet(parquet_paths, extra_columns="ignore").select(scan_columns)
    lf = lf.filter(pl.col("time").dt.date() == current_day)
    if icaos is not None:
        lf = lf.filter(pl.col("icao").is_in(list(icaos)))
    if pia_or_american_ladd_only:
        lf = lf.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)


def _date_range_inclusive(start_day: date, end_day: date) -> list[date]:
    return [start_day + timedelta(days=offset) for offset in range((end_day - start_day).days + 1)]


def _date_range_for_interval(start_dt: datetime, end_dt: datetime) -> list[date]:
    if end_dt <= start_dt:
        return []
    end_day = (end_dt - timedelta(microseconds=1)).date()
    return _date_range_inclusive(start_dt.date(), end_day)


def _scan_adsbx_days(
    days: Sequence[date],
    columns: Sequence[str],
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
) -> pl.LazyFrame:
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = list(dict.fromkeys(list(columns) + ["time", "icao", *filter_columns]))
    for day in days:
        create_adsbx_parquet_for_day(day)
    buckets = _query_buckets(icaos=icaos)
    paths = []
    for day in days:
        day_dir = _adsbx_parquet_day_dir(day)
        if not _parquet_partition_has_data(day_dir):
            continue
        if buckets is None:
            paths.append(_adsbx_parquet_path(day))
        else:
            paths.extend(
                str(path)
                for bucket in buckets
                if (path := day_dir / f"icao_bucket={bucket}" / "data.parquet").exists()
            )
    if not paths:
        return _empty_scan(columns)

    lf = pl.scan_parquet(paths, extra_columns="ignore").select(scan_columns)
    if icaos is not None:
        lf = lf.filter(pl.col("icao").is_in(list(icaos)))
    if pia_or_american_ladd_only:
        lf = lf.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)


def _scan_opensky_days(
    days: Sequence[date],
    columns: Sequence[str],
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
) -> pl.LazyFrame:
    from data_engineering.openairframes.read import scan_latest_icao_info_for_join
    from data_engineering.opensky.read_opensky_trino_states import (
        OPENAIRFRAMES_COLUMNS,
        join_lf_with_openairframes,
        scan_cached_opensky_state_vectors_for_day,
    )
    from data_engineering.opensky.preprocess import (
        OPENSKY_PREPROCESSING_COLUMNS,
        preprocess_opensky_adsb,
    )

    metadata_columns = set(OPENAIRFRAMES_COLUMNS)
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = list(dict.fromkeys(list(columns) + ["time", "icao", *filter_columns]))
    cached_columns = list(
        dict.fromkeys(
            [
                *[
                    column
                    for column in scan_columns
                    if column not in metadata_columns
                ],
                *OPENSKY_PREPROCESSING_COLUMNS,
            ]
        )
    )

    lf = pl.concat(
        [
            scan_cached_opensky_state_vectors_for_day(
                day,
                columns=cached_columns,
                icaos=icaos,
            )
            for day in days
        ]
    )
    if icaos is not None:
        lf = lf.filter(pl.col("icao").is_in([icao.lower() for icao in icaos]))
    lf = preprocess_opensky_adsb(lf)

    needs_metadata = bool(metadata_columns.intersection(scan_columns))
    if needs_metadata or pia_or_american_ladd_only:
        df_openairframes = scan_latest_icao_info_for_join().with_columns(
            pl.col("icao").str.to_lowercase()
        )
        lf = join_lf_with_openairframes(lf, df_openairframes)

    if pia_or_american_ladd_only:
        lf = lf.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)


def ensure_adsb_source_data(
    start: date | datetime,
    end: datetime | None = None,
    *,
    source: ADSBSource = ADSBSource.ADSBLOL,
    pia_or_american_ladd_only: bool = False,
) -> list[date]:
    if not isinstance(source, ADSBSource):
        raise TypeError(f"source must be an ADSBSource, got {type(source).__name__}")
    if isinstance(start, datetime):
        start_dt = start
        end_dt = end if end is not None else start_dt + timedelta(days=1)
    else:
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = end if end is not None else start_dt + timedelta(days=1)

    days = _date_range_for_interval(start_dt, end_dt)
    if source is ADSBSource.ADSBLOL:
        return [
            day
            for day in days
            if create_adsb_parquet_for_day(
                day,
                pia_or_american_ladd_only=pia_or_american_ladd_only,
            )
        ]
    elif source is ADSBSource.ADSBX:
        for day in days:
            create_adsbx_parquet_for_day(day)
    elif source is ADSBSource.OPENSKY:
        from data_engineering.opensky.read_opensky_trino_states import (
            write_opensky_state_vectors_cache_for_day,
        )

        for day in days:
            write_opensky_state_vectors_cache_for_day(day)
    else:
        raise AssertionError(f"Unhandled ADS-B source: {source!r}")
    return []


@overload
def read_adsb(
    start: date | datetime,
    end: datetime | None = None,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
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
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: list[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    lazy: Literal[True],
) -> pl.LazyFrame: ...


def read_adsb(
    start: date | datetime,
    end: datetime | None = None,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: list[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read ADS-B messages.

    Pass a `date` for `start` to read a single full day, or `datetime`s for
    `start`/`end` to read a `[start, end)` range spanning one or more days
    (e.g. a context window around a target day).

    `source` selects the data provider with `ADSBSource`. Set `lazy=True` to
    return a `polars.LazyFrame`; by default the result is collected and returned
    as a `polars.DataFrame`.
    """
    if not isinstance(source, ADSBSource):
        raise TypeError(f"source must be an ADSBSource, got {type(source).__name__}")
    if isinstance(start, datetime):
        start_dt = start
        end_dt = end if end is not None else start_dt + timedelta(days=1)
    else:
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = end if end is not None else start_dt + timedelta(days=1)

    requested_columns = _column_list(columns, additional_columns)
    scan_columns = _column_list(requested_columns, ["time"])
    days = _date_range_for_interval(start_dt, end_dt)

    if icaos is not None and not icaos:
        empty = _empty_scan(requested_columns)
        return empty if lazy else empty.collect()

    if source is ADSBSource.OPENSKY:
        lf = _scan_opensky_days(
            days,
            scan_columns,
            icaos=icaos,
            pia_or_american_ladd_only=pia_or_american_ladd_only,
        )
    elif source is ADSBSource.ADSBLOL:
        lf = pl.concat(
            [
                scan_adsb(
                    day,
                    columns=scan_columns,
                    icaos=icaos,
                    pia_or_american_ladd_only=pia_or_american_ladd_only,
                )
                for day in days
            ]
        )
    elif source is ADSBSource.ADSBX:
        lf = _scan_adsbx_days(
            days,
            scan_columns,
            icaos=icaos,
            pia_or_american_ladd_only=pia_or_american_ladd_only,
        )
    else:
        raise AssertionError(f"Unhandled ADS-B source: {source!r}")
    lf = lf.filter((pl.col("time") >= start_dt) & (pl.col("time") < end_dt)).select(requested_columns)
    return lf if lazy else lf.collect()


def read_icaos_from_adsb(current_day: date, icaos: list[str], columns=DEFAULT_COLUMNS_SET):
    return read_adsb(current_day, columns=columns, icaos=icaos)



def get_icaos_in_adsb(
    current_day: date,
    source: ADSBSource = ADSBSource.ADSBLOL,
) -> list[str]:
    '''
    List of distinct ICAOs seen on `current_day` for the given `source`.
    Invalid ICAOs are filtered out for the adsb.lol source.
    '''
    if not isinstance(source, ADSBSource):
        raise TypeError(f"source must be an ADSBSource, got {type(source).__name__}")
    if source is ADSBSource.OPENSKY:
        from data_engineering.opensky.read_opensky_trino_states import get_icaos_in_opensky
        return get_icaos_in_opensky(current_day)
    if source is ADSBSource.ADSBX:
        df = _scan_adsbx_days([current_day], ["icao"]).unique().collect()
        return df.get_column("icao").drop_nulls().to_list()
    if source is ADSBSource.ADSBLOL:
        df = scan_adsb(current_day, columns=["icao"]).filter(exclude_invalid_icaos()).unique().collect()
        return df["icao"].to_list()
    raise AssertionError(f"Unhandled ADS-B source: {source!r}")

def exclude_invalid_icaos(col: str = "icao") -> pl.Expr:
    return (pl.col(col) >= "010000") & (pl.col(col) != "ffffff") & ~pl.col(col).str.starts_with("~")


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
    current_day = datetime(2025, 8, 1)
    df_adsb = read_adsb(current_day)
    print(df_adsb)
