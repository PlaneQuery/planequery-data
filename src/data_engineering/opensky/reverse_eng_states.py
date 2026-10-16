"""Lossless input subset for the reverse-engineered OpenSky flight algorithm.

Canonical ADS-B discards the snapshot clock and changes samples/altitudes.
This separate cache preserves both clocks, metre altitudes and callsigns. It
only applies the algorithm's initial freshness/finite-position predicate.
"""

import fcntl
import shutil
import tempfile
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl

from data_engineering.adsb.partitions import (
    ICAO_BUCKET_COUNT,
    ICAO_BUCKET_SIZE,
    query_icao_buckets,
)
from data_engineering.opensky.paths import partitioned_day_dir
from data_engineering.opensky.read_opensky_trino_states import (
    complete_opensky_state_vector_files,
)
from data_engineering.utils import OUTPUT_DIR
from planequery_data.utils import date_range

STATE_CACHE_VERSION = 1
STATE_CACHE_ROOT = (
    OUTPUT_DIR
    / "data/intermediate/opensky/reverse-eng-states"
    / f"version={STATE_CACHE_VERSION}"
)
STATE_SCHEMA = {
    "icao": pl.String,
    "time": pl.Int64,
    "lastposupdate": pl.Float64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "baroaltitude": pl.Float64,
    "geoaltitude": pl.Float64,
    "callsign": pl.String,
}


def valid_position() -> pl.Expr:
    """Keep fresh finite positions, including those marked on the ground."""
    return pl.all_horizontal(
        pl.col(c).is_finite() for c in ("lastposupdate", "lat", "lon", "baroaltitude")
    ) & ((pl.col("time") - pl.col("lastposupdate")) < 1)


def ensure_reverse_eng_states(day: date) -> Path:
    """Read raw hours once and stage sorted aircraft buckets for batch reuse.

    The lock also protects overlapping days requested by different workers.
    Incomplete raw days fail before writing; only a fully built day is exposed.
    """
    output = partitioned_day_dir(STATE_CACHE_ROOT, day)
    success = output / "_SUCCESS"
    if success.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if success.is_file():
            return output
        files = complete_opensky_state_vector_files(day)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            for hour, path in enumerate(files):
                print(
                    f"Preparing OpenSky reverse-eng states: {day} {hour:02d}/23",
                    flush=True,
                )
                states = (
                    pl.scan_parquet(path, hive_partitioning=False)
                    .filter(valid_position())
                    .rename({"icao24": "icao"})
                    .select(pl.col(c).cast(dtype) for c, dtype in STATE_SCHEMA.items())
                    .with_columns(
                        pl.col("icao")
                        .str.to_integer(base=16, strict=False)
                        .floordiv(ICAO_BUCKET_SIZE)
                        .clip(0, ICAO_BUCKET_COUNT - 1)
                        .fill_null(ICAO_BUCKET_COUNT - 1)
                        .cast(pl.UInt8)
                        .alias("icao_bucket")
                    )
                    .sort(["icao", "time"], maintain_order=True)
                )

                def hourly_path(args, current_hour=hour):
                    bucket = args.partition_keys.get_column("icao_bucket").item()
                    return (
                        staging
                        / f"icao_bucket={bucket}"
                        / f"hour={current_hour:02d}-{args.index_in_partition}.parquet"
                    )

                states.sink_parquet(
                    pl.PartitionBy(
                        staging,
                        key="icao_bucket",
                        include_key=False,
                        file_path_provider=hourly_path,
                    ),
                    compression="zstd",
                    row_group_size=25_000,
                    maintain_order=True,
                    mkdir=True,
                )
            (staging / "_SUCCESS").touch()
            if output.exists():
                shutil.rmtree(output)
            staging.replace(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return output


def read_reverse_eng_states(
    start: date | datetime,
    end: datetime | None = None,
    *,
    columns: Sequence[str] | None = None,
    icaos: Sequence[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read a snapshot-time interval without canonical ADS-B normalization."""
    start_dt = (
        start if isinstance(start, datetime) else datetime.combine(start, time.min)
    )
    end_dt = end if end is not None else start_dt + timedelta(days=1)
    start_dt = (
        start_dt.replace(tzinfo=timezone.utc)
        if start_dt.tzinfo is None
        else start_dt.astimezone(timezone.utc)
    )
    end_dt = (
        end_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None
        else end_dt.astimezone(timezone.utc)
    )
    files = []
    if end_dt > start_dt and (icaos is None or len(icaos)):
        buckets = query_icao_buckets(icaos)
        final_day = (end_dt - timedelta(microseconds=1)).date()
        days = list(date_range(start_dt.date(), final_day + timedelta(days=1)))
        # Validate the whole context before spending hours preparing earlier
        # days only to discover that the final raw partition is missing.
        for day in days:
            if not (partitioned_day_dir(STATE_CACHE_ROOT, day) / "_SUCCESS").is_file():
                complete_opensky_state_vector_files(day)
        for day in days:
            directory = ensure_reverse_eng_states(day)
            if buckets is None:
                files.extend(sorted(directory.glob("icao_bucket=*/*.parquet")))
            else:
                for bucket in buckets:
                    files.extend(
                        sorted((directory / f"icao_bucket={bucket}").glob("*.parquet"))
                    )
    scan = (
        pl.scan_parquet(files, hive_partitioning=False)
        if files
        else pl.DataFrame(schema=STATE_SCHEMA).lazy()
    )
    scan = scan.filter(
        (pl.col("time") >= start_dt.timestamp()) & (pl.col("time") < end_dt.timestamp())
    )
    if icaos is not None:
        scan = scan.filter(pl.col("icao").is_in(icaos))
        if len(icaos):
            # Explicit bounds let Parquet skip sorted row groups in each hour.
            scan = scan.filter(
                pl.col("icao").is_between(pl.lit(min(icaos)), pl.lit(max(icaos)))
            )
    if columns is not None:
        scan = scan.select(columns)
    return scan if lazy else scan.collect(engine="streaming")
