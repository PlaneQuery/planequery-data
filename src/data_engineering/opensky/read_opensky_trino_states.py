import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from data_engineering.adsb.canonical import (
    ADSBSource,
    adsb_day_dir,
    canonicalize_adsb_frame,
    write_canonical_adsb_parquet,
)
from data_engineering.adsb.partitions import (
    ICAO_BUCKET_COUNT,
    ICAO_BUCKET_SIZE,
)
from data_engineering.opensky.paths import (
    OPENSKY_RAW_STATE_VECTORS_ROOT,
    coerce_date,
    raw_state_vectors_day_dir,
)


DEFAULT_OPENSKY_TRINO_ROOT = OPENSKY_RAW_STATE_VECTORS_ROOT
OPENSKY_PROCESSING_SHARD_COUNT = 256
OPENSKY_PROCESSING_SHARD_SIZE = 0x1000000 // OPENSKY_PROCESSING_SHARD_COUNT

# A state-vector row is a snapshot, not necessarily a newly received position.
# OpenSky may repeat the last known state for minutes. Only retain snapshots
# whose aircraft contact and position update are both recent.
MAX_STATE_VECTOR_AGE_SECONDS = 5.0
MAX_FUTURE_TIMESTAMP_SKEW_SECONDS = 1.0


def opensky_state_vectors_dir(
    current_day: date | datetime | str,
) -> Path:
    return raw_state_vectors_day_dir(current_day)


def opensky_state_vector_files(
    current_day: date | datetime | str,
) -> list[Path]:
    state_vectors_dir = opensky_state_vectors_dir(current_day)
    return sorted(state_vectors_dir.glob("hour=*/data.parquet"))


def _complete_opensky_state_vector_files(
    current_day: date | datetime | str,
) -> list[Path]:
    parquet_files = opensky_state_vector_files(current_day)
    if not parquet_files:
        raise FileNotFoundError(
            f"No OpenSky state vector parquet files found under "
            f"{opensky_state_vectors_dir(current_day)}"
        )

    files_by_hour = {path.parent.name: path for path in parquet_files}
    expected_hours = [f"hour={hour:02d}" for hour in range(24)]
    missing_hours = [hour for hour in expected_hours if hour not in files_by_hour]
    if missing_hours:
        raise FileNotFoundError(
            f"OpenSky raw day is incomplete under "
            f"{opensky_state_vectors_dir(current_day)}; missing "
            f"{', '.join(missing_hours)}"
        )
    return [files_by_hour[hour] for hour in expected_hours]


def opensky_processed_state_vectors_path(
    current_day: date | datetime | str,
) -> Path:
    """Return the canonical OpenSky Parquet directory for one day."""
    return adsb_day_dir(ADSBSource.OPENSKY, current_day)


def scan_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
    *,
    require_complete: bool = True,
) -> pl.LazyFrame:
    parquet_files = (
        _complete_opensky_state_vector_files(current_day)
        if require_complete
        else opensky_state_vector_files(current_day)
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No OpenSky state vector parquet files found under "
            f"{opensky_state_vectors_dir(current_day)}"
        )

    df = pl.scan_parquet(
        parquet_files,
        # The files already contain Trino's Unix-hour ``hour`` column. The
        # directory's hour=00 label is only filesystem organization.
        hive_partitioning=False,
    )
    if columns:
        df = df.select(columns)
    return df


MAP_DICT = {
    "icao24": "icao",
    "velocity": "ground_speed_kt", # m/s -> knots
    "heading": "track_deg",
    # OpenSky exposes one vertical-rate value without a barometric/geometric
    # discriminator. Use it as the canonical rate consumed by the airport NN.
    "vertrate": "baro_rate_fpm", # m/s -> ft/min
    "onground": "on_ground",
    "baroaltitude": "baro_altitude_ft", # meters -> ft
    "geoaltitude": "geom_altitude_ft", # meters -> ft
}
ADDITIONAL_COLUMNS = ["time", "lat", "lon", "callsign"]
FRESHNESS_COLUMNS = ["lastcontact", "lastposupdate"]
DEFAULT_OPENSKY_STATE_VECTOR_COLUMNS = (
    list(MAP_DICT.keys()) + ADDITIONAL_COLUMNS + FRESHNESS_COLUMNS
)
RAW_COLUMN_ALIASES = {
    "onGround": "onground",
    "baroAltitude": "baroaltitude",
    "geoAltitude": "geoaltitude",
    "lastContact": "lastcontact",
    "lastPosUpdate": "lastposupdate",
}
OPENAIRFRAMES_COLUMNS = [
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


def align_to_adsb_parquet_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the normalized OpenSky subset to canonical ADS-B types."""
    return canonicalize_adsb_frame(df, include_missing=False).filter(
        pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
    )


def _prepare_opensky_state_vectors(df: pl.LazyFrame) -> pl.LazyFrame:
    input_columns = set(df.collect_schema().names())
    aliases = {
        old: new
        for old, new in RAW_COLUMN_ALIASES.items()
        if old in input_columns and new not in input_columns
    }
    if aliases:
        df = df.rename(aliases)

    required_columns = {
        "time",
        "icao24",
        "lastcontact",
        "lastposupdate",
    }
    missing_columns = required_columns.difference(df.collect_schema().names())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"OpenSky normalization requires freshness columns: {missing}"
        )

    return df


def _fresh_opensky_state_vectors(df: pl.LazyFrame) -> pl.LazyFrame:
    df = _prepare_opensky_state_vectors(df)

    contact_age = (
        pl.col("time").cast(pl.Float64) - pl.col("lastcontact")
    )
    position_age = (
        pl.col("time").cast(pl.Float64) - pl.col("lastposupdate")
    )
    df = (
        df
        .filter(
            pl.col("lastcontact").is_finite()
            & pl.col("lastposupdate").is_finite()
            & contact_age.is_between(
                -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS,
                MAX_STATE_VECTOR_AGE_SECONDS,
                closed="both",
            )
            & position_age.is_between(
                -MAX_FUTURE_TIMESTAMP_SKEW_SECONDS,
                MAX_STATE_VECTOR_AGE_SECONDS,
                closed="both",
            )
        )
        .unique(
            subset=["icao24", "lastposupdate"],
            keep="any",
            maintain_order=False,
        )
    )

    return df


def _normalize_fresh_opensky_state_vectors(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    return df.select(
        pl.from_epoch(pl.col("lastposupdate"), time_unit="s")
        .dt.replace_time_zone(None)
        .cast(pl.Datetime("ms"))
        .alias("time"),
        pl.col("icao24").str.to_lowercase().alias("icao"),
        (pl.col("velocity") * 1.94384).alias("ground_speed_kt"),
        pl.col("heading").alias("track_deg"),
        (pl.col("vertrate") * 196.850394).alias("baro_rate_fpm"),
        pl.col("onground").alias("on_ground"),
        (pl.col("baroaltitude") * 3.28084).alias("baro_altitude_ft"),
        (pl.col("geoaltitude") * 3.28084).alias("geom_altitude_ft"),
        *[pl.col(column) for column in ADDITIONAL_COLUMNS if column != "time"],
    )


def normalize_opensky_state_vectors(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert fresh OpenSky state vectors to canonical ADS-B-like rows.

    OpenSky's ``time`` is the state-vector snapshot time. Position values may
    have been carried forward from an older update, whose actual timestamp is
    ``lastPosUpdate``. Stale snapshots are removed, repeated snapshots of the
    same position update are collapsed, and the output is timestamped with the
    actual position-update time.
    """
    return _normalize_fresh_opensky_state_vectors(
        _fresh_opensky_state_vectors(df)
    )


def write_opensky_state_vectors_cache_for_day(
    current_day: date | datetime | str,
    overwrite: bool = False,
) -> Path:
    """Build cleaned canonical OpenSky Parquet with bounded memory.

    The raw OpenSky day can contain hundreds of millions of rows. Keeping the
    normalization and deduplication query bounded to one hour prevents Polars
    from retaining a day-wide hash table. A state vector can only remain fresh
    for ``MAX_STATE_VECTOR_AGE_SECONDS``, so retaining that small tail of exact
    deduplication keys removes repeats across hour boundaries.

    Normalized rows are staged in fine-grained processing shards. Each shard
    contains complete aircraft traces, allowing source cleanup to be applied
    once without changing the canonical eight-bucket output layout.
    """
    output_dir = opensky_processed_state_vectors_path(current_day)
    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and not overwrite:
        return output_dir

    parquet_files = _complete_opensky_state_vector_files(current_day)
    target_day = coerce_date(current_day)
    day_start = datetime(
        target_day.year,
        target_day.month,
        target_day.day,
        tzinfo=timezone.utc,
    )

    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    staging_dir = temp_dir / "_normalized"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous_boundary = pl.DataFrame(
            schema={"icao24": pl.String, "lastposupdate": pl.Float64}
        )
        deduplication_columns = ["icao24", "lastposupdate"]

        for hour, parquet_file in enumerate(parquet_files):
            print(
                f"Building OpenSky cache for {target_day}: hour {hour:02d}/23",
                flush=True,
            )
            raw = pl.scan_parquet(
                parquet_file,
                hive_partitioning=False,
            ).select(DEFAULT_OPENSKY_STATE_VECTOR_COLUMNS)
            fresh = _fresh_opensky_state_vectors(raw)
            if not previous_boundary.is_empty():
                fresh = fresh.join(
                    previous_boundary.lazy(),
                    on=deduplication_columns,
                    how="anti",
                )

            next_hour_epoch = (
                day_start + timedelta(hours=hour + 1)
            ).timestamp()
            previous_boundary = (
                fresh.filter(
                    pl.col("lastposupdate")
                    >= next_hour_epoch - MAX_STATE_VECTOR_AGE_SECONDS
                )
                .select(deduplication_columns)
                .collect(engine="streaming")
            )

            normalized = _normalize_fresh_opensky_state_vectors(fresh)
            normalized = align_to_adsb_parquet_schema(normalized)
            normalized = normalized.with_columns(
                (
                    pl.col("icao")
                    .str.to_integer(base=16, strict=False)
                    .floordiv(OPENSKY_PROCESSING_SHARD_SIZE)
                    .clip(0, OPENSKY_PROCESSING_SHARD_COUNT - 1)
                    .fill_null(OPENSKY_PROCESSING_SHARD_COUNT - 1)
                    .cast(pl.UInt8)
                ).alias("processing_shard")
            )

            def hourly_output_path(args, current_hour=hour):
                shard = args.partition_keys.get_column("processing_shard").item()
                return (
                    staging_dir
                    / f"processing_shard={shard}"
                    / (
                        f"hour={current_hour:02d}-"
                        f"part={args.index_in_partition:05d}.parquet"
                    )
                )

            normalized.sink_parquet(
                pl.PartitionBy(
                    staging_dir,
                    key="processing_shard",
                    include_key=False,
                    file_path_provider=hourly_output_path,
                ),
                compression="zstd",
                maintain_order=False,
                mkdir=True,
            )

        from data_engineering.openairframes.read import (
            scan_latest_icao_info_for_join,
        )
        from data_engineering.opensky.preprocess import preprocess_opensky_adsb

        openairframes = (
            scan_latest_icao_info_for_join(at_date=target_day)
            .with_columns(pl.col("icao").str.to_lowercase())
            .collect(engine="streaming")
        )
        shard_dirs = sorted(staging_dir.glob("processing_shard=*"))
        for shard_number, shard_dir in enumerate(shard_dirs, start=1):
            processing_shard = int(shard_dir.name.partition("=")[2])
            print(
                f"Cleaning OpenSky cache for {target_day}: "
                f"shard {shard_number}/{len(shard_dirs)}",
                flush=True,
            )
            cleaned = preprocess_opensky_adsb(
                pl.scan_parquet(
                    sorted(shard_dir.glob("*.parquet")),
                    hive_partitioning=False,
                )
            )
            enriched = join_lf_with_openairframes(cleaned, openairframes)
            canonical = canonicalize_adsb_frame(enriched, include_missing=True)

            icao_bucket = min(
                processing_shard
                * OPENSKY_PROCESSING_SHARD_SIZE
                // ICAO_BUCKET_SIZE,
                ICAO_BUCKET_COUNT - 1,
            )
            write_canonical_adsb_parquet(
                canonical,
                temp_dir
                / f"icao_bucket={icao_bucket}"
                / f"part={processing_shard:03d}.parquet",
            )

        shutil.rmtree(staging_dir)
        (temp_dir / "_SUCCESS").touch()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return output_dir


def join_lf_with_openairframes(
    df: pl.LazyFrame,
    df_openairframes: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    if isinstance(df_openairframes, pl.DataFrame):
        df_openairframes = df_openairframes.lazy()
    return df.join(
        df_openairframes.select(["icao", *OPENAIRFRAMES_COLUMNS]),
        on="icao",
        how="left",
    )
