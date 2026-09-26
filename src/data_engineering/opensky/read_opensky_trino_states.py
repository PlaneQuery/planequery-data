import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import polars as pl
import pyarrow as pa

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA
from data_engineering.adsb.partitions import (
    ICAO_BUCKET_COUNT,
    ICAO_BUCKET_SIZE,
    query_icao_buckets,
)
from data_engineering.opensky.paths import (
    OPENSKY_PROCESSED_STATE_VECTORS_ROOT,
    OPENSKY_RAW_STATE_VECTORS_ROOT,
    processed_state_vectors_day_dir,
    raw_state_vectors_day_dir,
)


DEFAULT_OPENSKY_TRINO_ROOT = OPENSKY_RAW_STATE_VECTORS_ROOT
DEFAULT_OPENSKY_PROCESSED_ROOT = OPENSKY_PROCESSED_STATE_VECTORS_ROOT

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


def opensky_processed_state_vectors_path(
    current_day: date | datetime | str,
) -> Path:
    """Return the generated, ICAO-bucketed cache directory for one day."""
    return processed_state_vectors_day_dir(current_day)


def scan_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
    *,
    require_complete: bool = True,
) -> pl.LazyFrame:
    parquet_files = opensky_state_vector_files(current_day)
    if not parquet_files:
        raise FileNotFoundError(
            f"No OpenSky state vector parquet files found under "
            f"{opensky_state_vectors_dir(current_day)}"
        )
    if require_complete:
        found_hours = {path.parent.name for path in parquet_files}
        expected_hours = {f"hour={hour:02d}" for hour in range(24)}
        missing_hours = sorted(expected_hours - found_hours)
        if missing_hours:
            raise FileNotFoundError(
                f"OpenSky raw day is incomplete under "
                f"{opensky_state_vectors_dir(current_day)}; missing "
                f"{', '.join(missing_hours)}"
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
ADSB_PARQUET_COLUMN_NAMES = set(PARQUET_SCHEMA.names)


def _polars_dtype_from_arrow_type(arrow_type: pa.DataType) -> pl.DataType:
    if pa.types.is_timestamp(arrow_type):
        return pl.Datetime(arrow_type.unit, arrow_type.tz)
    if pa.types.is_string(arrow_type):
        return pl.Utf8
    if pa.types.is_boolean(arrow_type):
        return pl.Boolean
    if pa.types.is_int32(arrow_type):
        return pl.Int32
    if pa.types.is_int64(arrow_type):
        return pl.Int64
    if pa.types.is_float32(arrow_type):
        return pl.Float32
    if pa.types.is_float64(arrow_type):
        return pl.Float64
    if pa.types.is_list(arrow_type) and pa.types.is_string(arrow_type.value_type):
        return pl.List(pl.Utf8)
    raise TypeError(f"Unsupported ADS-B parquet schema type: {arrow_type}")


def align_to_adsb_parquet_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """Validate and cast existing columns to the canonical ADS-B parquet schema."""
    columns = df.collect_schema().names()
    unknown_columns = sorted(set(columns) - ADSB_PARQUET_COLUMN_NAMES)
    if unknown_columns:
        raise ValueError(
            "OpenSky ADS-B columns are not in PARQUET_SCHEMA: "
            f"{', '.join(unknown_columns)}"
        )

    schema_fields = [field for field in PARQUET_SCHEMA if field.name in columns]
    df = df.select(
        [
            pl.col(field.name)
            .cast(_polars_dtype_from_arrow_type(field.type))
            .alias(field.name)
            for field in schema_fields
        ]
    )
    df = df.filter(pl.col("lat").is_not_null())
    df = df.filter(pl.col("lon").is_not_null())
    return df


def normalize_opensky_state_vectors(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert fresh OpenSky state vectors to canonical ADS-B-like rows.

    OpenSky's ``time`` is the state-vector snapshot time. Position values may
    have been carried forward from an older update, whose actual timestamp is
    ``lastPosUpdate``. Stale snapshots are removed, repeated snapshots of the
    same position update are collapsed, and the output is timestamped with the
    actual position-update time.
    """
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


def scan_normalized_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
) -> pl.LazyFrame:
    """Scan and normalize one day of raw OpenSky state-vector parquet."""
    return normalize_opensky_state_vectors(
        scan_opensky_state_vectors_for_day(
            current_day,
            DEFAULT_OPENSKY_STATE_VECTOR_COLUMNS,
        )
    )


def write_opensky_state_vectors_cache_for_day(
    current_day: date | datetime | str,
    overwrite: bool = False,
) -> Path:
    """Build a canonical, ICAO-bucketed cache from the 24 raw hour files."""
    output_dir = opensky_processed_state_vectors_path(current_day)
    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and not overwrite:
        return output_dir

    df = scan_normalized_opensky_state_vectors_for_day(current_day)
    df = align_to_adsb_parquet_schema(df)
    df = df.with_columns(
        (
            pl.col("icao")
            .str.to_integer(base=16, strict=False)
            .floordiv(ICAO_BUCKET_SIZE)
            .clip(0, ICAO_BUCKET_COUNT - 1)
            .fill_null(ICAO_BUCKET_COUNT - 1)
            .cast(pl.Int8)
        ).alias("icao_bucket")
    )

    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.sink_parquet(
            pl.PartitionBy(
                temp_dir,
                key="icao_bucket",
                include_key=False,
            ),
            compression="zstd",
            maintain_order=False,
            mkdir=True,
        )
        (temp_dir / "_SUCCESS").touch()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return output_dir


def _empty_normalized_scan(columns: Sequence[str]) -> pl.LazyFrame:
    fields = {field.name: field for field in PARQUET_SCHEMA}
    return pl.DataFrame(
        schema={
            column: (
                _polars_dtype_from_arrow_type(fields[column].type)
                if column in fields
                else pl.Null
            )
            for column in columns
        }
    ).lazy()


def _cached_opensky_files(
    current_day: date | datetime | str,
    icaos: Sequence[str] | None,
) -> list[Path]:
    output_dir = opensky_processed_state_vectors_path(current_day)
    buckets = query_icao_buckets(icaos)
    if buckets is None:
        return sorted(output_dir.glob("icao_bucket=*/*.parquet"))
    return sorted(
        path
        for bucket in buckets
        for path in (output_dir / f"icao_bucket={bucket}").glob("*.parquet")
    )


def scan_cached_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
    overwrite: bool = False,
    *,
    icaos: Sequence[str] | None = None,
) -> pl.LazyFrame:
    write_opensky_state_vectors_cache_for_day(
        current_day=current_day,
        overwrite=overwrite,
    )
    parquet_files = _cached_opensky_files(current_day, icaos)
    if not parquet_files:
        if columns is None:
            columns = [field.name for field in PARQUET_SCHEMA]
        return _empty_normalized_scan(columns)

    df = pl.scan_parquet(parquet_files, hive_partitioning=False)
    if icaos is not None:
        df = df.filter(pl.col("icao").is_in([icao.lower() for icao in icaos]))
    if columns:
        df = df.select(columns)
    return df


def read_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
    overwrite_cache: bool = False,
) -> pl.DataFrame:
    return scan_cached_opensky_state_vectors_for_day(
        current_day=current_day,
        columns=columns,
        overwrite=overwrite_cache,
    ).collect()


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


def get_icaos_in_opensky(current_day: date | datetime | str) -> list[str]:
    df = (
        scan_cached_opensky_state_vectors_for_day(current_day, columns=["icao"])
        .unique()
        .collect(engine="streaming")
    )
    return df.get_column("icao").drop_nulls().to_list()


def scan_opensky_messages_for_range(
    start_dt: datetime,
    end_dt: datetime,
    icaos: Sequence[str] | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Scan OpenSky state vectors as ADS-B-like messages for a single day.

    OpenSky state vector parquet is only downloaded one day at a time, so
    this reads the day of `start_dt` (unlike ADSB.lol/ADSBX, there is no
    surrounding-day context to look up).

    Joins in OpenAirframes metadata and renames columns to match the ADS-B
    message schema used elsewhere (see `data_engineering.adsb.read_adsb.read_adsb`),
    so OpenSky can be used as an `adsb_src` for the flights algorithm.
    """
    from data_engineering.openairframes.read import scan_latest_icao_info_for_join

    df = scan_cached_opensky_state_vectors_for_day(
        start_dt.date(),
        icaos=icaos,
    )
    df = df.filter(
        (pl.col("time") >= start_dt)
        & (pl.col("time") < end_dt)
        & pl.col("lat").is_not_null()
        & pl.col("lon").is_not_null()
    )
    df_openairframes = scan_latest_icao_info_for_join().with_columns(
        pl.col("icao").str.to_lowercase()
    )
    df = df.with_columns(pl.col("icao").str.to_lowercase())
    df = join_lf_with_openairframes(df, df_openairframes)

    if columns is not None:
        df = df.select(columns)
    return df


if __name__ == "__main__":
    df = read_opensky_state_vectors_for_day(
        "2026-03-01",
    )
    print(df)
