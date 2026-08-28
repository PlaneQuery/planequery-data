from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import polars as pl
import pyarrow as pa

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA
from data_engineering.utils import OUTPUT_DIR


DEFAULT_OPENSKY_TRINO_ROOT = OUTPUT_DIR / "data/raw/opensky/trino-tables"
DEFAULT_OPENSKY_PROCESSED_ROOT = OUTPUT_DIR / "data/processed/opensky/v2"

# A state-vector row is a snapshot, not necessarily a newly received position.
# OpenSky may repeat the last known state for minutes. Only retain snapshots
# whose aircraft contact and position update are both recent.
MAX_STATE_VECTOR_AGE_SECONDS = 5.0
MAX_FUTURE_TIMESTAMP_SKEW_SECONDS = 1.0


def _coerce_date(current_day: date | datetime | str) -> date:
    if isinstance(current_day, datetime):
        return current_day.date()
    if isinstance(current_day, date):
        return current_day
    return datetime.strptime(current_day, "%Y-%m-%d").date()


def opensky_state_vectors_dir(
    current_day: date | datetime | str,
) -> Path:
    current_day = _coerce_date(current_day)
    return (
        DEFAULT_OPENSKY_TRINO_ROOT
        / f"{current_day:%Y-%m-%d}"
        / "opensky_state_vectors"
        / "trino-tables"
        / "state_vectors"
    )


def opensky_state_vector_files(
    current_day: date | datetime | str,
) -> list[Path]:
    state_vectors_dir = opensky_state_vectors_dir(current_day)
    return sorted(state_vectors_dir.glob("hour=*/part-*.parquet"))


def opensky_processed_state_vectors_path(current_day: date | datetime | str) -> Path:
    current_day = _coerce_date(current_day)
    return (
        DEFAULT_OPENSKY_PROCESSED_ROOT
        / f"year={current_day:%Y}"
        / f"month={current_day:%m}"
        / f"day={current_day:%d}"
        / "data.parquet"
    )


def scan_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    parquet_files = opensky_state_vector_files(current_day)
    if not parquet_files:
        raise FileNotFoundError(
            f"No OpenSky state vector parquet files found under "
            f"{opensky_state_vectors_dir(current_day)}"
        )

    df = pl.scan_parquet(
        parquet_files,
        hive_partitioning=True,
    )
    if columns:
        df = df.select(columns)
    return df


MAP_DICT = {
    "icao24": "icao",
    "velocity": "ground_speed_kt", # m/s -> knots
    "heading": "track_deg",
    "onGround": "on_ground",
    "baroAltitude": "baro_altitude_ft", # meters -> ft
}
ADDITIONAL_COLUMNS = ["time", "lat", "lon", "callsign"]
FRESHNESS_COLUMNS = ["lastContact", "lastPosUpdate"]
DEFAULT_OPENSKY_STATE_VECTOR_COLUMNS = (
    list(MAP_DICT.keys()) + ADDITIONAL_COLUMNS + FRESHNESS_COLUMNS
)
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
    required_columns = {
        "time",
        "icao24",
        "lastContact",
        "lastPosUpdate",
    }
    missing_columns = required_columns.difference(df.collect_schema().names())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"OpenSky normalization requires freshness columns: {missing}"
        )

    contact_age = (
        pl.col("time").cast(pl.Float64) - pl.col("lastContact")
    )
    position_age = (
        pl.col("time").cast(pl.Float64) - pl.col("lastPosUpdate")
    )
    df = (
        df
        .filter(
            pl.col("lastContact").is_finite()
            & pl.col("lastPosUpdate").is_finite()
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
            subset=["icao24", "lastPosUpdate"],
            keep="any",
            maintain_order=False,
        )
    )

    return df.select(
        pl.from_epoch(pl.col("lastPosUpdate"), time_unit="s")
        .dt.replace_time_zone(None)
        .cast(pl.Datetime("ms"))
        .alias("time"),
        pl.col("icao24").str.to_lowercase().alias("icao"),
        (pl.col("velocity") * 1.94384).alias("ground_speed_kt"),
        pl.col("heading").alias("track_deg"),
        pl.col("onGround").alias("on_ground"),
        (pl.col("baroAltitude") * 3.28084).alias("baro_altitude_ft"),
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
    output_path = opensky_processed_state_vectors_path(current_day)
    if output_path.exists() and not overwrite:
        return output_path

    df = scan_normalized_opensky_state_vectors_for_day(current_day)
    df = align_to_adsb_parquet_schema(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sink_parquet(
        output_path,
        compression="zstd",
        maintain_order=False,
    )
    return output_path


def scan_cached_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = None,
    overwrite: bool = False,
) -> pl.LazyFrame:
    output_path = write_opensky_state_vectors_cache_for_day(
        current_day=current_day,
        overwrite=overwrite,
    )
    df = pl.scan_parquet(output_path)
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
    df = scan_opensky_state_vectors_for_day(current_day, columns=["icao24"]).collect()
    return df.get_column("icao24").drop_nulls().unique().to_list()


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

    df = scan_cached_opensky_state_vectors_for_day(start_dt.date())
    df = df.filter(
        (pl.col("time") >= start_dt)
        & (pl.col("time") < end_dt)
        & pl.col("lat").is_not_null()
        & pl.col("lon").is_not_null()
    )
    if icaos is not None:
        df = df.filter(pl.col("icao").is_in(list(icaos)))

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
