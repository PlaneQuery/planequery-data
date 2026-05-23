from datetime import date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA


DEFAULT_OPENSKY_TRINO_ROOT = Path(
    "/Volumes/T2-SSD/planequery/data/raw/opensky/trino-tables"
)


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


def join():
    pass


MAP_DICT = {
    "icao24": "icao",
    "velocity": "ground_speed_kt", # m/s -> knots
    "heading": "track_deg",
    "onGround": "on_ground",
    "baroAltitude": "baro_altitude_ft", # meters -> ft
}
ADDITIONAL_COLUMNS = ["time", "lat", "lon", "callsign"]
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
    columns = df.collect_schema().names()
    if "time" in columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("time"), time_unit="s")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("ms"))
            .alias("time")
        )
    if "baroAltitude" in columns:
        df = df.with_columns(
            (pl.col("baroAltitude") * 3.28084).alias("baro_altitude_ft")
        )
        df = df.drop("baroAltitude")
    if "velocity" in columns:
        df = df.with_columns(
            (pl.col("velocity") * 1.94384).alias("ground_speed_kt")
        )
        df = df.drop("velocity")
    map_dict = {k: v for k, v in MAP_DICT.items() if k not in ["velocity", "baroAltitude", "time"]}
    for old_col, new_col in map_dict.items():
        if old_col in columns:
            df = df.rename({old_col: new_col})
    return df


def scan_normalized_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = list(MAP_DICT.keys()) + ADDITIONAL_COLUMNS,
) -> pl.LazyFrame:
    df = scan_opensky_state_vectors_for_day(
        current_day=current_day,
        columns=columns,
    )
    return normalize_opensky_state_vectors(df)


def read_opensky_state_vectors_for_day(
    current_day: date | datetime | str,
    columns: list[str] | None = list(MAP_DICT.keys()) + ADDITIONAL_COLUMNS,
) -> pl.DataFrame:
    return scan_normalized_opensky_state_vectors_for_day(
        current_day=current_day,
        columns=columns,
    ).collect()


def join_df_with_openairframes(df: pl.DataFrame, df_openairframes: pl.DataFrame) -> pl.DataFrame:
    df = (
        df
        .join(
            df_openairframes.select(["icao", *OPENAIRFRAMES_COLUMNS]),
            on="icao",
            how="left",
        )
    )
    return df


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


if __name__ == "__main__":
    df = read_opensky_state_vectors_for_day("2026-03-01", columns = ["time", "icao24", "lat", "lon", "baroAltitude"])
    print(df)
