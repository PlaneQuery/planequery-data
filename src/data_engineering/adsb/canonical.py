"""Canonical Parquet layout shared by every ADS-B provider."""

from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA
from data_engineering.adsb.partitions import query_icao_buckets
from data_engineering.utils import OUTPUT_DIR


class ADSBSource(StrEnum):
    ADSBLOL = "adsblol"
    ADSBX = "adsbx"
    OPENSKY = "opensky"


ADSB_PARQUET_ROOT = OUTPUT_DIR / "data" / "processed" / "adsb" / "parquet_output"
ADSB_PARQUET_VERSIONS = {
    ADSBSource.ADSBLOL: 6,
    ADSBSource.ADSBX: 2,
    ADSBSource.OPENSKY: 4,
}
SUCCESS_MARKER = "_SUCCESS"


def coerce_adsb_source(source: ADSBSource) -> ADSBSource:
    if not isinstance(source, ADSBSource):
        raise TypeError(
            f"source must be an ADSBSource, got {type(source).__name__}"
        )
    return source


def coerce_day(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def adsb_parquet_root(source: ADSBSource) -> Path:
    source = coerce_adsb_source(source)
    return (
        ADSB_PARQUET_ROOT
        / f"source={source.value}"
        / f"version={ADSB_PARQUET_VERSIONS[source]}"
    )


def adsb_day_dir(source: ADSBSource, value: date | datetime | str) -> Path:
    day = coerce_day(value)
    return (
        adsb_parquet_root(source)
        / f"year={day.year}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
    )


def adsb_day_is_complete(
    source: ADSBSource,
    value: date | datetime | str,
) -> bool:
    return (adsb_day_dir(source, value) / SUCCESS_MARKER).is_file()


def mark_adsb_day_complete(day_dir: Path) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / SUCCESS_MARKER).touch()


def adsb_day_files(
    source: ADSBSource,
    value: date | datetime | str,
    *,
    icaos: Sequence[str] | None = None,
) -> list[Path]:
    day_dir = adsb_day_dir(source, value)
    buckets = query_icao_buckets(icaos)
    if buckets is None:
        return sorted(day_dir.glob("icao_bucket=*/*.parquet"))
    return sorted(
        path
        for bucket in buckets
        for path in (day_dir / f"icao_bucket={bucket}").glob("*.parquet")
    )


def polars_dtype_from_arrow_type(arrow_type: pa.DataType) -> pl.DataType:
    if pa.types.is_timestamp(arrow_type):
        return pl.Datetime(arrow_type.unit, arrow_type.tz)
    if pa.types.is_string(arrow_type):
        return pl.String
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
        return pl.List(pl.String)
    raise TypeError(f"Unsupported ADS-B parquet schema type: {arrow_type}")


def canonicalize_adsb_frame(
    df: pl.LazyFrame,
    *,
    include_missing: bool,
) -> pl.LazyFrame:
    """Cast ADS-B columns and optionally insert the full canonical schema."""
    columns = set(df.collect_schema().names())
    canonical_columns = {field.name for field in PARQUET_SCHEMA}
    unknown_columns = sorted(columns - canonical_columns)
    if unknown_columns:
        raise ValueError(
            "ADS-B columns are not in PARQUET_SCHEMA: "
            + ", ".join(unknown_columns)
        )

    expressions = []
    for field in PARQUET_SCHEMA:
        dtype = polars_dtype_from_arrow_type(field.type)
        if field.name in columns:
            expressions.append(pl.col(field.name).cast(dtype))
        elif include_missing:
            expressions.append(pl.lit(None, dtype=dtype).alias(field.name))
    return df.select(expressions)


def write_canonical_adsb_parquet(
    df: pl.LazyFrame,
    output_path: Path,
    *,
    batch_size: int = 100_000,
) -> None:
    """Stream a canonical frame to Parquet with the exact Arrow schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        output_path,
        PARQUET_SCHEMA,
        compression="zstd",
    ) as writer:
        for batch in df.collect_batches(
            chunk_size=batch_size,
            maintain_order=False,
            engine="streaming",
        ):
            writer.write_table(batch.to_arrow().cast(PARQUET_SCHEMA))


def empty_adsb_scan(columns: Sequence[str]) -> pl.LazyFrame:
    fields = {field.name: field for field in PARQUET_SCHEMA}
    return pl.DataFrame(
        schema={
            column: (
                polars_dtype_from_arrow_type(fields[column].type)
                if column in fields
                else pl.Null
            )
            for column in columns
        }
    ).lazy()


def scan_adsb_parquet(
    source: ADSBSource,
    days: Sequence[date],
    columns: Sequence[str],
    *,
    icaos: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan canonical provider Parquet with date and ICAO partition pruning."""
    source = coerce_adsb_source(source)
    paths = [
        path
        for day in days
        for path in adsb_day_files(source, day, icaos=icaos)
    ]
    if not paths:
        return empty_adsb_scan(columns)

    scan_columns = list(dict.fromkeys([*columns, "icao"]))
    lf = pl.scan_parquet(
        paths,
        hive_partitioning=False,
        extra_columns="ignore",
        missing_columns="insert",
    ).select(scan_columns)
    if icaos is not None:
        requested_icaos = [icao.lower() for icao in icaos]
        lf = lf.filter(pl.col("icao").str.to_lowercase().is_in(requested_icaos))
    return lf.select(columns)
