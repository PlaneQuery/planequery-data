"""Convert a legacy Spark OpenSky day to direct-Trino hourly parquet layout."""

import argparse
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.opensky.paths import (
    OPENSKY_RAW_STATE_VECTORS_ROOT,
    raw_state_vectors_hour_path,
)


BATCH_SIZE = 100_000
DEFAULT_SOURCE_DIR = Path(
    "/Volumes/T2-SSD/planequery/data/raw/opensky/trino-tables/2026-03-01"
)
LEGACY_STATE_VECTORS_RELATIVE_DIR = Path(
    "opensky_state_vectors/trino-tables/state_vectors"
)
STATE_VECTORS_SCHEMA = pa.schema(
    [
        ("time", pa.int64()),
        ("icao24", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("velocity", pa.float64()),
        ("heading", pa.float64()),
        ("vertrate", pa.float64()),
        ("callsign", pa.string()),
        ("onground", pa.bool_()),
        ("alert", pa.bool_()),
        ("spi", pa.bool_()),
        ("squawk", pa.string()),
        ("baroaltitude", pa.float64()),
        ("geoaltitude", pa.float64()),
        ("lastposupdate", pa.float64()),
        ("lastcontact", pa.float64()),
        ("serials", pa.list_(pa.int64())),
        ("hour", pa.int64()),
    ]
)


LEGACY_COLUMN_ALIASES = {
    "vertRate": "vertrate",
    "onGround": "onground",
    "baroAltitude": "baroaltitude",
    "geoAltitude": "geoaltitude",
    "lastPosUpdate": "lastposupdate",
    "lastContact": "lastcontact",
}


def _hour_from_legacy_path(path: Path) -> datetime:
    try:
        timestamp = int(path.parent.name.removeprefix("hour="))
    except ValueError as exc:
        raise ValueError(f"Invalid legacy hour partition: {path.parent}") from exc
    return datetime.fromtimestamp(timestamp, timezone.utc)


def _convert_table(table: pa.Table, hour_timestamp: int) -> pa.Table:
    table = table.rename_columns(
        [LEGACY_COLUMN_ALIASES.get(name, name) for name in table.column_names]
    )
    if "hour" not in table.column_names:
        table = table.append_column(
            "hour",
            pa.array([hour_timestamp] * len(table), type=pa.int64()),
        )
    missing = set(STATE_VECTORS_SCHEMA.names) - set(table.column_names)
    if missing:
        raise ValueError(
            "Legacy state-vector data is missing columns: "
            + ", ".join(sorted(missing))
        )
    return table.select(STATE_VECTORS_SCHEMA.names).cast(STATE_VECTORS_SCHEMA)


def convert_hour(
    source_files: Sequence[Path],
    start: datetime,
    *,
    output_root: Path = OPENSKY_RAW_STATE_VECTORS_ROOT,
    batch_size: int = BATCH_SIZE,
) -> tuple[Path, int]:
    """Stream one legacy hour into one atomically published parquet file."""
    if not source_files:
        raise ValueError(f"No source files supplied for {start}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_path = raw_state_vectors_hour_path(start, root=output_root)
    if output_path.exists():
        rows = pq.ParquetFile(output_path).metadata.num_rows
        print(f"Skipping {start}: {output_path} already contains {rows:,} rows")
        return output_path, rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    hour_timestamp = int(start.timestamp())

    try:
        writer = pq.ParquetWriter(
            temp_path,
            STATE_VECTORS_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        for source_path in source_files:
            parquet_file = pq.ParquetFile(source_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = _convert_table(pa.Table.from_batches([batch]), hour_timestamp)
                writer.write_table(table)
                total_rows += len(table)
                print(f"\r  {start:%H}:00: {total_rows:,} rows", end="", flush=True)
        writer.close()
        writer = None
        temp_path.replace(output_path)
        print(f"\nSaved {total_rows:,} rows -> {output_path}")
        return output_path, total_rows
    except BaseException:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
        raise


def convert_day(
    source_dir: Path,
    target_date: date,
    *,
    output_root: Path = OPENSKY_RAW_STATE_VECTORS_ROOT,
    batch_size: int = BATCH_SIZE,
) -> list[Path]:
    """Convert all 24 legacy Unix-hour partitions for one UTC day."""
    nested_source_dir = source_dir / LEGACY_STATE_VECTORS_RELATIVE_DIR
    if nested_source_dir.is_dir():
        source_dir = nested_source_dir

    files_by_hour: dict[datetime, list[Path]] = {}
    for source_path in sorted(source_dir.glob("hour=*/*.parquet")):
        start = _hour_from_legacy_path(source_path)
        files_by_hour.setdefault(start, []).append(source_path)

    expected_hours = {
        datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour,
            tzinfo=timezone.utc,
        )
        for hour in range(24)
    }
    found_hours = set(files_by_hour)
    if found_hours != expected_hours:
        missing = sorted(expected_hours - found_hours)
        unexpected = sorted(found_hours - expected_hours)
        raise FileNotFoundError(
            f"Expected exactly 24 partitions for {target_date}; "
            f"missing={[value.isoformat() for value in missing]}, "
            f"unexpected={[value.isoformat() for value in unexpected]}"
        )

    outputs = []
    for start in sorted(expected_hours):
        output_path, _ = convert_hour(
            files_by_hour[start],
            start,
            output_root=output_root,
            batch_size=batch_size,
        )
        outputs.append(output_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=date.fromisoformat)
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_SOURCE_DIR,
        help=f"legacy day or state-vectors directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OPENSKY_RAW_STATE_VECTORS_ROOT,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    convert_day(
        args.source_dir,
        args.date,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
