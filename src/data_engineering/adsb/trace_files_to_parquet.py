from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA
from data_engineering.adsb.process_trace_file import process_file


DEFAULT_TRACE_FOLDER = (
    "/Volumes/T2-SSD/planequery/data/raw/adsb-exchange/traces/2026-03-01/"
    "adsbx-trace-download/traces/2026/03/01"
)
DEFAULT_OUTPUT_ROOT = "/Volumes/T2-SSD/planequery/data/raw/adsb-exchange/traces_parquet"
ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT
DATA_SOURCE = "adsb-exchange"


def _parse_date_from_trace_folder(folder_path: Path) -> tuple[str, str, str]:
    day = folder_path.name
    month = folder_path.parent.name
    year = folder_path.parent.parent.name
    if not (len(year) == 4 and len(month) == 2 and len(day) == 2):
        raise ValueError(f"Could not parse YYYY/MM/DD from {folder_path}")
    return year, month, day


def _icao_bucket(icao: str) -> int:
    if icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    icao_int = int(icao.lower(), 16)
    return min(icao_int // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def _empty_cols() -> dict[str, list]:
    return {col: [] for col in COLUMNS}


def _extend_cols(batch_cols: dict[str, list], cols: dict[str, list]) -> int:
    row_count = len(cols["time"])
    for col in COLUMNS:
        batch_cols[col].extend(cols[col])
    return row_count


def _write_cols(writer: pq.ParquetWriter | None, parquet_path: Path, cols: dict[str, list]) -> pq.ParquetWriter:
    arrays = [pa.array(cols[col], type=PARQUET_SCHEMA.field(col).type) for col in COLUMNS]
    table = pa.table(dict(zip(COLUMNS, arrays)), schema=PARQUET_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def _trace_files(folder_path: Path) -> list[Path]:
    return sorted(
        [*folder_path.rglob("*.json"), *folder_path.rglob("*.json.gz")],
        key=lambda path: str(path),
    )


def process_trace_files(
    folder_path=DEFAULT_TRACE_FOLDER,
    output_root=DEFAULT_OUTPUT_ROOT,
    batch_size: int = 100_000,
) -> int:
    folder_path = Path(folder_path)
    output_root = Path(output_root)
    year, month, day = _parse_date_from_trace_folder(folder_path)

    bucket_dirs = {
        bucket: output_root / f"year={year}" / f"month={month}" / f"day={day}" / f"icao_bucket={bucket}"
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    parquet_paths = {bucket: bucket_dirs[bucket] / "data.parquet" for bucket in range(ICAO_BUCKET_COUNT)}
    for bucket_dir in bucket_dirs.values():
        bucket_dir.mkdir(parents=True, exist_ok=True)

    files = _trace_files(folder_path)
    print(f"Found {len(files)} trace files")

    batch_cols_by_bucket = {bucket: _empty_cols() for bucket in range(ICAO_BUCKET_COUNT)}
    batch_count_by_bucket = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    total_rows_by_bucket = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    writers: dict[int, pq.ParquetWriter | None] = {
        bucket: None for bucket in range(ICAO_BUCKET_COUNT)
    }

    try:
        for file in files:
            print(f"Processing {file}")
            cols = process_file(file, data_source=DATA_SOURCE)
            if cols is None:
                print(f"No valid data in {file}")
                continue

            bucket = _icao_bucket(cols["icao"][0])
            row_count = _extend_cols(batch_cols_by_bucket[bucket], cols)
            batch_count_by_bucket[bucket] += row_count
            total_rows_by_bucket[bucket] += row_count

            if batch_count_by_bucket[bucket] >= batch_size:
                writers[bucket] = _write_cols(
                    writers[bucket],
                    parquet_paths[bucket],
                    batch_cols_by_bucket[bucket],
                )
                print(
                    f"Wrote bucket {bucket} batch: {batch_count_by_bucket[bucket]} rows "
                    f"(total: {total_rows_by_bucket[bucket]})"
                )
                batch_cols_by_bucket[bucket] = _empty_cols()
                batch_count_by_bucket[bucket] = 0

        for bucket in range(ICAO_BUCKET_COUNT):
            if batch_count_by_bucket[bucket]:
                writers[bucket] = _write_cols(
                    writers[bucket],
                    parquet_paths[bucket],
                    batch_cols_by_bucket[bucket],
                )
                print(
                    f"Wrote bucket {bucket} final batch: {batch_count_by_bucket[bucket]} rows "
                    f"(total: {total_rows_by_bucket[bucket]})"
                )
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    total_rows = sum(total_rows_by_bucket.values())
    print(f"Total rows written: {total_rows}")
    for bucket, row_count in total_rows_by_bucket.items():
        print(f"Bucket {bucket}: {row_count} rows -> {parquet_paths[bucket]}")
    return total_rows


if __name__ == "__main__":
    process_trace_files()
