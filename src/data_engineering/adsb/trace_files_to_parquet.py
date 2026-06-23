import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA
from data_engineering.adsb.process_trace_file import process_file
from data_engineering.utils import OUTPUT_DIR


ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" / "adsblol"
DEFAULT_OUTPUT_ROOT = ADSBLOL_DIR / "parquet_output" / "v6"
ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT
DEFAULT_MAX_WORKERS = min(os.cpu_count() or 1, 32)
PROGRESS_INTERVAL_FILES = 1_000


def _date_partition_value(path_part: str, partition_name: str) -> str:
    prefix = f"{partition_name}="
    if path_part.startswith(prefix):
        return path_part[len(prefix):]
    return path_part


def _parse_date_from_trace_folder(folder_path: Path) -> tuple[str, str, str]:
    day = _date_partition_value(folder_path.name, "day")
    month = _date_partition_value(folder_path.parent.name, "month")
    year = _date_partition_value(folder_path.parent.parent.name, "year")
    if not (len(year) == 4 and len(month) == 2 and len(day) == 2):
        raise ValueError(f"Could not parse YYYY/MM/DD from {folder_path}")
    return year, month, day


def _icao_bucket(icao: str) -> int:
    if icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    icao_int = int(icao.lower(), 16)
    return min(icao_int // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def _icao_from_trace_file(path: Path) -> str:
    filename = path.name
    if filename.endswith(".json"):
        filename = filename[:-len(".json")]
    if not filename.startswith("trace_full_"):
        raise ValueError(f"Unexpected trace filename: {path}")
    return filename[len("trace_full_"):]


def _trace_file_sort_key(path: Path) -> tuple[int, str, str]:
    icao = _icao_from_trace_file(path).lower()
    return _icao_bucket(icao), icao, str(path)


def _empty_cols() -> dict[str, list]:
    return {col: [] for col in COLUMNS}


def _sort_cols_by_icao_time(cols: dict[str, list]) -> dict[str, list]:
    row_count = len(cols["time"])
    if row_count < 2:
        return cols
    row_indices = sorted(
        range(row_count),
        key=lambda row_idx: (cols["icao"][row_idx], cols["time"][row_idx]),
    )
    return {
        col: [cols[col][row_idx] for row_idx in row_indices]
        for col in COLUMNS
    }


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
        folder_path.rglob("trace_full_*.json"),
        key=_trace_file_sort_key,
    )


def _process_file_for_parquet(file: Path, pia_or_american_ladd_only: bool):
    return process_file(
        file,
        log_dedupes=False,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )


def _processed_trace_files(
    files: list[Path],
    max_workers: int,
    submit_chunk_size: int,
    pia_or_american_ladd_only: bool,
):
    if max_workers == 1:
        for file in files:
            yield file, _process_file_for_parquet(file, pia_or_american_ladd_only)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk_start in range(0, len(files), submit_chunk_size):
            chunk = files[chunk_start:chunk_start + submit_chunk_size]
            futures = [
                executor.submit(
                    _process_file_for_parquet,
                    file,
                    pia_or_american_ladd_only,
                )
                for file in chunk
            ]
            for file, future in zip(chunk, futures):
                yield file, future.result()


def process_trace_files(
    folder_path: str | Path,
    output_root=DEFAULT_OUTPUT_ROOT,
    batch_size: int = 100_000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    submit_chunk_size: int | None = None,
    pia_or_american_ladd_only: bool = False,
) -> int:
    start_time = time.perf_counter()
    if submit_chunk_size is None:
        submit_chunk_size = max_workers * 8
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
    print(f"Processing with {max_workers} workers and {submit_chunk_size} submitted files per chunk")

    batch_cols_by_bucket = {bucket: _empty_cols() for bucket in range(ICAO_BUCKET_COUNT)}
    batch_count_by_bucket = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    total_rows_by_bucket = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    writers: dict[int, pq.ParquetWriter | None] = {
        bucket: None for bucket in range(ICAO_BUCKET_COUNT)
    }

    current_bucket = None
    processed_file_count = 0
    for file, cols in _processed_trace_files(
        files,
        max_workers,
        submit_chunk_size,
        pia_or_american_ladd_only,
    ):
        processed_file_count += 1
        if processed_file_count % PROGRESS_INTERVAL_FILES == 0:
            print(f"Processed {processed_file_count}/{len(files)} trace files")
        if cols is None:
            if not pia_or_american_ladd_only:
                print(f"No valid data in {file}")
            continue

        cols = _sort_cols_by_icao_time(cols)
        bucket = _icao_bucket(cols["icao"][0])
        if current_bucket is not None and bucket != current_bucket and batch_count_by_bucket[current_bucket]:
            writers[current_bucket] = _write_cols(
                writers[current_bucket],
                parquet_paths[current_bucket],
                batch_cols_by_bucket[current_bucket],
            )
            print(
                f"Wrote bucket {current_bucket} final sorted batch: "
                f"{batch_count_by_bucket[current_bucket]} rows "
                f"(total: {total_rows_by_bucket[current_bucket]})"
            )
            batch_cols_by_bucket[current_bucket] = _empty_cols()
            batch_count_by_bucket[current_bucket] = 0
        current_bucket = bucket

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

    for writer in writers.values():
        if writer is not None:
            writer.close()

    total_rows = sum(total_rows_by_bucket.values())
    print(f"Total rows written: {total_rows}")
    for bucket, row_count in total_rows_by_bucket.items():
        print(f"Bucket {bucket}: {row_count} rows -> {parquet_paths[bucket]}")
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    return total_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert adsb.lol trace files to bucketed parquet")
    parser.add_argument("folder_path", help="Trace folder, e.g. .../extract/year=2026/month=03/day=01")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--pia-or-american-ladd-only", action="store_true")
    args = parser.parse_args()

    process_trace_files(
        folder_path=args.folder_path,
        output_root=args.output_root,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        pia_or_american_ladd_only=args.pia_or_american_ladd_only,
    )
