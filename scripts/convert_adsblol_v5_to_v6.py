"""Convert one legacy adsb.lol v5 Parquet file to the bucketed v6 layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA


ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT


def icao_bucket(icao: str | None) -> int:
    """Return the canonical v6 bucket for an ICAO identifier."""
    if not icao or icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    return min(int(icao, 16) // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def v6_table(table: pa.Table) -> pa.Table:
    """Discard v5-only data and cast values to the canonical v6 schema."""
    arrays = []
    for field in PARQUET_SCHEMA:
        if field.name == "aircraft-year":
            arrays.append(pa.nulls(table.num_rows, type=field.type))
        else:
            arrays.append(pc.cast(table[field.name], field.type, safe=False))
    return pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)


def convert(source: Path, destination_day: Path, batch_size: int) -> dict[int, int]:
    outputs = {
        bucket: destination_day / f"icao_bucket={bucket}" / "data.parquet"
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    row_counts = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    source_file = pq.ParquetFile(source)
    try:
        for batch in source_file.iter_batches(batch_size=batch_size, columns=[
            column for column in COLUMNS if column != "aircraft-year"
        ]):
            table = v6_table(pa.Table.from_batches([batch]))
            buckets = [icao_bucket(icao) for icao in table["icao"].to_pylist()]
            for bucket in range(ICAO_BUCKET_COUNT):
                mask = pa.array([value == bucket for value in buckets])
                bucket_table = table.filter(mask)
                if not bucket_table.num_rows:
                    continue
                if bucket not in writers:
                    writers[bucket] = pq.ParquetWriter(
                        outputs[bucket], PARQUET_SCHEMA, compression="zstd"
                    )
                writers[bucket].write_table(bucket_table)
                row_counts[bucket] += bucket_table.num_rows
    finally:
        for writer in writers.values():
            writer.close()

    return row_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination_day", type=Path)
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()

    counts = convert(args.source, args.destination_day, args.batch_size)
    print(f"Total rows: {sum(counts.values())}")
    for bucket, count in counts.items():
        print(f"Bucket {bucket}: {count}")
