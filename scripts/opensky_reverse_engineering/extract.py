"""Bounded-memory evidence extraction from local OpenSky parquet files.

This does not modify raw data or contact Trino. The flight table is used only
to choose the study population and endpoint probes, never to infer segments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path('/mnt/storage2/planequery-data/data/raw/opensky/trino-downloads')
OUT = Path('/mnt/storage2/planequery-data/data/processed/opensky_reverse_engineering/2024-06-01')
COLUMNS = ['icao24', 'time', 'lat', 'lon', 'baroaltitude', 'geoaltitude',
           'velocity', 'heading', 'vertrate', 'onground', 'lastposupdate',
           'lastcontact', 'callsign']


def read_hour(path: Path) -> pa.BufferReader:
    """Read bounded ranges concurrently, without a second full-file copy."""
    data = bytearray(path.stat().st_size)
    view = memoryview(data)
    fd = os.open(path, os.O_RDONLY)
    def read_chunk(start):
        end = min(start + 8*1024*1024, len(data))
        while start < end:
            count = os.preadv(fd, [view[start:end]], start)
            if count == 0:
                raise EOFError(f'Unexpected end of {path}')
            start += count
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(read_chunk, range(0,len(data),8*1024*1024)))
    finally:
        os.close(fd)
    return pa.BufferReader(data)


def extract(raw: Path, out: Path, sample_size: int = 512, limit: int | None = None, only_hours: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    flights = pl.read_parquet(raw / 'flights_data4/year=2024/month=06/day=01/data.parquet').with_row_index('flight_id')
    # Hash order gives a reproducible aircraft sample without depending on file order.
    aircraft = sorted(flights['icao24'].unique(), key=lambda x: hashlib.sha256(('opensky-20240601:'+x).encode()).hexdigest())
    selected = aircraft[:sample_size]
    manifest = {'sample_size': sample_size, 'sample_aircraft': selected,
                'raw_root': str(raw), 'probe_offsets': list(range(-2, 4))}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Cache manifest differs; use a separate output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    flights.write_parquet(out / 'reference_flights.parquet')
    endpoints = pl.concat([
        flights.select('flight_id', 'icao24', pl.col(col).alias('endpoint_time')).with_columns(pl.lit(kind).alias('kind'))
        for col, kind in [('firstseen', 'departure'), ('lastseen', 'arrival')]
    ]).drop_nulls('endpoint_time')
    probes = endpoints.join(pl.DataFrame({'offset': list(range(-2, 4))}), how='cross').with_columns(
        (pl.col('endpoint_time') + pl.col('offset')).alias('time'))
    files = sorted(p for day in ['month=05/day=31','month=06/day=01','month=06/day=02']
                   for p in (raw/'state_vectors_data4/year=2024'/day).glob('hour=*/data.parquet'))
    if len(files) != 72:
        raise ValueError(f'Expected 72 input hours for May 31–June 2, found {len(files)}')
    files.sort(key=lambda p: (0 if 'month=06/day=01' in str(p) else 1 if 'month=05/day=31' in str(p) else 2, str(p)))
    for number, path in enumerate(files[:limit] if limit else files):
        label = '-'.join(part.split('=')[1] for part in path.parts[-5:-1])
        if only_hours and label not in only_hours:
            continue
        sample_path = out / 'sample_hours' / f'{label}.parquet'
        endpoint_path = out / 'endpoint_hours' / f'{label}.parquet'
        if sample_path.exists() and endpoint_path.exists():
            print(f'Skip {label}', flush=True)
            continue
        t0 = time.monotonic()
        # Read each compressed hour sequentially: thousands of projected
        # column reads cause excessive seeks on the source spinning disk.
        buffer = read_hour(path)
        pf = pq.ParquetFile(buffer)
        # Partition paths are UTC. Min/max metadata are not required to exist.
        from datetime import datetime, timezone
        hour = int(datetime.strptime(label, '%Y-%m-%d-%H').replace(tzinfo=timezone.utc).timestamp())
        hp = probes.filter(pl.col('time').is_between(hour, hour+3599))
        samples, hits = [], []
        for batch in pf.iter_batches(batch_size=500_000, columns=COLUMNS):
            d = pl.from_arrow(batch)
            samples.append(d.filter(pl.col('icao24').is_in(selected)))
            if hp.height:
                hits.append(d.join(hp, on=['icao24', 'time'], how='inner'))
        sample_path.parent.mkdir(exist_ok=True)
        endpoint_path.parent.mkdir(exist_ok=True)
        sample = pl.concat(samples)
        hit = pl.concat(hits) if hits else pl.DataFrame(schema={**sample.schema, **hp.drop('icao24', 'time').schema})
        for data, dest in [(sample, sample_path), (hit, endpoint_path)]:
            temp = dest.with_suffix('.tmp')
            data.write_parquet(temp, compression='zstd')
            temp.replace(dest)
        print(f'{number+1}/{len(files)} {label}: {pf.metadata.num_rows:,} rows; sample {sample.height:,}; probes {hit.height:,}; {time.monotonic()-t0:.1f}s', flush=True)
        del samples, hits, sample, hit, d, batch, pf, buffer, data
        gc.collect()
        pa.default_memory_pool().release_unused()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sample-size', type=int, default=512)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only-hours', nargs='+')
    args = parser.parse_args()
    extract(args.raw, args.out, args.sample_size, args.limit, args.only_hours)
