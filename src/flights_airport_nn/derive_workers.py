"""Fill disjoint airport-NN day parts in parallel with a running derivation."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import polars as pl

from data_engineering.flights.flight_type import flights_to_flights_df, get_flights
from flights_airport_nn.derive_day import (
    DEFAULT_MODEL_PATH,
    derived_path,
    source_signature,
)
from flights_airport_nn.inference import load_model, run_inference


def fill_range(day, source, model_path, batch_aircraft, first, stop):
    flights = get_flights(day, no_airports_model=True, adsb_src=source)
    icaos = sorted(flights["icao"].unique().to_list())
    spec = source_signature(day, source, model_path, batch_aircraft)
    run_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    part_dir = derived_path(day, source, model_path).parent / "_parts" / run_hash
    part_dir.mkdir(parents=True, exist_ok=True)
    model = config = None
    for index in range(first, min(stop, (len(icaos) + batch_aircraft - 1) // batch_aircraft)):
        part = part_dir / f"part-{index:04d}.parquet"
        if part.exists():
            print(f"Worker {first}-{stop}: part {index} already complete", flush=True)
            continue
        if model is None:
            model, config = load_model(model_path)
        block = icaos[index * batch_aircraft : (index + 1) * batch_aircraft]
        subset = flights.filter(pl.col("icao").is_in(block))
        predicted = run_inference(
            day,
            model,
            config,
            aircraft_batch_size=batch_aircraft,
            adsb_src=source,
            flights=subset,
        )
        if len(predicted) != subset.height:
            raise ValueError(f"Incomplete airport NN batch {index}")
        frame = flights_to_flights_df(list(predicted.values()))
        temp = part.with_suffix(f".{os.getpid()}.tmp")
        frame.write_parquet(temp)
        temp.replace(part)
        print(f"Worker {first}-{stop}: part {index} complete", flush=True)
    return (first, stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", default="adsblol")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-aircraft", type=int, default=512)
    parser.add_argument("--ranges", nargs="+", required=True)
    args = parser.parse_args()
    ranges = [tuple(map(int, value.split(":"))) for value in args.ranges]
    with ProcessPoolExecutor(
        max_workers=len(ranges), mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                fill_range,
                args.date,
                args.source,
                args.model,
                args.batch_aircraft,
                first,
                stop,
            )
            for first, stop in ranges
        ]
        for future in futures:
            print(f"Completed range {future.result()}", flush=True)
