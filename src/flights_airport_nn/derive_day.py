"""Persist resumable airport-NN predictions for a full day of algorithm flights."""

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import (
    _adsbx_parquet_day_dir,
    _parquet_day_dir,
)
from data_engineering.flights.flight_type import (
    add_flight_id_col,
    flights_algorithm_output_path,
    flights_to_flights_df,
    get_flights,
)
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.inference import load_model, run_inference

DEFAULT_MODEL_PATH = (
    OUTPUT_DIR
    / "data/models/flights_airport_nn/runs/2026-09-17-broad-finetune-v1/selected_model.pt"
)
SUPPORTED_SOURCES = ("adsblol", "adsbx")


def derived_path(day, source="adsblol", model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    root = OUTPUT_DIR / "data/flights/algorithm-nn"
    if source != "adsblol" or model_path.resolve() != DEFAULT_MODEL_PATH.resolve():
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        root = root / f"{source}-model-{model_hash}"
    return (
        root
        / f"year={day.year}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
        / "flights.parquet"
    )


def source_signature(day, source, model_path, batch_aircraft):
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported airport-NN source: {source!r}")
    raw_day = (
        _parquet_day_dir(day)
        if source == "adsblol"
        else _adsbx_parquet_day_dir(day)
    )
    paths = [
        flights_algorithm_output_path(day, adsb_src=source, no_airports_model=True),
        OUTPUT_DIR / "data/raw/ourairports/airports.csv",
        Path(__file__),
        Path(__file__).parent / "inference.py",
        Path(__file__).parent / "features/features.py",
        Path(__file__).parents[1] / "airports/airport_lookup.py",
    ]
    paths += sorted(raw_day.glob("icao_bucket=*/data.parquet"))
    if not any(raw_day.glob("icao_bucket=*/data.parquet")):
        raise FileNotFoundError(f"No {source} observations for {day}: {raw_day}")
    return {
        "date": day.isoformat(),
        "source": source,
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "batch_aircraft": batch_aircraft,
        "inputs": [
            [str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths
        ],
    }


def derive_day(
    day,
    source="adsblol",
    model_path=DEFAULT_MODEL_PATH,
    flights=None,
    batch_aircraft=512,
):
    """Return a complete cached parquet path, resuming atomic per-batch parts."""
    if batch_aircraft < 1:
        raise ValueError("batch_aircraft must be positive")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Airport NN checkpoint not found: {model_path}")
    if flights is None:
        flights = get_flights(day, no_airports_model=True, adsb_src=source)
    output = derived_path(day, source, model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = source_signature(day, source, model_path, batch_aircraft)
    manifest_path = output.with_suffix(".manifest.json")
    if (
        output.exists()
        and manifest_path.exists()
        and json.loads(manifest_path.read_text()) == spec
    ):
        print(f"Using cached airport-NN flights: {output}", flush=True)
        return output
    flight_ids = add_flight_id_col(flights)["flight_id"]
    if flight_ids.n_unique() != flights.height:
        raise ValueError("Algorithm segments contain duplicate flight IDs")
    icaos = sorted(flights["icao"].unique().to_list())
    run_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    part_dir = output.parent / "_parts" / run_hash
    part_dir.mkdir(parents=True, exist_ok=True)
    model = config = None
    started = time.monotonic()
    part_paths = []
    for index, start in enumerate(range(0, len(icaos), batch_aircraft)):
        part = part_dir / f"part-{index:04d}.parquet"
        part_paths.append(part)
        if not part.exists():
            if model is None:
                model, config = load_model(model_path)
            block = icaos[start : start + batch_aircraft]
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
            temp = part.with_suffix(".tmp")
            frame.write_parquet(temp)
            temp.replace(part)
        done = min(start + batch_aircraft, len(icaos))
        elapsed = time.monotonic() - started
        print(
            f"{day} aircraft {done:,}/{len(icaos):,} "
            f"({done / len(icaos):.1%}); elapsed {elapsed / 60:.1f} min",
            flush=True,
        )
    if part_paths:
        result = pl.concat([pl.read_parquet(path) for path in part_paths])
    else:
        result = flights.head(0)
    if result.height != flights.height:
        raise ValueError(
            f"Derived {result.height} flights; source contains {flights.height}"
        )
    if set(add_flight_id_col(result)["flight_id"]) != set(flight_ids):
        raise ValueError("Derived flight IDs differ from source segments")
    temp = output.with_suffix(".tmp")
    result.write_parquet(temp)
    temp.replace(output)
    manifest_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Saved {result.height:,} airport-NN flights to {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", choices=SUPPORTED_SOURCES, default="adsblol")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-aircraft", type=int, default=512)
    args = parser.parse_args()
    derive_day(args.date, args.source, args.model, batch_aircraft=args.batch_aircraft)
