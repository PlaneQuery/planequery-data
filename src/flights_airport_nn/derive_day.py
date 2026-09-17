"""Derive and cache standard Flight-schema Parquets for the shared evaluator."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import (
    _adsbx_parquet_day_dir,
    _parquet_day_dir,
    read_adsb,
)
from data_engineering.flights.flight_type import get_flights, with_flight_schema_columns
from data_engineering.utils import OUTPUT_DIR

DEFAULT_MODEL_PATH = (
    OUTPUT_DIR / "data/models/flights_airport_nn/february-2026/model.pt"
)


def derive_day(
    day,
    source="adsblol",
    model_path=DEFAULT_MODEL_PATH,
    flights=None,
    batch_aircraft=256,
):
    from flights_airport_nn.features import COLUMNS
    from flights_airport_nn.inference import predict_flights
    from flights_airport_nn.model import load_model

    if source not in ("adsblol", "adsbx") or batch_aircraft < 1:
        raise ValueError("Require adsblol/adsbx and a positive batch size")
    model_path = Path(model_path)
    if flights is None:
        flights = get_flights(day, no_airports_model=True, adsb_src=source)
    flights = with_flight_schema_columns(flights)
    raw = sorted(
        (
            _parquet_day_dir(day)
            if source == "adsblol"
            else _adsbx_parquet_day_dir(day)
        ).glob("icao_bucket=*/data.parquet")
    )
    if not raw and flights.height:
        raise FileNotFoundError(f"No local {source} observations for {day}")
    # Include supplied flight contents: custom subsets must not reuse a full-day
    # cache or silently lose changes to metadata/boundaries.
    spec = {
        "model": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "flights": hashlib.sha256(flights.serialize()).hexdigest(),
        "source": source,
        "batch": batch_aircraft,
        "inputs": [
            (str(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in raw + [OUTPUT_DIR / "data/raw/ourairports/airports.csv"]
        ],
        "code": hashlib.sha256(
            b"".join(
                Path(__file__).with_name(n).read_bytes()
                for n in ("derive_day.py", "features.py", "inference.py", "model.py")
            )
        ).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    root = (
        OUTPUT_DIR
        / f"data/flights/algorithm-nn/{source}/year={day.year}/month={day.month:02d}/day={day.day:02d}/{key}"
    )
    root.mkdir(parents=True, exist_ok=True)
    output = root / "flights.parquet"
    if output.exists():
        return output
    (root / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    aircraft = flights["icao"].unique().sort().to_list()
    print(
        f"{day} {source}: assigning airports to {flights.height:,} flights ({len(aircraft):,} aircraft)",
        flush=True,
    )
    parts, model = [], None
    for index, start in enumerate(range(0, len(aircraft), batch_aircraft)):
        part = root / f"part-{index:04d}.parquet"
        parts.append(part)
        if not part.exists():
            if model is None:
                import torch

                torch.set_num_threads(2)
                model = load_model(model_path)
            subset = flights.filter(
                pl.col("icao").is_in(aircraft[start : start + batch_aircraft])
            )
            messages = read_adsb(
                day,
                source=source,
                icaos=subset["icao"].unique().to_list(),
                columns=COLUMNS,
            )
            predicted = predict_flights(subset, messages, model)
            temp = part.with_suffix(".tmp")
            predicted.write_parquet(temp)
            temp.replace(part)
        print(
            f"{day} {source}: {min(start + batch_aircraft, len(aircraft))}/{len(aircraft)} aircraft",
            flush=True,
        )
    result = pl.concat([pl.read_parquet(p) for p in parts]) if parts else flights
    if result.height != flights.height or result.schema != flights.schema:
        raise ValueError("Inference changed the flight count or schema")
    temp = output.with_suffix(".tmp")
    result.write_parquet(temp)
    temp.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", choices=["adsblol", "adsbx"], default="adsblol")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(derive_day(args.date, args.source, args.model))
