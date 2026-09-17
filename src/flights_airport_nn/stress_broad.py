"""Full held-out FAA and reception-loss assessment for broadly trained checkpoints."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from data_engineering.flights.flight_type import flights_algorithm_output_path
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from flights_airport_nn.broad_cache import CACHE_ROOT
from flights_airport_nn.broader_evaluation import (
    load_catalog,
    mask_reception,
    match_by_time,
    read_endpoint_messages,
    resolve_labels,
    score_batch,
    stress_selected,
    summarize_broader,
    write_report,
)
from flights_airport_nn.inference import load_model


def run(model_path, dates, output, batch_aircraft=128):
    model_path = Path(model_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model, config = load_model(model_path)
    torch.set_num_threads(4)
    if set(dates) & (set(model.training_dates) | set(model.validation_dates)):
        raise ValueError("Assessment dates overlap model training or validation")
    source_dates = json.loads((model_path.parent / "config.json").read_text())[
        "train_dates"
    ]
    seen_aircraft = set()
    seen_airports = set()
    for day_text in source_dates:
        for path in (CACHE_ROOT / day_text).glob("part-*.npz"):
            with np.load(path, allow_pickle=False) as cache:
                seen_aircraft.update(f.split("_")[0] for f in cache["flight_id"])
                seen_airports.update(cache["truth"][cache["target"] >= 0])
    spec = {
        "model": str(model_path),
        "model_sha256": model_hash,
        "dates": [d.isoformat() for d in dates],
        "batch_aircraft": batch_aircraft,
        "stress_seconds": [0, 120, 300],
        "feature_version": 2,
        "seen_definition": "broad fine-tuning dates only",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    existing = output / "config.json"
    if existing.exists() and json.loads(existing.read_text()) != spec:
        raise ValueError("Output directory has different assessment configuration")
    existing.write_text(json.dumps(spec, indent=2) + "\n")
    catalog = load_catalog()
    matching = {}
    for day in dates:
        labels = pl.read_parquet(sfdps_flights_parquet_path(day))
        segments = pl.read_parquet(
            flights_algorithm_output_path(day, no_airports_model=True)
        )
        matched, stats = match_by_time(segments, labels)
        stats["raw_label_rows"] = labels.height
        matching[day.isoformat()] = stats
        (output / "matching.json").write_text(json.dumps(matching, indent=2) + "\n")
        resolved = resolve_labels(
            set(matched["raw_takeoff_label"]) | set(matched["raw_landing_label"]),
            catalog,
        )
        icaos = matched["icao"].unique().sort().to_list()
        print(day, stats, "reading observations", len(icaos), "aircraft", flush=True)
        obs = read_endpoint_messages(day, matched)
        print(day, "loaded", obs.height, "endpoint observations", flush=True)
        for start in range(0, len(icaos), batch_aircraft):
            part = output / f"{day}-part-{start // batch_aircraft:04d}.parquet"
            if part.exists():
                continue
            block = icaos[start : start + batch_aircraft]
            flights = matched.filter(pl.col("icao").is_in(block))
            messages = obs.filter(pl.col("icao").is_in(block))
            results = [
                score_batch(
                    day,
                    flights,
                    messages,
                    model,
                    config,
                    catalog,
                    resolved,
                    seen_aircraft,
                    seen_airports,
                )
            ]
            selected = flights.filter(
                pl.col("flight_id").map_elements(
                    stress_selected, return_dtype=pl.Boolean
                )
            )
            if selected.height:
                for seconds in (120, 300):
                    masked = mask_reception(selected, messages, seconds)
                    results.append(
                        score_batch(
                            day,
                            selected,
                            masked,
                            model,
                            config,
                            catalog,
                            resolved,
                            seen_aircraft,
                            seen_airports,
                            seconds,
                        )
                    )
            table = pl.concat(results, how="diagonal_relaxed")
            temp = part.with_suffix(".tmp")
            table.write_parquet(temp)
            temp.replace(part)
            print(
                day,
                f"aircraft {min(start + batch_aircraft, len(icaos))}/{len(icaos)}",
                flush=True,
            )
    predictions = pl.concat(
        [pl.read_parquet(path) for path in sorted(output.glob("*-part-*.parquet"))],
        how="diagonal_relaxed",
    )
    predictions.write_parquet(output / "predictions.parquet")
    metrics = summarize_broader(predictions, matching)
    metrics["limitations"].append(
        "Seen-aircraft and seen-airport cohorts use broad fine-tuning dates only; "
        "they do not include the checkpoint's earlier commercial pretraining."
    )
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
        raise RuntimeError("Checkpoint changed")
    metrics["model_unchanged"] = True
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.filter(
        pl.col("scorable") & (pl.col("coverage_loss_s") == 0) & ~pl.col("correct")
    ).sort("confidence", descending=True).write_csv(output / "errors.csv")
    write_report(output, metrics)
    print(
        json.dumps(
            {"groups": metrics["groups"], "stress": metrics["stress"]}, indent=2
        ),
        flush=True,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-aircraft", type=int, default=128)
    args = parser.parse_args()
    run(args.model, args.dates, args.output_dir, args.batch_aircraft)
