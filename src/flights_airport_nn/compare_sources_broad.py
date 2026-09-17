"""Score one FAA-labeled flight cohort on ADSB.lol and ADSBx observations."""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    match_by_time,
    read_endpoint_messages,
    resolve_labels,
    score_batch,
)
from flights_airport_nn.inference import load_model

SOURCES = ("adsblol", "adsbx")
AIRPORT_TYPES = ("large_airport", "medium_airport", "small_airport")
_WORKER = {}


def _initialize_worker(model_path, resolved, seen_aircraft, seen_airports):
    torch.set_num_threads(1)
    model, config = load_model(model_path)
    _WORKER.update(
        model=model,
        config=config,
        catalog=load_catalog(),
        resolved=resolved,
        seen_aircraft=seen_aircraft,
        seen_airports=seen_airports,
    )


def _score_part(day, source, flights, path):
    messages = read_endpoint_messages(day, flights, source=source)
    scored = score_batch(
        day,
        flights,
        messages,
        _WORKER["model"],
        _WORKER["config"],
        _WORKER["catalog"],
        _WORKER["resolved"],
        _WORKER["seen_aircraft"],
        _WORKER["seen_airports"],
    ).with_columns(pl.lit(source).alias("source"))
    if scored.height != flights.height * 2:
        raise ValueError(f"{source}: expected two endpoints for every flight")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    scored.write_parquet(temporary)
    temporary.replace(path)
    return source, path.name, flights.height, messages.height


def _training_cohorts(model_path):
    config = json.loads((model_path.parent / "config.json").read_text())
    seen_aircraft = set()
    seen_airports = set()
    for day_text in config["train_dates"]:
        for path in (CACHE_ROOT / day_text).glob("part-*.npz"):
            with np.load(path, allow_pickle=False) as cache:
                seen_aircraft.update(s.split("_")[0] for s in cache["flight_id"])
                seen_airports.update(cache["truth"][cache["target"] >= 0])
    return seen_aircraft, seen_airports


def _paired_predictions(predictions, expected_endpoints):
    names = ["date", "flight_id", "icao", "endpoint", "raw_label", "truth", "airport_type"]
    left = predictions.filter(pl.col("source") == "adsblol").select(
        names
        + [
            pl.col("prediction").alias("adsblol_nn"),
            pl.col("nearest").alias("adsblol_nearest"),
            pl.col("correct").alias("adsblol_nn_correct"),
            pl.col("nearest_correct").alias("adsblol_nearest_correct"),
            pl.col("observation_available").alias("adsblol_observed"),
            pl.col("retrieved").alias("adsblol_retrieved"),
        ]
    )
    right = predictions.filter(pl.col("source") == "adsbx").select(
        "flight_id",
        "endpoint",
        pl.col("truth").alias("adsbx_truth"),
        pl.col("prediction").alias("adsbx_nn"),
        pl.col("nearest").alias("adsbx_nearest"),
        pl.col("correct").alias("adsbx_nn_correct"),
        pl.col("nearest_correct").alias("adsbx_nearest_correct"),
        pl.col("observation_available").alias("adsbx_observed"),
        pl.col("retrieved").alias("adsbx_retrieved"),
    )
    if left.height != expected_endpoints or right.height != expected_endpoints:
        raise ValueError("A source did not score every matched endpoint")
    if left.select("flight_id", "endpoint").is_duplicated().any() or right.select(
        "flight_id", "endpoint"
    ).is_duplicated().any():
        raise ValueError("Duplicate flight endpoint in source predictions")
    paired = left.join(right, on=["flight_id", "endpoint"], validate="1:1")
    if paired.height != expected_endpoints or paired.filter(
        pl.col("truth") != pl.col("adsbx_truth")
    ).height:
        raise ValueError("Source cohorts or FAA labels differ")
    return paired.drop("adsbx_truth")


def _group_metrics(frame):
    result = {"endpoints": frame.height, "flights": frame["flight_id"].n_unique()}
    for source in SOURCES:
        result[f"{source}_observed"] = frame[f"{source}_observed"].sum()
        result[f"{source}_candidate_recall"] = (
            frame[f"{source}_retrieved"].mean() if frame.height else None
        )
        for method in ("nn", "nearest"):
            count = frame[f"{source}_{method}_correct"].sum()
            result[f"{source}_{method}_correct"] = count
            result[f"{source}_{method}_accuracy"] = (
                count / frame.height if frame.height else None
            )
    return result


def run(day, model_path, output, batch_aircraft=128, workers=3):
    model_path = Path(model_path)
    output = Path(output)
    if batch_aircraft < 1 or workers < 1:
        raise ValueError("batch_aircraft and workers must be positive")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if day.isoformat() in set(checkpoint.get("train_dates", [])) | set(
        checkpoint.get("validation_dates", [])
    ):
        raise ValueError("Comparison date overlaps training or validation")
    segment_path = flights_algorithm_output_path(day, no_airports_model=True)
    label_path = sfdps_flights_parquet_path(day)
    spec = {
        "date": day.isoformat(),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "segments": [str(segment_path), segment_path.stat().st_size, segment_path.stat().st_mtime_ns],
        "labels": [str(label_path), label_path.stat().st_size, label_path.stat().st_mtime_ns],
        "batch_aircraft": batch_aircraft,
        "sources": SOURCES,
        "definition": "Same ADSB.lol algorithm segments matched uniquely to FAA SFDPS flights within 15 minutes at both boundaries; only endpoint observations change between sources.",
        "interpretation": "This date was used in an earlier commercial evaluation and is an exploratory source comparison, not a fresh held-out model test.",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(spec)):
        raise ValueError("Output directory belongs to a different source comparison")
    manifest.write_text(json.dumps(spec, indent=2) + "\n")

    segments = pl.read_parquet(segment_path)
    labels = pl.read_parquet(label_path)
    matched, matching = match_by_time(segments, labels)
    (output / "matching.json").write_text(json.dumps(matching, indent=2) + "\n")
    print(day, matching, flush=True)
    catalog = load_catalog()
    resolved = resolve_labels(
        set(matched["raw_takeoff_label"]) | set(matched["raw_landing_label"]),
        catalog,
    )
    seen_aircraft, seen_airports = _training_cohorts(model_path)
    aircraft = matched["icao"].unique().sort().to_list()
    blocks = [
        matched.filter(pl.col("icao").is_in(aircraft[start : start + batch_aircraft]))
        for start in range(0, len(aircraft), batch_aircraft)
    ]
    part_dir = output / "parts"
    part_dir.mkdir(exist_ok=True)
    tasks = []
    for source in SOURCES:
        for index, block in enumerate(blocks):
            path = part_dir / f"{source}-part-{index:04d}.parquet"
            if not path.exists():
                tasks.append((source, block, path))
    print(f"Scoring {len(tasks)} remaining source batches", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(model_path, resolved, seen_aircraft, seen_airports),
    ) as pool:
        futures = {
            pool.submit(_score_part, day, source, block, path): (source, path)
            for source, block, path in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            source, name, flights, messages = future.result()
            print(
                f"{completed}/{len(tasks)} {source} {name}: "
                f"{flights} flights, {messages} endpoint observations",
                flush=True,
            )
    paths = [
        part_dir / f"{source}-part-{index:04d}.parquet"
        for source in SOURCES
        for index in range(len(blocks))
    ]
    predictions = pl.concat([pl.read_parquet(path) for path in paths])
    paired = _paired_predictions(predictions, matched.height * 2)
    paired.write_parquet(output / "paired_predictions.parquet")
    scored = paired.filter(
        (pl.col("truth") != "") & pl.col("airport_type").is_in(AIRPORT_TYPES)
    )
    metrics = {
        **spec,
        "matching": matching,
        "endpoints_total": paired.height,
        "endpoints_unresolved": paired.filter(pl.col("truth") == "").height,
        "endpoints_unsupported_airport_type": paired.filter(
            (pl.col("truth") != "") & ~pl.col("airport_type").is_in(AIRPORT_TYPES)
        ).height,
        "groups": {
            "all": _group_metrics(scored),
            **{
                name: _group_metrics(scored.filter(pl.col("airport_type") == name))
                for name in AIRPORT_TYPES
            },
            "takeoff": _group_metrics(scored.filter(pl.col("endpoint") == "takeoff")),
            "landing": _group_metrics(scored.filter(pl.col("endpoint") == "landing")),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        f"# FAA airport-label source comparison, {day}",
        "",
        spec["definition"],
        spec["interpretation"],
        "Unresolved labels and unsupported airport types are excluded from the scored rows. Missing observations or airport candidates count as errors.",
        "",
        "| Cohort | Endpoints | ADSB.lol NN | ADSB.lol nearest | ADSBx NN | ADSBx nearest |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["groups"].items():
        if row["endpoints"]:
            lines.append(
                f"| {name} | {row['endpoints']:,} | "
                f"{row['adsblol_nn_accuracy']:.2%} | "
                f"{row['adsblol_nearest_accuracy']:.2%} | "
                f"{row['adsbx_nn_accuracy']:.2%} | "
                f"{row['adsbx_nearest_accuracy']:.2%} |"
            )
    lines += [
        "",
        f"Source observations available on {metrics['groups']['all']['adsblol_observed']:,} ADSB.lol and {metrics['groups']['all']['adsbx_observed']:,} ADSBx scored endpoints.",
        "FAA SFDPS is a single label source; disagreements are not independently verified model errors.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(json.dumps(metrics["groups"], indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-aircraft", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.date, args.model, args.output_dir, args.batch_aircraft, args.workers)
