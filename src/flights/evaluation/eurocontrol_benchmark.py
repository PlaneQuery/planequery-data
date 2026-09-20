"""Prepare independent European validation/test endpoints; never fit weights.

EUROCONTROL departure timestamps are off-block, not takeoff. Matching is by
aircraft and time only: -5/+60 minutes at departure, +/-15 at arrival. Labels
are canonical derived identifiers. No proximity or prediction quality filter
is applied to the primary European airport cohort.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import polars as pl
import torch

from data_engineering.eurocontrol.read import eurocontrol_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.airport_benchmark import prepare_part
from flights_airport_nn.data import TYPES, catalog, fingerprint
from flights_airport_nn.evaluate import metrics, score, write_report
from flights_airport_nn.model import load_model


def references(day):
    path = eurocontrol_flights_parquet_path(day)
    fields = [
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
    ]
    raw = pl.read_parquet(path).select(fields).unique().sort(fields)
    airports = catalog()
    european = [
        k for k, a in airports.items() if a["type"] in TYPES and a["continent"] == "EU"
    ]
    refs = (
        raw.with_row_index("id")
        .with_columns(
            pl.col("icao").str.to_lowercase(),
            (pl.lit(f"{day}:EUROCONTROL:") + pl.col("id").cast(pl.String)).alias(
                "reference_id"
            ),
            pl.lit("EUROCONTROL").alias("source"),
            pl.lit("EUROCONTROL").alias("label_source"),
        )
        .drop("id")
    )
    valid = (
        pl.col("icao").str.contains(r"^[0-9a-f]{6}$")
        & (pl.col("takeoff_time").dt.date() == day)
        & (pl.col("landing_time").dt.date() == day)
        & (pl.col("landing_time") > pl.col("takeoff_time"))
        & pl.any_horizontal(
            pl.col(f"{end}_airport_ident").is_in(european)
            for end in ("takeoff", "landing")
        )
    ).fill_null(False)
    excluded = refs.filter(~valid)
    refs = refs.filter(valid)
    for end in ("takeoff", "landing"):
        refs = refs.with_columns(
            pl.when(pl.col(f"{end}_airport_ident").is_in(european))
            .then(pl.col(f"{end}_airport_ident"))
            .otherwise(pl.lit(""))
            .alias(f"{end}_truth"),
            pl.lit(True).alias(f"{end}_reported"),
            pl.lit(False).alias(f"{end}_conflict"),
        )
    pairs = refs.join(refs, on="icao", suffix="_other").filter(
        (pl.col("reference_id") < pl.col("reference_id_other"))
        & (pl.col("takeoff_time") < pl.col("landing_time_other"))
        & (pl.col("takeoff_time_other") < pl.col("landing_time"))
    )
    conflicts = set(pairs["reference_id"]) | set(pairs["reference_id_other"])
    return (
        refs.with_columns(
            pl.col("reference_id").is_in(conflicts).alias("time_conflict")
        ),
        excluded,
        path,
    )


def match_eurocontrol(segments, refs):
    pairs = segments.join(
        refs.filter(~pl.col("time_conflict")), on="icao", suffix="_ref"
    ).filter(
        (pl.col("takeoff_time") - pl.col("takeoff_time_ref"))
        .dt.total_seconds()
        .is_between(-300, 3600)
        & (
            (pl.col("landing_time") - pl.col("landing_time_ref"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
        & (
            pl.min_horizontal("landing_time", "landing_time_ref")
            > pl.max_horizontal("takeoff_time", "takeoff_time_ref")
        )
    )
    return pairs.filter(
        (pl.len().over("reference_id") == 1) & (pl.len().over("segment") == 1)
    )


def prepare(day, network, acquisition, output, workers=4):
    acquisition, output = Path(acquisition), Path(output)
    refs, excluded, ref_path = references(day)
    manifest_path = acquisition / f"acquisition-{network}-{day}.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["date"] != str(day)
        or manifest["source"] != network
        or manifest["reference_sha256"]
        != hashlib.sha256(ref_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Acquisition does not match reference file/date/network")
    aircraft = refs["icao"].unique().sort().to_list()
    if not set(aircraft) <= set(manifest["reference_aircraft"]):
        raise ValueError("Acquisition did not cover every reference aircraft")
    root = acquisition / "parquet" / network
    raw = sorted(
        (root / f"year={day.year}/month={day.month:02d}/day={day.day:02d}").glob(
            "icao_bucket=*/*.parquet"
        )
    )
    if not raw:
        raise FileNotFoundError(f"Missing observations for {network} {day}")
    folder = output / str(day)
    folder.mkdir(parents=True, exist_ok=True)
    parts = (len(aircraft) + 255) // 256
    code_root = Path(__file__).parents[2]
    spec = {
        "date": str(day),
        "network": network,
        "parts": parts,
        "label_source": "EUROCONTROL",
        "scope": "European endpoints only",
        "matching": "off-block -5/+60 min, arrival +/-15 min; unique aircraft/time",
        "inputs": [fingerprint(p) for p in [ref_path, manifest_path, *raw]],
        "code": [
            fingerprint(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("airport_benchmark.py"),
                code_root / "flights_airport_nn/features.py",
                code_root / "flights_airport_nn/data.py",
                code_root / "flights_algorithm/main.py",
                code_root / "flights_algorithm/readsb_legs.py",
                code_root / "airports/airport_lookup.py",
                OUTPUT_DIR / "data/raw/ourairports/airports.csv",
            ]
        ],
    }
    marker = folder / "manifest.json"
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError(f"Changed preparation inputs: choose a fresh output: {folder}")
    marker.write_text(json.dumps(spec, indent=2) + "\n")
    refs.write_parquet(folder / "reference-definitions.parquet")
    excluded.write_parquet(folder / "excluded-references.parquet")
    paths = [folder / f"part-{i:04d}.npz" for i in range(parts)]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                prepare_part,
                day,
                network,
                refs.filter(pl.col("icao").is_in(aircraft[i * 256 : (i + 1) * 256])),
                path,
                root,
                match_eurocontrol,
            )
            for i, path in enumerate(paths)
            if not path.exists()
        ]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"{day} {network}: {i}/{len(futures)} {future.result()}", flush=True)
    return paths


def report(model_path, paths, output):
    """Frozen test: preserve all references and count unmatched as recovery misses."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    used = (
        checkpoint["training"]["train_dates"]
        + checkpoint["training"]["validation_dates"]
    )
    if any(p.parent.name in used for p in paths):
        raise ValueError("Final test overlaps model development")
    torch.set_num_threads(4)
    model = load_model(model_path, "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = score(model, paths)
    summary = write_report(data, paths, output)
    refs = pl.read_parquet(Path(output) / "references.parquet")
    rows = []
    for end in ("takeoff", "landing"):
        side = data.filter(pl.col("endpoint") == end).select(
            "reference_id",
            "nn",
            "nearest",
            "nn_correct",
            "nearest_correct",
            pl.col("takeoff_time").alias("native_takeoff_time"),
            pl.col("landing_time").alias("native_landing_time"),
        )
        rows.append(
            refs.with_columns(
                pl.lit(end).alias("endpoint"),
                pl.col(f"{end}_truth").alias("truth"),
            ).join(side, on="reference_id", how="left", validate="1:1")
        )
    all_endpoints = pl.concat(rows).with_columns(
        pl.col("nn_correct", "nearest_correct").fill_null(False),
        pl.when(pl.col("matched"))
        .then(pl.lit("both"))
        .otherwise(pl.lit("reference_only"))
        .alias("match_status"),
    )
    all_endpoints.write_parquet(Path(output) / "all-reference-endpoints.parquet")
    eligible = all_endpoints.filter(~pl.col("time_conflict") & (pl.col("truth") != ""))
    summary["reference_endpoint_recovery"] = {
        "endpoints": eligible.height,
        "matched": eligible["matched"].sum(),
        "nn": eligible["nn_correct"].mean(),
        "nearest": eligible["nearest_correct"].mean(),
    }
    strict = data.filter(
        (pl.col("takeoff_time") - pl.col("takeoff_time_ref"))
        .dt.total_seconds()
        .is_between(-120, 1800)
        & (
            (pl.col("landing_time") - pl.col("landing_time_ref"))
            .dt.total_seconds()
            .abs()
            <= 300
        )
    )
    summary["strict_time_subset"] = metrics(strict)
    summary["model_sha256"] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    (Path(output) / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["primary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--network", choices=["adsblol", "adsbx"], default="adsblol")
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    paths = prepare(
        args.date, args.network, args.acquisition, args.output, args.workers
    )
    if args.model:
        report(args.model, paths, args.output / str(args.date) / "results")
