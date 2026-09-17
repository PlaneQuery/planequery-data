"""Reference-first airport evaluation with independent segmentation per network.

No reference timestamps or airport labels enter segmentation or prediction.
References are matched afterwards by aircraft/time, retaining unmatched rows.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from airports.airport_lookup import AirportLookup
from data_engineering.adsb.read_adsb import (
    _adsbx_parquet_day_dir,
    _parquet_day_dir,
    read_adsb,
)
from data_engineering.flights.flight_type import with_flight_schema_columns
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.audit_source_matrix import _CATALOG, evidence
from flights_airport_nn.broad_cache import CACHE_ROOT
from flights_airport_nn.broader_evaluation import (
    ADSB_COLUMNS,
    load_catalog,
    mask_reception,
)
from flights_airport_nn.dataset import cache_path
from flights_airport_nn.derive_day import DEFAULT_MODEL_PATH
from flights_airport_nn.features.features import create_features
from flights_airport_nn.inference import load_model, predict_endpoints
from flights_airport_nn.reference_labels import AIRPORT_TYPES, resolve_faa_point
from flights_algorithm.main import _segment_adsb_df

VERSION = 1
WORKER = {}
KEY = ["reference_id", "endpoint"]


def fingerprint(path):
    path = Path(path)
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


def model_lineage(model_path):
    """Include pretraining when checking leakage and unseen aircraft/airports."""
    configs, paths = [], []
    while model_path:
        path = Path(model_path)
        if str(path) in paths:
            raise ValueError("Cycle in model lineage")
        paths.append(str(path))
        config = json.loads(path.with_name("config.json").read_text())
        configs.append(config)
        model_path = config.get("initial_model")
    aircraft, airports, train_dates, selection_dates = set(), set(), set(), set()
    for config in configs:
        train_dates.update(config["train_dates"])
        selection_dates.update(config["validation_dates"])
        for text in config["train_dates"]:
            files = (
                list((CACHE_ROOT / text).glob("part-*.npz"))
                if config.get("initial_model")
                else [cache_path(date.fromisoformat(text))]
            )
            if not files or any(not p.exists() for p in files):
                raise FileNotFoundError(
                    f"Cannot establish training membership for {text}"
                )
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    aircraft.update(s.split("_")[0] for s in data["flight_id"])
                    airports.update(s for s in data["truth"] if s)
    return {
        "aircraft": sorted(aircraft),
        "airports": sorted(airports),
        "train_dates": sorted(train_dates),
        "validation_dates": sorted(selection_dates),
        "checkpoints": paths,
    }


def load_references(day, catalog):
    inputs, frames, excluded = [], [], []
    resolved = {}
    paths = {
        "FAA": sfdps_flights_parquet_path(day),
        "BTS": OUTPUT_DIR
        / f"data/flights/bts/v3/year={day.year}/month={day.month:02d}/bts_flights.parquet",
    }
    for source, path in paths.items():
        if not path.exists():
            continue
        inputs.append(fingerprint(path))
        data = pl.read_parquet(path).filter(pl.col("takeoff_time").dt.date() == day)
        data = data.unique(
            subset=[
                "icao",
                "takeoff_time",
                "landing_time",
                "takeoff_airport_ident",
                "landing_airport_ident",
            ],
            maintain_order=True,
        ).sort(
            [
                "icao",
                "takeoff_time",
                "landing_time",
                "takeoff_airport_ident",
                "landing_airport_ident",
            ]
        )
        for index, row in enumerate(data.iter_rows(named=True)):
            record = {
                "reference_id": f"{day}:{source}:{index}",
                "reference_source": source,
                "date": day.isoformat(),
                "icao": row["icao"],
                "takeoff_time": row["takeoff_time"],
                "landing_time": row["landing_time"],
                "callsign": row.get("callsign") or "",
            }
            if (
                not row["icao"]
                or row["landing_time"] is None
                or row["landing_time"] <= row["takeoff_time"]
                or row["landing_time"].date() != day
            ):
                excluded.append(
                    {
                        **record,
                        "exclusion": "missing_identity_or_invalid_or_cross_day_time",
                    }
                )
                continue
            record["icao"] = record["icao"].lower()
            for endpoint in ("takeoff", "landing"):
                raw = row[f"{endpoint}_airport_ident"] or ""
                if source == "FAA":
                    if raw not in resolved:
                        resolved[raw] = resolve_faa_point(raw, catalog)
                    truth, status = resolved[raw]
                else:
                    truth, status = (
                        (raw, "bts_reported_airport")
                        if catalog.get(raw, {}).get("type") in AIRPORT_TYPES
                        else ("", "unsupported_bts_airport")
                    )
                record[f"{endpoint}_raw"] = raw
                record[f"{endpoint}_truth"] = truth
                record[f"{endpoint}_resolution"] = status
            frames.append(record)
    if not frames:
        raise FileNotFoundError(f"No local references for {day}")
    return (
        pl.DataFrame(frames, infer_schema_length=None),
        pl.DataFrame(excluded, infer_schema_length=None),
        inputs,
    )


def temporal_matches(segments, references, tolerance=900):
    """Match independently within each reference source; never use airport labels."""
    # Explicit per-reference bounds support off-block times without pretending
    # they are takeoff times. Defaults preserve symmetric airborne matching.
    bounds = {
        "start_min_s": -tolerance,
        "start_max_s": tolerance,
        "end_tolerance_s": tolerance,
    }
    references = references.with_columns(
        *[pl.lit(v).alias(k) for k, v in bounds.items() if k not in references.columns]
    )
    candidates = (
        segments.select("flight_id", "icao", "takeoff_time", "landing_time")
        .join(
            references.select(
                "reference_id",
                "reference_source",
                "icao",
                pl.col("takeoff_time").alias("reference_start"),
                pl.col("landing_time").alias("reference_end"),
                *bounds,
            ),
            on="icao",
        )
        .with_columns(
            (pl.col("takeoff_time") - pl.col("reference_start"))
            .dt.total_milliseconds()
            .truediv(1000)
            .alias("signed_start_delta_s"),
            (pl.col("landing_time") - pl.col("reference_end"))
            .dt.total_milliseconds()
            .abs()
            .truediv(1000)
            .alias("end_delta_s"),
        )
        .filter(
            pl.col("signed_start_delta_s").is_between(
                pl.col("start_min_s"), pl.col("start_max_s")
            )
            & (pl.col("end_delta_s") <= pl.col("end_tolerance_s"))
        )
        .with_columns(pl.col("signed_start_delta_s").abs().alias("start_delta_s"))
    )
    return candidates.with_columns(
        (
            (pl.len().over("reference_id") == 1)
            & (pl.len().over(["flight_id", "reference_source"]) == 1)
        ).alias("unique_match")
    )


def initialize(model_path, parquet_roots=None):
    if parquet_roots:
        import importlib

        reader = importlib.import_module("data_engineering.adsb.read_adsb")
        for source, root in parquet_roots.items():
            setattr(reader, "ADSBLOL_PARQUET_ROOT" if source == "adsblol" else "ADSBX_PARQUET_ROOT", Path(root))
    torch.set_num_threads(1)
    model, config = load_model(model_path)
    WORKER.update(model=model, config=config)
    _CATALOG.update(load_catalog())
    AirportLookup()


def atomic_parquet(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def predict_segments(segments, messages, loss):
    inputs = with_flight_schema_columns(segments).with_columns(
        pl.lit(None, dtype=pl.String).alias("takeoff_airport_ident"),
        pl.lit(None, dtype=pl.String).alias("landing_airport_ident"),
    )
    observations = mask_reception(inputs, messages, loss) if loss else messages
    features = create_features(inputs, observations)
    predictions = predict_endpoints(features, WORKER["model"], WORKER["config"])
    metadata = {r["flight_id"]: r for r in segments.iter_rows(named=True)}
    tracks = observations.partition_by("icao", as_dict=True)
    candidates = {
        (f.flight_id, f.endpoint): [c.predicted_airport.ident for c in f.candidates]
        for f in features.endpoints
    }
    rows = []
    for p in predictions:
        m = metadata[p["flight_id"]]
        track = tracks.get((m["icao"],))
        point, nearest = None, ""
        if track is not None:
            sample = track.filter(
                (pl.col("time") >= m["takeoff_time"])
                & (pl.col("time") <= m["landing_time"])
            )
            boundary = m[f"{p['endpoint']}_time"]
            sample = sample.filter(
                (pl.col("time") - boundary).dt.total_milliseconds().abs() <= 600_000
            ).sort("time", descending=p["endpoint"] == "landing")
            if sample.height:
                point = sample.row(0, named=True)
                nearest = (
                    AirportLookup().getClosestAirport(point["lat"], point["lon"]).ident
                )
        rows.append(
            {
                "flight_id": p["flight_id"],
                "endpoint": p["endpoint"],
                "loss_seconds": loss,
                "nn": p["airport_ident"] or "",
                "nearest": nearest,
                "confidence": p["confidence"],
                "candidates": candidates[p["flight_id"], p["endpoint"]],
                "observed": point is not None,
                "observation_time": point["time"] if point else None,
                "observation_lat": point["lat"] if point else None,
                "observation_lon": point["lon"] if point else None,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("candidates").cast(pl.List(pl.String))
    )


def run_part(day, source, references, path):
    messages = (
        read_adsb(
            day,
            source=source,
            icaos=references["icao"].unique().to_list(),
            columns=list(
                dict.fromkeys(ADSB_COLUMNS + ["geom_altitude_ft", "callsign"])
            ),
            lazy=True,
        )
        .filter(
            pl.col("time").is_not_null()
            & pl.col("lat").is_finite()
            & pl.col("lon").is_finite()
            & (pl.col("lat").abs() <= 90)
            & (pl.col("lon").abs() <= 180)
        )
        .unique(subset=["icao", "time"])
        .sort(["icao", "time"])
        .collect(engine="streaming")
    )
    # Only motion/ground-state segmentation. No airport guesses or airport-based filters.
    segments = _segment_adsb_df(messages)
    native = with_flight_schema_columns(
        pl.DataFrame(
            [
                {
                    "icao": s.icao,
                    "takeoff_time": s.takeoff_time,
                    "landing_time": s.landing_time,
                }
                for s in segments
            ],
            schema={
                "icao": pl.String,
                "takeoff_time": pl.Datetime("ms"),
                "landing_time": pl.Datetime("ms"),
            },
        )
    ).with_columns(
        (
            pl.col("icao") + "_" + pl.col("takeoff_time").dt.strftime("%Y-%m-%d_%H-%M")
        ).alias("flight_id")
    )
    collisions = native.filter(pl.len().over("flight_id") > 1)
    native = native.filter(pl.len().over("flight_id") == 1)
    atomic_parquet(native, path.with_name(path.stem + "-segments.parquet"))
    matches = temporal_matches(native, references)
    chosen = matches.filter(pl.col("unique_match"))
    selected = native.filter(pl.col("flight_id").is_in(chosen["flight_id"].implode()))
    prediction_parts = []
    if selected.height:
        prediction_parts.append(predict_segments(selected, messages, 0))
        stress_aircraft = [
            i
            for i in selected["icao"].unique()
            if int(hashlib.sha256(f"{day}:{i}".encode()).hexdigest(), 16) % 5 == 0
        ]
        stress = selected.filter(pl.col("icao").is_in(stress_aircraft))
        if stress.height:
            for loss in (60, 180):
                prediction_parts.append(predict_segments(stress, messages, loss))
    predicted = (
        pl.concat(prediction_parts, how="diagonal_relaxed")
        if prediction_parts
        else None
    )
    tracks = {}
    for (icao,), frame in messages.partition_by("icao", as_dict=True).items():
        tracks[icao] = {
            "time": frame["time"].dt.epoch("ms").to_numpy(),
            "lat": frame["lat"].to_numpy(),
            "lon": frame["lon"].to_numpy(),
            "alt": frame["baro_altitude_ft"].cast(pl.Float64).to_numpy(),
            "speed": frame["ground_speed_kt"].cast(pl.Float64).to_numpy(),
            "ground": frame["on_ground"].fill_null(False).to_numpy(),
        }
    records = []
    for r in references.with_columns(
        pl.col("takeoff_time").dt.epoch("ms").alias("start_ms"),
        pl.col("landing_time").dt.epoch("ms").alias("end_ms"),
    ).iter_rows(named=True):
        track = tracks.get(r["icao"])
        for endpoint in ("takeoff", "landing"):
            time = r["start_ms" if endpoint == "takeoff" else "end_ms"]
            row = {
                k: r[k]
                for k in (
                    "reference_id",
                    "reference_source",
                    "date",
                    "icao",
                    "callsign",
                )
            }
            row.update(
                endpoint=endpoint,
                source=source,
                truth=r[f"{endpoint}_truth"],
                raw_label=r[f"{endpoint}_raw"],
                resolution=r[f"{endpoint}_resolution"],
                reference_time=r[f"{endpoint}_time"],
                truth_ground=False,
                truth_ground_points=0,
                truth_ground_span_s=0.0,
                truth_closest_km=None,
                reference_observation_gap_s=None,
            )
            if track is not None:
                lo, hi = np.searchsorted(
                    track["time"], [time - 900_000, time + 900_000]
                )
                if hi > lo:
                    e = evidence(track, row["truth"], lo, hi)
                    row.update(
                        truth_ground=e["strong"],
                        truth_ground_points=e["points"],
                        truth_ground_span_s=e["span_s"],
                        truth_closest_km=e["closest_km"],
                        reference_observation_gap_s=float(
                            np.abs(track["time"][lo:hi] - time).min() / 1000
                        ),
                    )
            records.append(row)
    endpoints = pl.DataFrame(records, infer_schema_length=None).with_columns(
        pl.col("truth_closest_km").cast(pl.Float64),
        pl.col("reference_observation_gap_s").cast(pl.Float64),
    )
    counts = matches.group_by("reference_id").agg(pl.len().alias("candidate_segments"))
    matched = chosen.select(
        "reference_id",
        "flight_id",
        "start_delta_s",
        "end_delta_s",
        pl.col("takeoff_time").alias("segment_start"),
        pl.col("landing_time").alias("segment_end"),
    )
    endpoints = (
        endpoints.join(counts, on="reference_id", how="left")
        .join(matched, on="reference_id", how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("flight_id").is_not_null())
            .then(pl.lit("matched"))
            .when(pl.col("candidate_segments").fill_null(0) > 0)
            .then(pl.lit("ambiguous"))
            .otherwise(pl.lit("unmatched"))
            .alias("match_status")
        )
    )
    if predicted is not None:
        endpoints = endpoints.join(predicted, on=["flight_id", "endpoint"], how="left")
    else:
        endpoints = endpoints.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("loss_seconds"),
            pl.lit(None, dtype=pl.String).alias("nn"),
            pl.lit(None, dtype=pl.String).alias("nearest"),
            pl.lit(None, dtype=pl.Float64).alias("confidence"),
            pl.lit(None, dtype=pl.List(pl.String)).alias("candidates"),
            pl.lit(False).alias("observed"),
            pl.lit(None, dtype=pl.Datetime("ms")).alias("observation_time"),
            pl.lit(None, dtype=pl.Float64).alias("observation_lat"),
            pl.lit(None, dtype=pl.Float64).alias("observation_lon"),
        )
    endpoints = endpoints.with_columns(
        pl.col("loss_seconds").fill_null(0),
        pl.col("nn").fill_null(""),
        pl.col("nearest").fill_null(""),
        pl.col("observed").fill_null(False),
    )
    atomic_parquet(endpoints, path)
    return {
        "source": source,
        "aircraft": references["icao"].n_unique(),
        "reference_flights": references.height,
        "native_segments": native.height,
        "flight_id_collisions": collisions.height,
        "observations": messages.height,
        "matched": chosen.height,
        "path": path.name,
    }


def main(
    days, sources, output, model_path=DEFAULT_MODEL_PATH, workers=3, batch_size=128,
    reference_loader=load_references, parquet_roots=None, extra_policy=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lineage = model_lineage(model_path)
    if {d.isoformat() for d in days} & set(
        lineage["train_dates"] + lineage["validation_dates"]
    ):
        raise ValueError(
            "Evaluation overlaps model training or selection, including pretraining"
        )
    catalog = load_catalog()
    tasks, ref_frames = [], []
    input_files = [fingerprint(OUTPUT_DIR / "data/raw/ourairports/airports.csv")]
    for relative in (
        "audit_source_matrix.py",
        "inference.py",
        "model.py",
        "features/features.py",
        "../flights_algorithm/main.py",
        "../flights_algorithm/readsb_legs.py",
        "../airports/airport_lookup.py",
        "../data_engineering/adsb/read_adsb.py",
    ):
        input_files.append(fingerprint(Path(__file__).parent / relative))
    for day in days:
        refs, excluded, inputs = reference_loader(day, catalog)
        ref_frames.append(refs)
        input_files.extend(inputs)
        refs.write_parquet(output / f"references-{day}.parquet")
        excluded.write_parquet(output / f"excluded-flights-{day}.parquet")
        aircraft = refs["icao"].unique().sort().to_list()
        for source in sources:
            raw = (
                Path(parquet_roots[source]) / f"year={day.year}/month={day.month:02d}/day={day.day:02d}"
                if parquet_roots and source in parquet_roots
                else (_parquet_day_dir if source == "adsblol" else _adsbx_parquet_day_dir)(day)
            )
            files = sorted(raw.glob("icao_bucket=*/data.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No local {source} input for {day}; do not substitute a different network or silently download a different representation"
                )
            input_files.extend(fingerprint(p) for p in files)
            for start in range(0, len(aircraft), batch_size):
                path = output / f"{day}-{source}-part-{start // batch_size:04d}.parquet"
                tasks.append(
                    (
                        day,
                        source,
                        refs.filter(
                            pl.col("icao").is_in(aircraft[start : start + batch_size])
                        ),
                        path,
                    )
                )
    spec = {
        "version": VERSION,
        "dates": [d.isoformat() for d in days],
        "sources": sources,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "lineage": lineage,
        "batch_size": batch_size,
        "inputs": input_files,
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("reference_labels.py").read_bytes()
        ).hexdigest(),
        "policy": {
            "segmentation": "Each network independently; readsb motion/ground-state segmentation; no airport-dependent segment filtering",
            "matching": "Unique aircraft and both times within 900 seconds; no airport or position matching; unmatched references retained",
            "labels": "BTS reported airport or conservative FAA code with physical corroboration flagged separately; no prediction-driven correction",
            "stress": "20 percent of aircraft selected by SHA256(date:icao) modulo 5; remove first/last 60 or 180 seconds from each native segment; fixed boundaries; label evidence comes from unmasked tracks",
            "scope": "Same UTC day flights with reference aircraft identity. FAA cached aircraft associations may depend on ADSB.lol; BTS uses registration mapping. Observational corroboration is not an independent third label source.",
        },
    }
    if extra_policy:
        spec["policy"].update(extra_policy)
    if parquet_roots:
        spec["parquet_roots"] = {k: str(v) for k, v in parquet_roots.items()}
    manifest = output / "protocol.json"
    spec = json.loads(json.dumps(spec))
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Input/protocol changed; use a new output directory")
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    remaining = [t for t in tasks if not t[3].exists()]
    print(
        f"{len(remaining)}/{len(tasks)} native segmentation/prediction batches; frozen protocol {manifest}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(model_path, parquet_roots),
    ) as pool:
        futures = [pool.submit(run_part, *t) for t in remaining]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(i, len(remaining), result, flush=True)
    endpoints = pl.concat(
        [pl.read_parquet(t[3]) for t in tasks], how="diagonal_relaxed"
    )
    endpoints.write_parquet(output / "endpoint_predictions.parquet")
    print(
        f"Saved {endpoints.height:,} source/reference/endpoint/scenario rows",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["adsblol", "adsbx"],
        default=["adsblol", "adsbx"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch size must be positive")
    main(
        args.dates,
        args.sources,
        args.output_dir,
        args.model,
        args.workers,
        args.batch_size,
    )
