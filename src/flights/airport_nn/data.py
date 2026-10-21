"""Prepare ADS-B airport-model training and validation data."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from airports.models import FIXED_WING_AIRPORT_TYPES
from airports.runtime import airport_catalog_for_date, ensure_airport_history
from data_engineering.adsb.canonical import adsb_day_dir
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from data_engineering.utils import OUTPUT_DIR
from flights.airport_nn.features import (
    COLUMNS,
    VERSION,
    clean_messages,
    pack,
    trajectories,
)
from flights.flight_source import (
    flight_path,
    get_flights_for_catalog,
    write_flights_for_catalog,
)
from flights.providers.bts import BTSFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.segmentation.algorithm import _segment_adsb_df
from flights.workers import default_worker_count

CACHE = OUTPUT_DIR / "data/processed/flights_airport_nn/february-2026"
METADATA_SCHEMA = {
    "date": pl.String,
    "icao": pl.String,
    "flight_id": pl.String,
    "endpoint": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "landing_time": pl.Datetime("ms"),
    "truth_id": pl.Int64,
    "truth_ident": pl.String,
    "source": pl.String,
    "label_conflict": pl.Boolean,
    "eligible": pl.Boolean,
    "nearest_id": pl.Int64,
    "nearest_ident": pl.String,
    "candidate_ids": pl.List(pl.Int64),
    "candidate_idents": pl.List(pl.String),
    "target": pl.Int64,
}


def february(day):
    if (day.year, day.month) != (2026, 2):
        raise ValueError("This training dataset accepts only February 2026")
    return day


def fingerprint(path):
    path = Path(path)
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


def references(day):
    """Read canonical BTS/SFDPS labels and flag metadata conflicts."""
    airport_catalog = airport_catalog_for_date(day)
    # This defines the label domain; airport type is not stored as a model feature.
    valid_airport_ids = {
        airport.id
        for airport in airport_catalog.airports
        if airport.type in FIXED_WING_AIRPORT_TYPES
    }
    sources = {
        "SFDPS": SFDPSFlightSource(),
        "BTS": BTSFlightSource(),
    }
    paths = []
    frames = []
    fields = [
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ]
    for source_name, source in sources.items():
        path = flight_path(source, day)
        if not path.exists():
            write_flights_for_catalog(source, day, airport_catalog)
        paths.append(path)
        data = (
            get_flights_for_catalog(source, day, airport_catalog)
            .filter(pl.col("takeoff_time").dt.date() == day)
            .select(fields)
            .unique()
            .sort(fields)
        )
        data = data.filter(
            pl.col("icao").is_not_null()
            & (pl.col("icao") != "")
            & (pl.col("landing_time").dt.date() == day)
            & (pl.col("landing_time") > pl.col("takeoff_time"))
        )
        data = (
            data.with_columns(
                pl.col("icao").str.to_lowercase(),
                pl.lit(source_name).alias("source"),
            )
            .with_row_index("id")
            .with_columns(
                (
                    pl.lit(f"{day}:{source_name}:")
                    + pl.col("id").cast(pl.String)
                ).alias(
                    "reference_id"
                )
            )
            .drop("id")
        )
        for endpoint in ("takeoff", "landing"):
            airport_id = f"{endpoint}_airport_id"
            airport_ident = f"{endpoint}_airport_ident"
            valid = pl.col(airport_id).is_in(valid_airport_ids)
            data = data.with_columns(
                pl.when(valid)
                .then(pl.col(airport_id))
                .alias(f"{endpoint}_truth_id"),
                pl.when(valid)
                .then(pl.col(airport_ident))
                .otherwise(pl.lit(""))
                .alias(f"{endpoint}_truth_ident"),
            )
        frames.append(data)
    refs = pl.concat(frames)
    # Same aircraft cannot operate two overlapping flights in one reference source.
    pairs = refs.join(refs, on=["icao", "source"], suffix="_other").filter(
        (pl.col("reference_id") < pl.col("reference_id_other"))
        & (
            pl.min_horizontal("landing_time", "landing_time_other")
            - pl.max_horizontal("takeoff_time", "takeoff_time_other")
            > pl.duration(seconds=60)
        )
    )
    conflicts = set(pairs["reference_id"]) | set(pairs["reference_id_other"])
    return refs.with_columns(
        pl.col("reference_id").is_in(conflicts).alias("time_conflict")
    ), paths


def match_references(segments, refs):
    pairs = segments.join(
        refs.filter(~pl.col("time_conflict")), on="icao", suffix="_ref"
    ).filter(
        (
            (pl.col("takeoff_time") - pl.col("takeoff_time_ref"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
        & (
            (pl.col("landing_time") - pl.col("landing_time_ref"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
    )
    return pairs.filter(
        (pl.len().over("reference_id") == 1)
        & (pl.len().over(["segment", "source"]) == 1)
    )


def prepare_part(day, refs, path, parquet_root=None):
    start = datetime.combine(day, time())
    columns = COLUMNS + ["geom_altitude_ft", "callsign"]
    aircraft = refs["icao"].unique().to_list()
    if parquet_root is None:
        observations = read_adsb(
            day,
            source=ADSBSource.ADSBLOL,
            icaos=aircraft,
            columns=columns,
            lazy=True,
        )
    else:
        partition = (
            Path(parquet_root)
            / f"year={day.year}/month={day.month:02d}/day={day.day:02d}"
        )
        observations = (
            pl.scan_parquet(str(partition / "icao_bucket=*/*.parquet"))
            .filter(pl.col("icao").is_in(aircraft))
            .select(columns)
        )
    messages = observations.filter(
        (pl.col("time") >= start) & (pl.col("time") < start + timedelta(days=1))
    ).collect(engine="streaming")
    messages = clean_messages(messages)
    native = _segment_adsb_df(messages)
    segments = pl.DataFrame(
        [
            {
                "icao": s.icao,
                "takeoff_time": s.takeoff_time,
                "landing_time": s.landing_time,
            }
            for s in native
        ],
        schema={
            "icao": pl.String,
            "takeoff_time": pl.Datetime("ms"),
            "landing_time": pl.Datetime("ms"),
        },
    ).with_row_index("segment")
    matches = match_references(segments, refs)
    selected = segments.filter(pl.col("segment").is_in(matches["segment"].implode()))
    chosen = {k[0]: g for k, g in matches.partition_by("segment", as_dict=True).items()}
    airport_catalog = airport_catalog_for_date(day)
    endpoints = list(trajectories(selected, messages, airport_catalog))
    rows = []
    for endpoint in endpoints:
        flight = selected.row(endpoint.row, named=True)
        labels = chosen[flight["segment"]]
        truth_ids = set(
            labels[f"{endpoint.endpoint}_truth_id"].drop_nulls().to_list()
        )
        conflict = len(truth_ids) > 1
        truth_id = next(iter(truth_ids)) if len(truth_ids) == 1 else None
        truth_airport = (
            airport_catalog.id_index[truth_id]
            if truth_id is not None
            else None
        )
        rows.append(
            {
                "date": str(day),
                "icao": flight["icao"],
                "flight_id": f"{flight['icao']}_{flight['takeoff_time'].isoformat()}",
                "endpoint": endpoint.endpoint,
                "takeoff_time": flight["takeoff_time"],
                "landing_time": flight["landing_time"],
                "truth_id": truth_id,
                "truth_ident": truth_airport.ident if truth_airport else "",
                "source": "+".join(sorted(set(labels["source"]))),
                "label_conflict": conflict,
                "eligible": bool(
                    truth_id is not None and not conflict
                ),
                "nearest_id": endpoint.nearest_id,
                "nearest_ident": endpoint.nearest_ident,
                "candidate_ids": endpoint.candidate_ids,
                "candidate_idents": endpoint.candidate_idents,
                "target": endpoint.candidate_ids.index(truth_id)
                if truth_id in endpoint.candidate_ids
                else -1,
            }
        )
    metadata = pl.DataFrame(rows, schema=METADATA_SCHEMA)
    x, mask = pack(endpoints)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=x.astype(np.float16),
            mask=mask,
            target=metadata["target"].to_numpy(),
            eligible=metadata["eligible"].to_numpy(),
        )
    metadata.write_parquet(path.with_suffix(".parquet"))
    temp.replace(path)
    return {
        "day": str(day),
        "part": path.name,
        "endpoints": len(rows),
        "eligible": metadata["eligible"].sum(),
        "matched_references": matches.height,
        "references": refs.height,
    }


def prepare(days, output=CACHE, workers=None, batch_aircraft=256, parquet_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = default_worker_count()
    if workers < 1 or batch_aircraft < 1:
        raise ValueError("Workers and aircraft batch size must be positive")
    history = ensure_airport_history()
    tasks = []
    for day in days:
        refs, paths = references(day)
        raw_day = (
            adsb_day_dir(ADSBSource.ADSBLOL, day)
            if parquet_root is None
            else Path(parquet_root)
            / f"year={day.year}/month={day.month:02d}/day={day.day:02d}"
        )
        raw = sorted(raw_day.glob("icao_bucket=*/*.parquet"))
        if not raw:
            raise FileNotFoundError(f"No local ADSB.lol data for {day}")
        folder = output / str(day)
        folder.mkdir(exist_ok=True)
        files = (
            paths
            + raw
            + [
                Path(__file__),
                Path(__file__).with_name("features.py"),
                Path(__file__).parents[1] / "segmentation/algorithm.py",
                Path(__file__).parents[1] / "segmentation/readsb_legs.py",
                Path(__file__).parents[2] / "airports/catalog.py",
                Path(__file__).parents[2] / "airports/models.py",
                Path(__file__).parents[2] / "data_engineering/adsb/read_adsb.py",
                *history.paths.required_files,
            ]
        )
        aircraft = refs["icao"].unique().sort().to_list()
        manifest = {
            "date": str(day),
            "network": "adsblol",
            "batch_aircraft": batch_aircraft,
            "parts": (len(aircraft) + batch_aircraft - 1) // batch_aircraft,
            "inputs": [fingerprint(p) for p in files],
            "airport_identity": "ourairports_id",
            "feature_version": VERSION,
            "labels": "Stable OurAirports IDs from BTS or SFDPS; conflicts excluded",
            "segmentation": "Fresh native segments from this UTC day only; no cached airport guesses",
        }
        dest = folder / "manifest.json"
        if dest.exists() and json.loads(dest.read_text()) != manifest:
            raise ValueError(f"Inputs changed: use a new cache directory ({folder})")
        dest.write_text(json.dumps(manifest, indent=2) + "\n")
        for i, start in enumerate(range(0, len(aircraft), batch_aircraft)):
            path = folder / f"part-{i:04d}.npz"
            if not path.exists() or not path.with_suffix(".parquet").exists():
                tasks.append(
                    (
                        day,
                        refs.filter(
                            pl.col("icao").is_in(
                                aircraft[start : start + batch_aircraft]
                            )
                        ),
                        path,
                        parquet_root,
                    )
                )
    print(
        f"Preparing {len(tasks)} aircraft batches with {workers} worker(s)",
        flush=True,
    )
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            result = prepare_part(*task)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(f"{i}/{len(tasks)} {result}", flush=True)
        return

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(prepare_part, *task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(f"{i}/{len(tasks)} {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat)
    parser.add_argument(
        "--parquet-root",
        type=Path,
        help="ADSB.lol observations root containing year=/month=/day= partitions",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        help="Worker count (default: sized from CPU count and RAM)",
    )
    args = parser.parse_args()
    days = args.dates or [date(2026, 2, d) for d in range(1, 28)]
    output = args.output or CACHE
    prepare(days, output, args.workers, parquet_root=args.parquet_root)
