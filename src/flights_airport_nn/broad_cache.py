"""Candidate groups from FAA flight times; labels never enter feature construction."""

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data_engineering.adsb.read_adsb import _parquet_day_dir
from data_engineering.flights.flight_type import (
    flights_algorithm_output_path,
    with_flight_schema_columns,
)
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.broader_evaluation import (
    load_catalog,
    match_by_time,
    read_endpoint_messages,
    resolve_labels,
)
from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FEATURE_VERSION,
    MAX_CANDIDATES,
    MAX_POINTS,
    create_features,
)

CACHE_ROOT = OUTPUT_DIR / "data/processed/flights_airport_nn/broad_faa_v1"
TYPES = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}


def prepare(day, batch_aircraft=128):
    out = CACHE_ROOT / day.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    source = [
        sfdps_flights_parquet_path(day),
        flights_algorithm_output_path(day, no_airports_model=True),
        OUTPUT_DIR / "data/raw/ourairports/airports.csv",
    ]
    source += sorted(_parquet_day_dir(day).glob("icao_bucket=*/data.parquet"))
    code = [
        Path(__file__),
        Path(__file__).parent / "broader_evaluation.py",
        Path(__file__).parent / "features/features.py",
    ]
    signature = hashlib.sha256(
        json.dumps(
            [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in source + code]
            + [FEATURE_VERSION, batch_aircraft]
        ).encode()
    ).hexdigest()
    manifest = out / "manifest.json"
    spec = {
        "date": day.isoformat(),
        "signature": signature,
        "batch_aircraft": batch_aircraft,
        "feature_version": FEATURE_VERSION,
        "label_source": "SFDPS",
        "selection": "unique temporal ICAO/start/end matches within 15 minutes; no airport comparison or position filter",
    }
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError(f"Cache signature changed: {out}")
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    labels = pl.read_parquet(source[0])
    segments = pl.read_parquet(source[1])
    matched, stats = match_by_time(segments, labels)
    (out / "matching.json").write_text(json.dumps(stats, indent=2) + "\n")
    catalog = load_catalog()
    resolved = resolve_labels(
        set(matched["raw_takeoff_label"]) | set(matched["raw_landing_label"]), catalog
    )
    icaos = matched["icao"].unique().sort().to_list()
    print(day, stats, "reading observations", len(icaos), "aircraft", flush=True)
    obs = read_endpoint_messages(day, matched)
    print(day, "loaded", obs.height, "observations", flush=True)
    for start in range(0, len(icaos), batch_aircraft):
        path = out / f"part-{start // batch_aircraft:04d}.npz"
        if path.exists():
            continue
        block = icaos[start : start + batch_aircraft]
        flights = matched.filter(pl.col("icao").is_in(block))
        messages = obs.filter(pl.col("icao").is_in(block))
        unlabelled = with_flight_schema_columns(flights).with_columns(
            pl.lit(None, dtype=pl.String).alias("takeoff_airport_ident"),
            pl.lit(None, dtype=pl.String).alias("landing_airport_ident"),
        )
        features = create_features(unlabelled, messages)
        meta = {r["flight_id"]: r for r in flights.iter_rows(named=True)}
        n = len(features.endpoints)
        x = np.zeros(
            (n, MAX_CANDIDATES, MAX_POINTS, ALL_FEATURE_CONFIG.input_size), np.float16
        )
        length = np.zeros(n, np.uint8)
        count = np.zeros(n, np.uint8)
        target = np.full(n, -1, np.int16)
        truth_type = np.full(n, -1, np.int8)
        nearest = np.full(n, -1, np.int16)
        flight_id = np.empty(n, dtype="U40")
        endpoint = np.empty(n, dtype="U8")
        truth = np.empty(n, dtype="U16")
        for i, e in enumerate(features.endpoints):
            flight_id[i] = e.flight_id
            endpoint[i] = e.endpoint
            code = meta[e.flight_id][f"raw_{e.endpoint}_label"]
            ident = resolved.get(code) or ""
            truth[i] = ident
            airport = catalog.get(ident, {})
            truth_type[i] = TYPES.get(airport.get("type"), -1)
            count[i] = len(e.candidates)
            for j, c in enumerate(e.candidates):
                length[i] = len(c.features)
                x[i, j, : length[i]] = c.features
                if c.predicted_airport.ident == ident:
                    target[i] = j
            if count[i]:
                nearest[i] = int(np.argmin(x[i, : count[i], 0, 4]))
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("wb") as stream:
            np.savez(
                stream,
                x=x,
                length=length,
                count=count,
                target=target,
                truth_type=truth_type,
                nearest=nearest,
                flight_id=flight_id,
                endpoint=endpoint,
                truth=truth,
            )
        temp.replace(path)
        print(
            day,
            f"aircraft {min(start + batch_aircraft, len(icaos))}/{len(icaos)}",
            f"endpoints {n}",
            f"retrieved {(target >= 0).mean():.3%}",
            flush=True,
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--batch-aircraft", type=int, default=128)
    args = parser.parse_args()
    for day in args.dates:
        prepare(day, args.batch_aircraft)
