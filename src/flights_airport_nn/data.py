"""Prepare ADSB.lol training (February BTS/SFDPS) and European validation."""

import argparse
import csv
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from data_engineering.adsb.read_adsb import _parquet_day_dir, read_adsb
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.features import COLUMNS, clean_messages, pack, trajectories
from flights_algorithm.main import _segment_adsb_df

CACHE = OUTPUT_DIR / "data/processed/flights_airport_nn/february-2026"
EUROPE_CACHE = OUTPUT_DIR / "data/processed/flights_airport_nn/europe-validation"
TYPES = ("large_airport", "medium_airport", "small_airport")


def february(day):
    if (day.year, day.month) != (2026, 2):
        raise ValueError("This training dataset accepts only February 2026")
    return day


@lru_cache(maxsize=1)
def catalog():
    with (OUTPUT_DIR / "data/raw/ourairports/airports.csv").open() as f:
        return {r["ident"]: r for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def faa_codes():
    mapping = {}
    for ident, airport in catalog().items():
        if airport["type"] not in TYPES:
            continue
        codes = [
            c
            for c in (airport.get("icao_code"), airport.get("gps_code"))
            if c and re.fullmatch("[A-Z]{4}", c)
        ]
        local = airport.get("local_code", "")
        if (
            airport["iso_country"] == "US"
            and re.fullmatch("[A-Z0-9]{2,5}", local)
            and any(c.isdigit() for c in local)
        ):
            codes.append(local)
        for code in codes:
            mapping.setdefault(code, set()).add(ident)
    return {code: next(iter(ids)) for code, ids in mapping.items() if len(ids) == 1}


def fingerprint(path):
    path = Path(path)
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


def references(day):
    """Keep airline reports and conservative FAA codes; flag metadata conflicts."""
    february(day)
    paths = {
        "SFDPS": sfdps_flights_parquet_path(day),
        "BTS": OUTPUT_DIR
        / "data/flights/bts/v3/year=2026/month=02/bts_flights.parquet",
    }
    frames = []
    fields = [
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
    ]
    for source, path in paths.items():
        data = (
            pl.read_parquet(path)
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
                pl.col("icao").str.to_lowercase(), pl.lit(source).alias("source")
            )
            .with_row_index("id")
            .with_columns(
                (pl.lit(f"{day}:{source}:") + pl.col("id").cast(pl.String)).alias(
                    "reference_id"
                )
            )
            .drop("id")
        )
        for endpoint in ("takeoff", "landing"):
            column = f"{endpoint}_airport_ident"
            mapping = (
                faa_codes()
                if source == "SFDPS"
                else {k: k for k, a in catalog().items() if a["type"] in TYPES}
            )
            data = data.with_columns(
                pl.col(column)
                .replace_strict(mapping, default="")
                .fill_null("")
                .alias(f"{endpoint}_truth")
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
    ), list(paths.values())


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


def ground_support(messages, ident, boundary):
    """Physical corroboration only; predictions never enter label selection."""
    airport = catalog().get(ident)
    if airport is None:
        return False
    points = messages.filter(
        (pl.col("time") - boundary).dt.total_seconds().abs() <= 900
    )
    if points.height < 3:
        return False
    lat, lon = (
        np.radians(points["lat"].to_numpy()),
        np.radians(points["lon"].to_numpy()),
    )
    alat, alon = np.radians(
        [float(airport["latitude_deg"]), float(airport["longitude_deg"])]
    )
    distance = 12742 * np.arcsin(
        np.sqrt(
            np.clip(
                np.sin((lat - alat) / 2) ** 2
                + np.cos(lat) * np.cos(alat) * np.sin((lon - alon) / 2) ** 2,
                0,
                1,
            )
        )
    )
    elevation = float(airport["elevation_ft"]) if airport["elevation_ft"] else np.nan
    low = (
        np.abs(points["baro_altitude_ft"].cast(pl.Float64).to_numpy() - elevation)
        <= 300
    ) & (points["ground_speed_kt"].to_numpy() <= 100)
    grounded = points["on_ground"].fill_null(False).to_numpy() | low
    times = points["time"].dt.epoch("ms").to_numpy()[(distance <= 2) & grounded]
    return bool(len(times) >= 3 and times[-1] - times[0] >= 10_000)


def prepare_part(day, refs, path, parquet_root=None, europe=False):
    start = datetime.combine(day, time())
    columns = COLUMNS + ["geom_altitude_ft", "callsign"]
    aircraft = refs["icao"].unique().to_list()
    if parquet_root is None:
        observations = read_adsb(
            day, source="adsblol", icaos=aircraft, columns=columns, lazy=True
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
    if europe:
        from flights_airport_nn.europe import match_eurocontrol

        matches = match_eurocontrol(segments, refs)
    else:
        matches = match_references(segments, refs)
    selected = segments.filter(pl.col("segment").is_in(matches["segment"].implode()))
    chosen = {k[0]: g for k, g in matches.partition_by("segment", as_dict=True).items()}
    tracks = messages.partition_by("icao", as_dict=True)
    endpoints = list(trajectories(selected, messages))
    if europe:
        # Preserve reference order used by European validation batches.
        by_key = {(selected[e.row, "segment"], e.endpoint): e for e in endpoints}
        endpoints = [
            by_key[segment, end]
            for segment in matches["segment"]
            for end in ("takeoff", "landing")
        ]
    rows = []
    for endpoint in endpoints:
        flight = selected.row(endpoint.row, named=True)
        labels = chosen[flight["segment"]]
        truths = set(labels[f"{endpoint.endpoint}_truth"]) - {""}
        conflict = len(truths) > 1
        truth = next(iter(truths)) if len(truths) == 1 else ""
        bts = (
            labels.filter(
                (pl.col("source") == "BTS")
                & (pl.col(f"{endpoint.endpoint}_truth") == truth)
            ).height
            > 0
            and truth != ""
        )
        grounded = (
            ground_support(
                tracks[(flight["icao"],)], truth, flight[f"{endpoint.endpoint}_time"]
            )
            if truth and not europe
            else False
        )
        rows.append(
            {
                "date": str(day),
                "icao": flight["icao"],
                "flight_id": f"{flight['icao']}_{flight['takeoff_time'].isoformat()}",
                "endpoint": endpoint.endpoint,
                "takeoff_time": flight["takeoff_time"],
                "landing_time": flight["landing_time"],
                "truth": truth,
                "airport_type": catalog().get(truth, {}).get("type", ""),
                "source": "+".join(sorted(set(labels["source"]))),
                "ground_supported": grounded,
                "label_conflict": conflict,
                "eligible": bool(
                    truth and (europe or bts or grounded) and not conflict
                ),
                "nearest": endpoint.nearest,
                "candidates": endpoint.airports,
                "target": endpoint.airports.index(truth)
                if truth in endpoint.airports
                else -1,
            }
        )
    metadata = pl.DataFrame(rows, infer_schema_length=None)
    if not rows:
        metadata = pl.DataFrame(
            schema={
                "date": pl.String,
                "icao": pl.String,
                "flight_id": pl.String,
                "endpoint": pl.String,
                "truth": pl.String,
                "airport_type": pl.String,
                "source": pl.String,
                "eligible": pl.Boolean,
                "nearest": pl.String,
                "candidates": pl.List(pl.String),
                "target": pl.Int64,
            }
        )
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


def prepare(
    days, output=CACHE, workers=4, batch_aircraft=256, parquet_root=None, europe=False
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if workers < 1 or batch_aircraft < 1:
        raise ValueError("Workers and aircraft batch size must be positive")
    tasks = []
    for day in days:
        if europe:
            from flights_airport_nn.europe import references as european_references

            refs, ref_path = european_references(day)
            paths = [ref_path, Path(__file__).with_name("europe.py")]
        else:
            refs, paths = references(february(day))
        raw_day = (
            _parquet_day_dir(day)
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
                Path(__file__).parents[1] / "flights_algorithm/main.py",
                Path(__file__).parents[1] / "flights_algorithm/readsb_legs.py",
                Path(__file__).parents[1] / "airports/airport_lookup.py",
                Path(__file__).parents[1] / "data_engineering/adsb/read_adsb.py",
                OUTPUT_DIR / "data/raw/ourairports/airports.csv",
            ]
        )
        aircraft = refs["icao"].unique().sort().to_list()
        manifest = {
            "date": str(day),
            "network": "adsblol",
            "batch_aircraft": batch_aircraft,
            "parts": (len(aircraft) + batch_aircraft - 1) // batch_aircraft,
            "inputs": [fingerprint(p) for p in files],
            "labels": "EUROCONTROL European endpoints"
            if europe
            else "BTS or ground-supported SFDPS; conflicts excluded",
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
                        europe,
                    )
                )
    print(f"Preparing {len(tasks)} aircraft batches", flush=True)
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
        "--europe",
        action="store_true",
        help="Prepare EUROCONTROL validation, never optimizer data",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        help="ADSB.lol observations root containing year=/month=/day= partitions",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.europe and not args.dates:
        parser.error("European validation requires explicit --dates")
    days = args.dates or [date(2026, 2, d) for d in range(1, 28)]
    output = args.output or (EUROPE_CACHE if args.europe else CACHE)
    prepare(
        days, output, args.workers, parquet_root=args.parquet_root, europe=args.europe
    )
