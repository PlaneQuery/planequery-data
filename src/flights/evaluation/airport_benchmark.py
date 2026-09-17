"""Compare frozen airport models and nearest on shared, independent references.

This is evaluation only. It never writes training caches or updates a model.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from data_engineering.adsb.read_adsb import (
    _adsbx_parquet_day_dir,
    _parquet_day_dir,
    read_adsb,
)
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights.ground_truth.bts_and_sfdps_to_gold import ground_truth_flights_output_path
from flights_airport_nn.data import (
    TYPES,
    catalog,
    faa_codes,
    fingerprint,
    ground_support,
    match_references,
)
from flights_airport_nn.derive_day import DEFAULT_MODEL_PATH
from flights_airport_nn.evaluate import score
from flights_airport_nn.features import (
    COLUMNS,
    WINDOW_MS,
    clean_messages,
    pack,
    trajectories,
)
from flights_airport_nn.model import load_model
from flights_algorithm.main import _segment_adsb_df

ROOT = OUTPUT_DIR / "data/experiments/flights_airport_nn/february-2026-march-test"
KEY = ["reference_id", "endpoint"]


def references(day):
    paths = {
        "SFDPS": sfdps_flights_parquet_path(day),
        "BTS": OUTPUT_DIR
        / f"data/flights/bts/v3/year={day.year}/month={day.month:02d}/bts_flights.parquet",
        "original": ground_truth_flights_output_path(day, "adsblol"),
    }
    fields = [
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
    ]
    frames, exclusions = {}, []
    for source, path in paths.items():
        frame = (
            pl.read_parquet(path)
            .select(fields)
            .filter(pl.col("takeoff_time").dt.date() == day)
            .unique()
            .sort(fields)
            .with_columns(pl.col("icao").str.to_lowercase())
        )
        valid = (
            pl.col("icao").is_not_null()
            & (pl.col("icao") != "")
            & (pl.col("landing_time").dt.date() == day)
            & (pl.col("landing_time") > pl.col("takeoff_time"))
        ).fill_null(False)
        exclusions.append(
            {
                "source": source,
                "all_rows": frame.height,
                "invalid_or_cross_day": frame.filter(~valid).height,
            }
        )
        frame = (
            frame.filter(valid)
            .with_row_index("index")
            .with_columns(
                (pl.lit(f"{day}:{source}:") + pl.col("index").cast(pl.String)).alias(
                    "reference_id"
                )
            )
            .drop("index")
        )
        mapping = (
            faa_codes()
            if source == "SFDPS"
            else {k: k for k, a in catalog().items() if a["type"] in TYPES}
        )
        frame = frame.with_columns(
            *[
                pl.col(f"{end}_airport_ident")
                .replace_strict(mapping, default="")
                .fill_null("")
                .alias(f"{end}_truth")
                for end in ("takeoff", "landing")
            ]
        )
        frames[source] = frame
    # One broad reference per flight: corroborate unique FAA/BTS temporal pairs,
    # retain FAA-only flights, then add BTS flights without any FAA overlap.
    faa, bts = frames["SFDPS"], frames["BTS"]
    pairs = faa.join(bts, on="icao", suffix="_bts").filter(
        (
            (pl.col("takeoff_time") - pl.col("takeoff_time_bts"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
        & (
            (pl.col("landing_time") - pl.col("landing_time_bts"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
    )
    unique = pairs.filter(
        (pl.len().over("reference_id") == 1) & (pl.len().over("reference_id_bts") == 1)
    )
    paired = {r["reference_id"]: r for r in unique.iter_rows(named=True)}
    rows = []
    for r in faa.iter_rows(named=True):
        partner = paired.get(r["reference_id"])
        r.update(source="broad", label_source="SFDPS+BTS" if partner else "SFDPS")
        for end in ("takeoff", "landing"):
            truth = r[f"{end}_truth"]
            bts_truth = partner[f"{end}_truth_bts"] if partner else ""
            conflict = bool(truth and bts_truth and truth != bts_truth)
            r[f"{end}_conflict"] = conflict
            r[f"{end}_truth"] = "" if conflict else bts_truth or truth
            r[f"{end}_reported"] = bool(bts_truth and not conflict)
        rows.append(r)
    for source in ("BTS", "original"):
        frame = frames[source]
        if source == "BTS":
            frame = frame.filter(
                ~pl.col("reference_id").is_in(pairs["reference_id_bts"].implode())
            )
        for r in frame.iter_rows(named=True):
            r.update(
                source="original" if source == "original" else "broad",
                label_source=source,
            )
            for end in ("takeoff", "landing"):
                r[f"{end}_conflict"] = False
                r[f"{end}_reported"] = bool(r[f"{end}_truth"])
            rows.append(r)
    refs = pl.DataFrame(rows, infer_schema_length=None)
    overlap = refs.join(refs, on=["icao", "source"], suffix="_other").filter(
        (pl.col("reference_id") < pl.col("reference_id_other"))
        & (
            pl.min_horizontal("landing_time", "landing_time_other")
            - pl.max_horizontal("takeoff_time", "takeoff_time_other")
            > pl.duration(seconds=60)
        )
    )
    conflicts = set(overlap["reference_id"]) | set(overlap["reference_id_other"])
    refs = refs.with_columns(
        pl.col("reference_id").is_in(conflicts).alias("time_conflict")
    )
    return refs, {
        "files": [fingerprint(p) for p in paths.values()],
        "exclusions": exclusions,
        "bts_pairs_ambiguous": pairs.height - unique.height,
    }


def prepare_part(day, network, refs, path):
    start = datetime.combine(day, time())
    messages = clean_messages(
        read_adsb(
            day,
            source=network,
            icaos=refs["icao"].unique().to_list(),
            columns=COLUMNS + ["geom_altitude_ft", "callsign"],
            lazy=True,
        )
        .filter(
            (pl.col("time") >= start) & (pl.col("time") < start + timedelta(days=1))
        )
        .collect(engine="streaming")
    )
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
    trajectories_by_key = {
        (selected[e.row, "segment"], e.endpoint): e
        for e in trajectories(selected, messages)
    }
    tracks = messages.partition_by("icao", as_dict=True)
    endpoints, rows = [], []
    for match in matches.iter_rows(named=True):
        for end in ("takeoff", "landing"):
            endpoint = trajectories_by_key[match["segment"], end]
            truth = match[f"{end}_truth"]
            boundary = match[f"{end}_time"]
            window = tracks[(match["icao"],)].filter(
                pl.col("time").is_between(
                    max(
                        match["takeoff_time"],
                        boundary - timedelta(milliseconds=WINDOW_MS),
                    ),
                    min(
                        match["landing_time"],
                        boundary + timedelta(milliseconds=WINDOW_MS),
                    ),
                )
            )
            point = (
                window.row(-1 if end == "landing" else 0, named=True)
                if window.height
                else {}
            )
            rows.append(
                {
                    "reference_id": match["reference_id"],
                    "endpoint": end,
                    "date": str(day),
                    "network": network,
                    "icao": match["icao"],
                    "source": match["source"],
                    "label_source": match["label_source"],
                    "truth": truth,
                    "airport_type": catalog().get(truth, {}).get("type", ""),
                    "reported": match[f"{end}_reported"],
                    "label_conflict": match[f"{end}_conflict"],
                    "ground_supported": ground_support(
                        tracks[(match["icao"],)], truth, match[f"{end}_time_ref"]
                    )
                    if truth
                    else False,
                    "eligible": bool(truth and not match[f"{end}_conflict"]),
                    "nearest": endpoint.nearest,
                    "candidates": endpoint.airports,
                    "target": endpoint.airports.index(truth)
                    if truth in endpoint.airports
                    else -1,
                    "takeoff_time": match["takeoff_time"],
                    "landing_time": match["landing_time"],
                    "takeoff_time_ref": match["takeoff_time_ref"],
                    "landing_time_ref": match["landing_time_ref"],
                    "observed_time": point.get("time"),
                    "observed_lat": point.get("lat"),
                    "observed_lon": point.get("lon"),
                    "observed_altitude_ft": point.get("baro_altitude_ft"),
                    "observed_speed_kt": point.get("ground_speed_kt"),
                    "observed_on_ground": point.get("on_ground"),
                }
            )
            endpoints.append(endpoint)
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema={"target": pl.Int64, "eligible": pl.Boolean})
    )
    values, mask = pack(endpoints)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(
            f,
            x=values.astype(np.float16),
            mask=mask,
            target=frame["target"].to_numpy(),
            eligible=frame["eligible"].to_numpy(),
        )
    frame.write_parquet(path.with_suffix(".parquet"))
    refs.with_columns(
        pl.col("reference_id").is_in(matches["reference_id"].implode()).alias("matched")
    ).write_parquet(path.with_name(path.stem + "-references.parquet"))
    path.with_suffix(".tmp").replace(path)
    return str(path)


def prepare(day, networks, output, workers):
    refs, provenance = references(day)
    folder = output / str(day)
    folder.mkdir(parents=True, exist_ok=True)
    aircraft = refs["icao"].unique().sort().to_list()
    tasks, files = [], {}
    for network in networks:
        raw_dir = (
            _parquet_day_dir(day)
            if network == "adsblol"
            else _adsbx_parquet_day_dir(day)
        )
        raw = sorted(raw_dir.glob("icao_bucket=*/data.parquet"))
        if not raw:
            raise FileNotFoundError(f"No {network} observations for {day}")
        root = folder / network
        root.mkdir(exist_ok=True)
        spec = {
            "date": str(day),
            "network": network,
            "references": provenance,
            "inputs": [fingerprint(p) for p in raw],
            "code": [
                fingerprint(p)
                for p in [
                    Path(__file__),
                    Path(__file__).parents[2] / "flights_airport_nn/features.py",
                    Path(__file__).parents[2] / "flights_airport_nn/data.py",
                    Path(__file__).parents[2] / "flights_algorithm/main.py",
                    Path(__file__).parents[2] / "flights_algorithm/readsb_legs.py",
                    Path(__file__).parents[2] / "airports/airport_lookup.py",
                    OUTPUT_DIR / "data/raw/ourairports/airports.csv",
                ]
            ],
        }
        manifest = root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != spec:
            raise ValueError(
                f"Evaluation inputs changed; choose a new --output ({root})"
            )
        manifest.write_text(json.dumps(spec, indent=2) + "\n")
        files[network] = []
        for i, start in enumerate(range(0, len(aircraft), 256)):
            path = root / f"part-{i:04d}.npz"
            files[network].append(path)
            if not path.exists():
                tasks.append(
                    (
                        day,
                        network,
                        refs.filter(
                            pl.col("icao").is_in(aircraft[start : start + 256])
                        ),
                        path,
                    )
                )
    refs.write_parquet(folder / "reference-definitions.parquet")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(prepare_part, *task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"{i}/{len(tasks)} {future.result()}", flush=True)
    return refs, files


def summarize(frame):
    def rates(group):
        flights = group.group_by("reference_id").agg(
            pl.len().alias("ends"),
            pl.col("nn_correct").all(),
            pl.col("nearest_correct").all(),
        )
        flights = flights.filter(pl.col("ends") == 2)
        return {
            "endpoints": group.height,
            "airports": group["truth"].n_unique(),
            "nn_pct": 100 * group["nn_correct"].mean() if group.height else None,
            "nearest_pct": 100 * group["nearest_correct"].mean()
            if group.height
            else None,
            "complete_flights": flights.height,
            "nn_both_pct": 100 * flights["nn_correct"].mean()
            if flights.height
            else None,
            "nearest_both_pct": 100 * flights["nearest_correct"].mean()
            if flights.height
            else None,
            "candidate_recall_pct": 100
            * group.select((pl.col("target") >= 0).mean()).item()
            if group.height
            else None,
            "nn_only_correct": group.filter(
                pl.col("nn_correct") & ~pl.col("nearest_correct")
            ).height,
            "nearest_only_correct": group.filter(
                ~pl.col("nn_correct") & pl.col("nearest_correct")
            ).height,
            "both_wrong": group.filter(
                ~pl.col("nn_correct") & ~pl.col("nearest_correct")
            ).height,
        }

    primary = frame.filter(pl.col("quality_eligible"))
    return {
        "matched_endpoints": frame.height,
        "unresolved_or_conflicting_labels": frame.filter(~pl.col("eligible")).height,
        "primary": rates(primary),
        **{t: rates(primary.filter(pl.col("airport_type") == t)) for t in TYPES},
        "uncorroborated_agreement_only": rates(
            frame.filter(pl.col("eligible") & ~pl.col("quality_eligible"))
        ),
    }


def shared_quality(combined):
    """Apply the same evidence policy to both networks, without predictions."""
    quality = combined.group_by(KEY).agg(
        pl.col("ground_supported").any().alias("ground_supported_any_network")
    )
    return combined.join(quality, on=KEY).with_columns(
        (
            pl.col("eligible")
            & (pl.col("reported") | pl.col("ground_supported_any_network"))
        ).alias("quality_eligible")
    )


def run(day, networks, model_path, output=ROOT, workers=4):
    output = Path(output)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    used = (
        checkpoint["training"]["train_dates"]
        + checkpoint["training"]["validation_dates"]
    )
    if str(day) in used:
        raise ValueError("Evaluation day overlaps model development")
    _, files = prepare(day, networks, output, workers)
    torch.set_num_threads(2)
    model = load_model(model_path, "cuda" if torch.cuda.is_available() else "cpu")
    frames = [score(model, paths)[0] for paths in files.values()]
    combined = pl.concat(frames, how="diagonal_relaxed")
    # Corroboration may come from either network, but the reference and quality
    # policy are shared. Never select a cohort using NN/nearest correctness.
    combined = shared_quality(combined)
    folder = output / str(day)
    combined.write_parquet(folder / "endpoints.parquet")
    combined.filter(
        pl.col("quality_eligible")
        & (~pl.col("nn_correct") | ~pl.col("nearest_correct"))
    ).write_parquet(folder / "mismatches.parquet")
    summary = {
        "date": str(day),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "cohort_policy": "Same references; primary uses reported airports or ground corroboration in any evaluated network; temporal matches only",
        "results": {},
        "paired_networks": {},
    }
    statuses = []
    for network, paths in files.items():
        status = pl.concat(
            [
                pl.read_parquet(p.with_name(p.stem + "-references.parquet"))
                for p in paths
            ]
        )
        predicted = (
            combined.filter(pl.col("network") == network)
            .group_by("reference_id")
            .agg(
                pl.col("takeoff_time").first().alias("takeoff_time_predicted"),
                pl.col("landing_time").first().alias("landing_time_predicted"),
                *[
                    pl.col(method)
                    .filter(pl.col("endpoint") == end)
                    .first()
                    .alias(f"{end}_{method}")
                    for method in ("nn", "nearest")
                    for end in ("takeoff", "landing")
                ],
            )
        )
        statuses.append(
            status.join(predicted, on="reference_id", how="left").with_columns(
                pl.lit(network).alias("network"),
                pl.when(pl.col("matched"))
                .then(pl.lit("both"))
                .otherwise(pl.lit("reference_only"))
                .alias("match_status"),
            )
        )
        summary["results"][network] = {}
        for source in ("original", "broad"):
            frame = combined.filter(
                (pl.col("network") == network) & (pl.col("source") == source)
            )
            scoped = status.filter(pl.col("source") == source)
            summary["results"][network][source] = {
                **summarize(frame),
                "reference_flights": scoped.height,
                "matched_flights": scoped["matched"].sum(),
                "overlapping_references": scoped["time_conflict"].sum(),
                "match_pct": 100 * scoped["matched"].mean() if scoped.height else None,
            }
    if len(networks) > 1:
        shared = (
            combined.group_by(KEY)
            .agg(pl.col("network").n_unique().alias("networks"))
            .filter(pl.col("networks") == len(networks))
            .select(KEY)
        )
        paired = combined.join(shared, on=KEY)
        paired.write_parquet(folder / "paired-endpoints.parquet")
        for network in networks:
            summary["paired_networks"][network] = {
                source: summarize(
                    paired.filter(
                        (pl.col("network") == network) & (pl.col("source") == source)
                    )
                )
                for source in ("original", "broad")
            }
    pl.concat(statuses).write_parquet(folder / "references.parquet")
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=["adsblol", "adsbx"],
        default=["adsblol", "adsbx"],
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.date, args.networks, args.model, args.output, args.workers)
