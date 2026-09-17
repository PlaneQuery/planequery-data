"""Frozen-model assessment on broader, single-source FAA labels and reception loss."""

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from airports.airport_lookup import AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import (
    flights_algorithm_output_path,
    with_flight_schema_columns,
)
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.benchmark import summarize
from flights_airport_nn.dataset import cache_path
from flights_airport_nn.features.features import FEATURE_PERIOD, create_features
from flights_airport_nn.inference import load_model, predict_endpoints
from planequery_data.utils import haversine

ADSB_COLUMNS = [
    "time",
    "icao",
    "lat",
    "lon",
    "baro_altitude_ft",
    "ground_speed_kt",
    "track_deg",
    "baro_rate_fpm",
    "on_ground",
]
REGISTRATION_CALLSIGN = re.compile(
    r"^(?:N[1-9][0-9]{0,4}(?:[A-HJ-NP-Z]{1,2})?|[A-Z]{1,3}-[A-Z0-9]{2,5})$"
)


def match_by_time(segments, labels, tolerance_seconds=900):
    """Require one-to-one temporal matches; neither airport nor position enters matching."""
    labels = labels.filter(
        pl.col("icao").is_not_null()
        & pl.col("takeoff_time").is_not_null()
        & pl.col("landing_time").is_not_null()
        & (pl.col("landing_time") > pl.col("takeoff_time"))
    ).unique(
        subset=[
            "icao",
            "takeoff_time",
            "landing_time",
            "takeoff_airport_ident",
            "landing_airport_ident",
        ],
        maintain_order=True,
    )
    labels = with_flight_schema_columns(labels).with_row_index("_label_id")
    segments = with_flight_schema_columns(segments).with_row_index("_segment_id")
    pairs = (
        segments.select("_segment_id", "icao", "takeoff_time", "landing_time")
        .join(
            labels.select(
                "_label_id",
                "icao",
                pl.col("takeoff_time").alias("_label_start"),
                pl.col("landing_time").alias("_label_end"),
            ),
            on="icao",
            how="inner",
        )
        .with_columns(
            (pl.col("takeoff_time") - pl.col("_label_start"))
            .dt.total_milliseconds()
            .abs()
            .truediv(1000)
            .alias("start_delta_s"),
            (pl.col("landing_time") - pl.col("_label_end"))
            .dt.total_milliseconds()
            .abs()
            .truediv(1000)
            .alias("end_delta_s"),
        )
        .filter(
            (pl.col("start_delta_s") <= tolerance_seconds)
            & (pl.col("end_delta_s") <= tolerance_seconds)
        )
    )
    candidates = pairs.height
    pairs = pairs.filter(
        (pl.len().over("_segment_id") == 1) & (pl.len().over("_label_id") == 1)
    )
    label_metadata = labels.select(
        "_label_id",
        pl.col("takeoff_airport_ident").alias("raw_takeoff_label"),
        pl.col("landing_airport_ident").alias("raw_landing_label"),
        pl.col("callsign").alias("label_callsign"),
        pl.col("category").alias("label_category"),
        pl.col("aircraft_type").alias("label_aircraft_type"),
    )
    matched = segments.join(
        pairs.select("_segment_id", "_label_id", "start_delta_s", "end_delta_s"),
        on="_segment_id",
    ).join(label_metadata, on="_label_id")
    matched = matched.with_columns(
        (
            pl.col("icao") + "_" + pl.col("takeoff_time").dt.strftime("%Y-%m-%d_%H-%M")
        ).alias("flight_id")
    )
    collisions = matched.filter(pl.len().over("flight_id") > 1).height
    matched = matched.filter(pl.len().over("flight_id") == 1).sort(
        ["icao", "takeoff_time"]
    )
    return matched, {
        "label_flights": labels.height,
        "segmented_flights": segments.height,
        "temporal_candidate_pairs": candidates,
        "unambiguous_pairs": pairs.height,
        "excluded_flight_id_collisions": collisions,
        "matched_flights": matched.height,
    }


def load_catalog():
    with (OUTPUT_DIR / "data/raw/ourairports/airports.csv").open() as stream:
        return {r["ident"]: r for r in csv.DictReader(stream)}


def resolve_labels(codes, catalog):
    lookup = AirportLookup()
    result = {}
    for code in codes:
        if code in catalog:
            result[code] = code
            continue
        try:
            airport = lookup.get_airport_from_code(code) if code else None
        except ValueError:
            airport = None
        result[code] = airport.ident if airport else None
    return result


def stress_selected(flight_id):
    return int(hashlib.sha256(flight_id.encode()).hexdigest(), 16) % 10 == 0


def mask_reception(flights, messages, seconds):
    """Remove the first/last seconds of each segment; retain original segment times."""
    if seconds == 0:
        return messages
    windows = flights.select(
        "icao",
        (pl.col("takeoff_time") + pl.duration(seconds=seconds)).alias(
            "_observed_start"
        ),
        (pl.col("landing_time") - pl.duration(seconds=seconds)).alias("_observed_end"),
    )
    return (
        messages.join(windows, on="icao")
        .filter(
            (pl.col("time") >= pl.col("_observed_start"))
            & (pl.col("time") <= pl.col("_observed_end"))
        )
        .select(messages.columns)
        .unique(subset=["icao", "time"], maintain_order=True)
    )


def endpoint_windows(flights):
    """Merge overlaps so as-of matching retains every requested endpoint window."""
    windows = []
    for key, group in flights.partition_by("icao", as_dict=True).items():
        intervals = []
        for row in group.iter_rows(named=True):
            start, end = row["takeoff_time"], row["landing_time"]
            intervals += [
                (start, min(end, start + FEATURE_PERIOD)),
                (max(start, end - FEATURE_PERIOD), end),
            ]
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        windows += [(key[0], start, end) for start, end in merged]
    return pl.DataFrame(
        windows,
        schema={
            "icao": pl.String,
            "_window_start": pl.Datetime("ms"),
            "_window_end": pl.Datetime("ms"),
        },
        orient="row",
    )


def read_endpoint_messages(day, flights, source="adsblol"):
    windows = endpoint_windows(flights).sort(["icao", "_window_start"])
    messages = read_adsb(
        day,
        icaos=flights["icao"].unique().to_list(),
        columns=ADSB_COLUMNS,
        source=source,
        lazy=True,
    )
    return (
        messages.sort(["icao", "time"])
        .join_asof(
            windows.lazy(),
            left_on="time",
            right_on="_window_start",
            by="icao",
            strategy="backward",
            check_sortedness=False,
        )
        .filter(pl.col("time") <= pl.col("_window_end"))
        .select(ADSB_COLUMNS)
        .collect(engine="streaming")
    )


def score_batch(
    day,
    frame,
    messages,
    model,
    config,
    catalog,
    resolved,
    seen_aircraft,
    seen_airports,
    loss_seconds=0,
):
    # Labels stay outside the model's flight objects, including during candidate retrieval.
    unlabelled = with_flight_schema_columns(frame).with_columns(
        pl.lit(None, dtype=pl.String).alias("takeoff_airport_ident"),
        pl.lit(None, dtype=pl.String).alias("landing_airport_ident"),
    )
    features = create_features(unlabelled, messages)
    predictions = predict_endpoints(features, model, config)
    metadata = {r["flight_id"]: r for r in frame.iter_rows(named=True)}
    by_aircraft = messages.filter(
        pl.col("lat").is_finite()
        & pl.col("lon").is_finite()
        & (pl.col("lat").abs() <= 90)
        & (pl.col("lon").abs() <= 180)
    ).partition_by("icao", as_dict=True)
    endpoints = {(f.flight_id, f.endpoint): f for f in features.endpoints}
    records = []
    for row in predictions:
        flight_id, endpoint = row["flight_id"], row["endpoint"]
        meta = metadata[flight_id]
        truth_code = meta[f"raw_{endpoint}_label"]
        truth = resolved.get(truth_code)
        airport = catalog.get(truth, {})
        candidate_set = endpoints[flight_id, endpoint].candidates
        nearest = (
            min(
                candidate_set, key=lambda c: float(c.features[0, 4])
            ).predicted_airport.ident
            if candidate_set
            else ""
        )
        sample = by_aircraft.get((meta["icao"],))
        point = None
        if sample is not None:
            boundary = (
                meta["takeoff_time"] if endpoint == "takeoff" else meta["landing_time"]
            )
            sample = sample.filter(
                (pl.col("time") >= meta["takeoff_time"])
                & (pl.col("time") <= meta["landing_time"])
                & (
                    (pl.col("time") - boundary).dt.total_milliseconds().abs()
                    <= FEATURE_PERIOD.total_seconds() * 1000
                )
            ).sort("time", descending=endpoint == "landing")
            if sample.height:
                point = sample.row(0, named=True)
        truth_distance = None
        height = None
        nearest_distance = None
        if point:
            # Baseline queries the entire supported catalog, even when retrieval is empty.
            nearest = (
                AirportLookup().getClosestAirport(point["lat"], point["lon"]).ident
            )
            if airport:
                truth_distance = haversine(
                    point["lat"],
                    point["lon"],
                    float(airport["latitude_deg"]),
                    float(airport["longitude_deg"]),
                )
                if point["baro_altitude_ft"] is not None and airport.get(
                    "elevation_ft"
                ):
                    height = point["baro_altitude_ft"] - float(airport["elevation_ft"])
            if nearest:
                a = catalog[nearest]
                nearest_distance = haversine(
                    point["lat"],
                    point["lon"],
                    float(a["latitude_deg"]),
                    float(a["longitude_deg"]),
                )
        prediction = row["airport_ident"] or ""
        records.append(
            {
                "date": day.isoformat(),
                "flight_id": flight_id,
                "icao": meta["icao"],
                "endpoint": endpoint,
                "coverage_loss_s": loss_seconds,
                "stress_sample": stress_selected(flight_id),
                "label_source": "SFDPS",
                "raw_label": truth_code,
                "truth": truth or "",
                "scorable": bool(truth),
                "airport_type": airport.get("type", "unresolved"),
                "country": airport.get("iso_country", ""),
                "prediction": prediction,
                "prediction_type": catalog.get(prediction, {}).get("type", ""),
                "nearest": nearest,
                "correct": bool(truth and prediction == truth),
                "nearest_correct": bool(truth and nearest == truth),
                "retrieved": bool(
                    truth
                    and any(c.predicted_airport.ident == truth for c in candidate_set)
                ),
                "top3": bool(
                    truth and truth in [a["airport_ident"] for a in row["alternatives"]]
                ),
                "confidence": row["confidence"],
                "distance_km": nearest_distance,
                "truth_distance_km": truth_distance,
                "height_above_truth_ft": height,
                "ground_speed_kt": point["ground_speed_kt"] if point else None,
                "observation_available": point is not None,
                "observation_lat": point["lat"] if point else None,
                "observation_lon": point["lon"] if point else None,
                "start_delta_s": meta["start_delta_s"],
                "end_delta_s": meta["end_delta_s"],
                "registration_callsign": bool(
                    REGISTRATION_CALLSIGN.fullmatch(
                        (meta["label_callsign"] or "").strip()
                    )
                ),
                "callsign": meta["label_callsign"],
                "category": meta["label_category"],
                "aircraft_type": meta["label_aircraft_type"],
                "seen_aircraft": meta["icao"] in seen_aircraft,
                "seen_airport": truth in seen_airports,
            }
        )
    return pl.DataFrame(records, infer_schema_length=None)


def group_metrics(frame):
    if frame.is_empty():
        return {"endpoints": 0}
    stats = summarize(frame)["overall"]
    stats.update(
        errors=frame.filter(~pl.col("correct")).height,
        nearest_errors=frame.filter(~pl.col("nearest_correct")).height,
        flights=frame["flight_id"].n_unique(),
        airports=frame["truth"].n_unique(),
        high_confidence_errors=frame.filter(
            (~pl.col("correct")) & (pl.col("confidence") >= 0.9)
        ).height,
    )
    return stats


def summarize_broader(predictions, matching):
    normal = predictions.filter(pl.col("coverage_loss_s") == 0)
    scored = normal.filter(pl.col("scorable"))
    groups = {
        "all_resolved_labels": pl.lit(True),
        "registration_callsign_proxy": pl.col("registration_callsign"),
        "other_callsigns": ~pl.col("registration_callsign"),
        "small_airports": pl.col("airport_type") == "small_airport",
        "medium_airports": pl.col("airport_type") == "medium_airport",
        "large_airports": pl.col("airport_type") == "large_airport",
        "unsupported_airport_types": ~pl.col("airport_type").is_in(
            ["small_airport", "medium_airport", "large_airport"]
        ),
        "non_us_airports": pl.col("country") != "US",
        "unseen_aircraft": ~pl.col("seen_aircraft"),
        "unseen_airports": ~pl.col("seen_airport"),
        "temporal_match_within_5min": (pl.col("start_delta_s") <= 300)
        & (pl.col("end_delta_s") <= 300),
        "nearest_within_3km": pl.col("distance_km") <= 3,
        "small_with_nearby_low_endpoint": (pl.col("airport_type") == "small_airport")
        & (pl.col("truth_distance_km") <= 3)
        & (pl.col("height_above_truth_ft").abs() <= 1000),
    }
    result = {
        "matching": matching,
        "unresolved_label_endpoints": normal.filter(~pl.col("scorable")).height,
        "groups": {
            name: group_metrics(scored.filter(condition))
            for name, condition in groups.items()
        },
        "by_date": {
            str(key[0]): group_metrics(df)
            for key, df in scored.partition_by("date", as_dict=True).items()
        },
        "by_country": {
            str(key[0]): group_metrics(df)
            for key, df in scored.partition_by("country", as_dict=True).items()
        },
        "by_endpoint": {
            str(key[0]): group_metrics(df)
            for key, df in scored.partition_by("endpoint", as_dict=True).items()
        },
    }
    result["stress"] = {
        str(seconds): group_metrics(
            predictions.filter(
                pl.col("scorable")
                & pl.col("stress_sample")
                & (pl.col("coverage_loss_s") == seconds)
            )
        )
        for seconds in (0, 120, 300)
    }
    result["high_confidence"] = {}
    for threshold in (0.9, 0.99):
        confident = scored.filter(pl.col("confidence") >= threshold)
        result["high_confidence"][str(threshold)] = {
            **group_metrics(confident),
            "coverage": confident.height / max(1, scored.height),
        }
    small_wrong = scored.filter(
        (pl.col("airport_type") == "small_airport") & ~pl.col("correct")
    )
    result["small_airport_errors_by_prediction_type"] = {
        str(k[0]): df.height
        for k, df in small_wrong.partition_by("prediction_type", as_dict=True).items()
    }
    result["limitations"] = [
        "Agreement with single-source SFDPS labels, not independently verified airport correctness.",
        "Unique aircraft/time matches within 15 minutes at both ends; unmatched or ambiguous flights are excluded and counted in matching statistics.",
        "No airport-proximity or BTS-agreement filter was applied by this evaluation; upstream cached SFDPS data may have its own selection.",
        "Registration callsigns are a general-aviation/charter proxy, not a definitive operational classification.",
        "FAA coverage and cached labels limit geographic scope; this does not establish worldwide or VFR-only performance.",
        "Reception-loss tests mask the first and last 2/5 minutes of segments on a deterministic 10% sample, retaining segment times and regenerating candidates.",
        "Airport-proximity/low-altitude sensitivity subsets are conditional checks, not unbiased overall accuracy estimates.",
        "Binomial intervals ignore flight, aircraft, airport and day correlations.",
    ]
    return result


def write_report(output, metrics):
    lines = [
        "# Broader performance of the frozen airport network",
        "",
        "This evaluation measures agreement with later FAA SFDPS records. The model, candidate retrieval, and confidence temperature were frozen. No training or model selection used these results.",
        "",
        "| Cohort | Endpoints | NN agreement | Nearest agreement | Candidate recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in metrics["groups"].items():
        if row["endpoints"]:
            lines.append(
                f"| {name.replace('_', ' ')} | {row['endpoints']:,} | {row['accuracy']:.2%} | {row['nearest_accuracy']:.2%} | {row['candidate_recall']:.2%} |"
            )
    lines += [
        "",
        "## Reception-loss sensitivity",
        "",
        "The same sampled endpoints are used at each masking level.",
        "",
        "| Mask at each end | Endpoints | NN agreement | Nearest agreement | Candidate recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for seconds, row in metrics["stress"].items():
        if row["endpoints"]:
            lines.append(
                f"| {int(seconds) // 60} minutes | {row['endpoints']:,} | {row['accuracy']:.2%} | {row['nearest_accuracy']:.2%} | {row['candidate_recall']:.2%} |"
            )
    lines += ["", "## Interpretation limits", ""] + [
        "- " + s for s in metrics["limitations"]
    ]
    lines += [
        "",
        f"Unresolved airport-label endpoints excluded from agreement scoring: {metrics['unresolved_label_endpoints']:,}.",
        "",
        (
            "See `metrics.json` for matching coverage, country/date groups and confidence checks; "
            "`predictions.parquet` contains all predictions, including unresolved labels and simulated reception loss; "
            "`errors.csv` contains normal-coverage disagreements."
        ),
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))


def run(model_path, dates, output, batch_aircraft=128):
    model_path, output = Path(model_path), Path(output)
    frozen_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model, config = load_model(model_path)
    torch.set_num_threads(4)
    used = set(model.training_dates) | set(model.validation_dates)
    previous_metrics = model_path.parent / "test_metrics.json"
    if previous_metrics.exists():
        used.update(
            date.fromisoformat(d)
            for d in json.loads(previous_metrics.read_text()).get("test_dates", [])
        )
    if used & set(dates):
        raise ValueError(
            "Use dates outside the original training, validation and test periods"
        )
    if batch_aircraft < 1 or len(set(dates)) != len(dates):
        raise ValueError("Require positive batch size and distinct dates")
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "model": str(model_path),
        "model_sha256": frozen_hash,
        "dates": [d.isoformat() for d in dates],
        "batch_aircraft": batch_aircraft,
        "time_tolerance_seconds": 900,
        "stress_seconds": [0, 120, 300],
        "stress_sample_modulus": 10,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if (output / "config.json").exists() and json.loads(
        (output / "config.json").read_text()
    ) != spec:
        raise ValueError(
            "Output directory belongs to a different evaluation configuration"
        )
    (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
    catalog = load_catalog()
    seen_aircraft = set()
    seen_airports = set()
    for day in model.training_dates:
        with np.load(cache_path(day), allow_pickle=False) as cache:
            seen_aircraft.update(s.split("_")[0] for s in cache["flight_id"])
            seen_airports.update(cache["truth"])
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
        print(day, stats, flush=True)
        resolved = resolve_labels(
            set(matched["raw_takeoff_label"]) | set(matched["raw_landing_label"]),
            catalog,
        )
        icaos = matched["icao"].unique().sort().to_list()
        print(
            f"{day}: reading endpoint observations once for {len(icaos)} aircraft",
            flush=True,
        )
        day_messages = read_endpoint_messages(day, matched)
        print(
            f"{day}: loaded {day_messages.height:,} endpoint observations", flush=True
        )
        for start in range(0, len(icaos), batch_aircraft):
            part = output / f"{day}-part-{start // batch_aircraft:04}.parquet"
            if part.exists():
                continue
            selected = matched.filter(
                pl.col("icao").is_in(icaos[start : start + batch_aircraft])
            )
            messages = day_messages.filter(
                pl.col("icao").is_in(icaos[start : start + batch_aircraft])
            )
            results = [
                score_batch(
                    day,
                    selected,
                    messages,
                    model,
                    config,
                    catalog,
                    resolved,
                    seen_aircraft,
                    seen_airports,
                )
            ]
            stress = selected.filter(
                pl.Series([stress_selected(s) for s in selected["flight_id"]])
            )
            if stress.height:
                for seconds in (120, 300):
                    masked = mask_reception(stress, messages, seconds)
                    results.append(
                        score_batch(
                            day,
                            stress,
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
            pl.concat(results, how="diagonal_relaxed").write_parquet(
                part.with_suffix(".tmp")
            )
            part.with_suffix(".tmp").replace(part)
            print(
                f"{day}: aircraft {min(start + batch_aircraft, len(icaos))}/{len(icaos)}, flights {selected.height}",
                flush=True,
            )
        del day_messages
    predictions = pl.concat(
        [pl.read_parquet(p) for p in sorted(output.glob("*-part-*.parquet"))],
        how="diagonal_relaxed",
    )
    predictions.write_parquet(output / "predictions.parquet")
    metrics = summarize_broader(predictions, matching)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == frozen_hash
    metrics["model_unchanged"] = True
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.filter(
        pl.col("scorable") & (pl.col("coverage_loss_s") == 0) & ~pl.col("correct")
    ).sort("confidence", descending=True).write_csv(output / "errors.csv")
    write_report(output, metrics)
    print(json.dumps(metrics["groups"], indent=2), flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-aircraft", type=int, default=128)
    args = parser.parse_args(argv)
    run(args.model, args.dates, args.output_dir, args.batch_aircraft)


if __name__ == "__main__":
    main()
