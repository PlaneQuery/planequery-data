"""Conservative track corroboration of frozen-evaluation disagreements.

These are evidence categories, not corrected ground truth or accuracy estimates.
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data_engineering.flights.flight_type import flights_algorithm_output_path
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from flights_airport_nn.broader_evaluation import (
    load_catalog,
    match_by_time,
    read_endpoint_messages,
)
from planequery_data.utils import haversine


def airport_evidence(points, airport):
    """Require sustained slow, near-ground observations near the airport center."""
    empty = {
        "strong": False,
        "supporting_points": 0,
        "supporting_span_s": 0.0,
        "closest_km": None,
        "minimum_abs_height_ft": None,
    }
    if not airport or points.is_empty():
        return empty
    lat, lon = float(airport["latitude_deg"]), float(airport["longitude_deg"])
    distance = np.array(
        [haversine(p["lat"], p["lon"], lat, lon) for p in points.iter_rows(named=True)]
    )
    elevation = airport.get("elevation_ft")
    altitude = points["baro_altitude_ft"].cast(pl.Float64).to_numpy()
    height = (
        np.abs(altitude - float(elevation))
        if elevation
        else np.full(len(distance), np.nan)
    )
    speed = points["ground_speed_kt"].cast(pl.Float64).to_numpy()
    ground = points["on_ground"].fill_null(False).to_numpy().astype(bool)
    support = (distance <= 2) & (ground | ((height <= 300) & (speed <= 100)))
    times = points.filter(pl.Series(support))["time"]
    span = (times.max() - times.min()).total_seconds() if len(times) else 0.0
    return {
        "strong": bool(support.sum() >= 3 and span >= 10),
        "supporting_points": int(support.sum()),
        "supporting_span_s": span,
        "closest_km": float(distance.min()),
        "minimum_abs_height_ft": float(np.nanmin(height))
        if np.isfinite(height).any()
        else None,
    }


def classify(truth, prediction, other, separation, other_separation):
    if truth["strong"] and (prediction["strong"] or other["strong"]):
        return "conflicting_track_evidence"
    if truth["strong"] and separation is not None and separation >= 10:
        return "likely_model_error"
    if not truth["strong"] and (
        (prediction["strong"] and separation is not None and separation >= 10)
        or (other["strong"] and other_separation is not None and other_separation >= 10)
    ):
        return "likely_label_or_segment_error"
    return "unresolved"


def separation(catalog, a, b):
    if a not in catalog or b not in catalog:
        return None
    x, y = catalog[a], catalog[b]
    return haversine(
        float(x["latitude_deg"]),
        float(x["longitude_deg"]),
        float(y["latitude_deg"]),
        float(y["longitude_deg"]),
    )


def run(output, prediction_file=None):
    output = Path(output)
    catalog = load_catalog()
    predictions = pl.read_parquet(
        prediction_file or output / "predictions.parquet"
    ).filter((pl.col("coverage_loss_s") == 0) & pl.col("scorable"))
    errors = predictions.filter(~pl.col("correct"))
    audit_dir = output / "audit"
    audit_dir.mkdir(exist_ok=True)
    policy = {
        "purpose": "Track corroboration only; no relabeling or adjusted accuracy.",
        "endpoint_window_seconds": 120,
        "radius_km": 2,
        "maximum_absolute_height_ft": 300,
        "maximum_speed_kt": 100,
        "on_ground_alternative": True,
        "minimum_points": 3,
        "minimum_span_seconds": 10,
        "minimum_airport_separation_km": 10,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Same ADS-B source as model; corroboration is not independent verification.",
            "Airport centers approximate runway locations; barometric altitude is not corrected for local pressure.",
            "Low passes, diversions, incorrect ICAO association and segment boundaries can remain ambiguous.",
            "Missing corroboration does not mean a label is false.",
            "Likely label or segment error does not imply the model prediction is correct.",
            "Source flight caches are available for all dates; original FAA messages are not available for every date.",
        ],
    }
    (audit_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    records, source_records = [], []
    for day_text in sorted(errors["date"].unique()):
        day = date.fromisoformat(day_text)
        day_errors = errors.filter(pl.col("date") == day_text).sort(
            "flight_id", "endpoint"
        )
        cache = audit_dir / f"{day}-audited.parquet"
        source_cache = audit_dir / f"{day}-source-matches.parquet"
        manifest_path = audit_dir / f"{day}-manifest.json"
        manifest = {
            "policy": policy,
            "predictions_sha256": hashlib.sha256(
                day_errors.write_json().encode()
            ).hexdigest(),
        }
        if (
            cache.exists()
            and source_cache.exists()
            and manifest_path.exists()
            and json.loads(manifest_path.read_text()) == manifest
        ):
            records.extend(pl.read_parquet(cache).to_dicts())
            source_records.extend(pl.read_parquet(source_cache).to_dicts())
            print(f"{day}: reused verified audit cache", flush=True)
            continue
        record_start, source_start = len(records), len(source_records)
        labels = pl.read_parquet(sfdps_flights_parquet_path(day))
        segments = pl.read_parquet(
            flights_algorithm_output_path(day, no_airports_model=True)
        )
        matched, _ = match_by_time(segments, labels)
        matched = matched.filter(
            pl.col("flight_id").is_in(day_errors["flight_id"].unique().implode())
        )
        labels.filter(
            pl.col("icao").is_in(matched["icao"].unique().implode())
        ).write_parquet(audit_dir / f"{day}-cached-source-flights.parquet")
        meta = {r["flight_id"]: r for r in matched.iter_rows(named=True)}
        print(
            f"{day}: auditing {day_errors.height} disagreements from {matched.height} flights",
            flush=True,
        )
        messages = read_endpoint_messages(day, matched).filter(
            pl.col("lat").is_finite()
            & pl.col("lon").is_finite()
            & (pl.col("lat").abs() <= 90)
            & (pl.col("lon").abs() <= 180)
        )
        aircraft = messages.partition_by("icao", as_dict=True)
        for r in day_errors.iter_rows(named=True):
            flight = meta[r["flight_id"]]
            boundary = flight[
                "takeoff_time" if r["endpoint"] == "takeoff" else "landing_time"
            ]
            track = (
                aircraft.get((r["icao"],), messages.clear())
                .filter(
                    (pl.col("time") >= flight["takeoff_time"])
                    & (pl.col("time") <= flight["landing_time"])
                    & (
                        (pl.col("time") - boundary).dt.total_milliseconds().abs()
                        <= 120_000
                    )
                )
                .sort("time")
            )
            truth = airport_evidence(track, catalog.get(r["truth"]))
            prediction = airport_evidence(track, catalog.get(r["prediction"]))
            third = (
                r["nearest"]
                if r["nearest"] not in (r["truth"], r["prediction"])
                else ""
            )
            other = airport_evidence(track, catalog.get(third))
            sep = separation(catalog, r["truth"], r["prediction"])
            other_sep = separation(catalog, r["truth"], third)
            record = dict(
                r,
                audit_class=classify(truth, prediction, other, sep, other_sep),
                terminal_observations=track.height,
                airport_separation_km=sep,
                third_airport=third,
                takeoff_time=flight["takeoff_time"],
                landing_time=flight["landing_time"],
            )
            for prefix, evidence in [
                ("truth", truth),
                ("prediction", prediction),
                ("third", other),
            ]:
                record.update({f"{prefix}_{k}": v for k, v in evidence.items()})
            records.append(record)
            source_records.append(
                {
                    k: v
                    for k, v in flight.items()
                    if k
                    in (
                        "icao",
                        "flight_id",
                        "takeoff_time",
                        "landing_time",
                        "raw_takeoff_label",
                        "raw_landing_label",
                        "label_callsign",
                        "start_delta_s",
                        "end_delta_s",
                    )
                }
            )
        # Save exact observed track evidence for reproducible review of every disagreement.
        messages.write_parquet(audit_dir / f"{day}-endpoint-observations.parquet")
        pl.DataFrame(records[record_start:], infer_schema_length=None).write_parquet(
            cache
        )
        pl.DataFrame(
            source_records[source_start:], infer_schema_length=None
        ).write_parquet(source_cache)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    audited = pl.DataFrame(records, infer_schema_length=None)
    audited.write_parquet(audit_dir / "disagreements.parquet")
    audited.write_csv(audit_dir / "disagreements.csv")
    pl.DataFrame(source_records, infer_schema_length=None).unique().write_parquet(
        audit_dir / "source-matches.parquet"
    )
    counts = (
        audited.group_by("audit_class")
        .agg(pl.len().alias("endpoints"))
        .sort("audit_class")
    )
    by_type = (
        audited.group_by("airport_type", "audit_class")
        .agg(pl.len().alias("endpoints"))
        .sort("airport_type", "audit_class")
    )
    summary = {
        "total_disagreements": audited.height,
        "counts": counts.to_dicts(),
        "by_airport_type": by_type.to_dicts(),
        "policy": policy,
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Disagreement audit",
        "",
        (
            "All normal-coverage disagreements were screened using the same conservative evidence rule. "
            "These categories are tentative; the original agreement scores remain unchanged."
        ),
        "",
        (
            "Sustained evidence: at least three observations spanning ten seconds within two minutes of the "
            "segment boundary, within 2 km of the airport center, and either on ground or within 300 ft "
            "of field elevation at no more than 100 kt. Competing airports must be at least 10 km apart."
        ),
        "",
        "| Evidence category | Endpoints |",
        "|---|---:|",
    ]
    lines += [f"| {r['audit_class']} | {r['endpoints']:,} |" for r in counts.to_dicts()]
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Likely model error:** sustained evidence supports the labeled airport and not the predicted airport.",
        "- **Likely label or segment error:** sustained evidence supports the prediction or a third airport, not the label. This does not automatically vindicate the model.",
        "- **Conflicting:** more than one airport has supporting observations.",
        "- **Unresolved:** evidence is insufficient or the airports are too close for this rule.",
        "",
        "## Limits",
        "",
    ] + ["- " + s for s in policy["limitations"]]
    lines += [
        "",
        (
            "See `disagreements.csv` for every case and measured evidence, `source-matches.parquet` "
            "for cached FAA labels and time matching, and daily `*-endpoint-observations.parquet` for raw ADS-B observations."
        ),
        "",
    ]
    (audit_dir / "results.md").write_text("\n".join(lines))
    print(json.dumps(summary["counts"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional completed subset for incremental audit",
    )
    args = parser.parse_args()
    run(args.evaluation_dir, args.predictions)
