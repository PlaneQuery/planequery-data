"""Track corroboration where the broad network beats nearest on a held-out test."""

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.flights.flight_type import flights_algorithm_output_path
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from flights_airport_nn.broader_evaluation import (
    load_catalog,
    match_by_time,
    read_endpoint_messages,
)
from flights_airport_nn.error_audit import airport_evidence, separation


def run(evaluation_dir):
    root = Path(evaluation_dir)
    out = root / "audit"
    out.mkdir(exist_ok=True)
    all_predictions = pl.read_parquet(root / "predictions.parquet")
    normal = all_predictions.filter(
        (pl.col("coverage_loss_s") == 0) & pl.col("scorable")
    )
    wins = normal.filter(pl.col("correct") & ~pl.col("nearest_correct"))
    catalog = load_catalog()
    rows = []
    for day_text in sorted(wins["date"].unique()):
        day = date.fromisoformat(day_text)
        part = wins.filter(pl.col("date") == day_text)
        labels = pl.read_parquet(sfdps_flights_parquet_path(day))
        segments = pl.read_parquet(
            flights_algorithm_output_path(day, no_airports_model=True)
        )
        matched, _ = match_by_time(segments, labels)
        matched = matched.filter(
            pl.col("flight_id").is_in(part["flight_id"].unique().implode())
        )
        meta = {r["flight_id"]: r for r in matched.iter_rows(named=True)}
        print(day, "auditing", part.height, "network-only agreements", flush=True)
        messages = read_endpoint_messages(day, matched).filter(
            pl.col("lat").is_finite()
            & pl.col("lon").is_finite()
            & (pl.col("lat").abs() <= 90)
            & (pl.col("lon").abs() <= 180)
        )
        aircraft = messages.partition_by("icao", as_dict=True)
        for r in part.iter_rows(named=True):
            f = meta[r["flight_id"]]
            boundary = f[
                "takeoff_time" if r["endpoint"] == "takeoff" else "landing_time"
            ]
            track = (
                aircraft.get((r["icao"],), messages.clear())
                .filter(
                    (pl.col("time") >= f["takeoff_time"])
                    & (pl.col("time") <= f["landing_time"])
                    & (
                        (pl.col("time") - boundary).dt.total_milliseconds().abs()
                        <= 120_000
                    )
                )
                .sort("time")
            )
            truth = airport_evidence(track, catalog.get(r["truth"]))
            nearest = airport_evidence(track, catalog.get(r["nearest"]))
            sep = separation(catalog, r["truth"], r["nearest"])
            if truth["strong"] and nearest["strong"]:
                category = "conflicting_track_evidence"
            elif truth["strong"] and sep is not None and sep >= 10:
                category = "supports_network_label"
            elif nearest["strong"] and sep is not None and sep >= 10:
                category = "supports_nearest_despite_label"
            else:
                category = "unresolved"
            row = dict(
                r,
                category=category,
                terminal_observations=track.height,
                airport_separation_km=sep,
                takeoff_time=f["takeoff_time"],
                landing_time=f["landing_time"],
            )
            row.update({f"truth_{k}": v for k, v in truth.items()})
            row.update({f"nearest_{k}": v for k, v in nearest.items()})
            rows.append(row)
        messages.write_parquet(out / f"{day}-winner-observations.parquet")
    audited = pl.DataFrame(rows, infer_schema_length=None)
    audited.write_parquet(out / "network-only-wins.parquet")
    audited.write_csv(out / "network-only-wins.csv")
    summary = {
        "winner_endpoints": wins.height,
        "categories": audited.group_by("category").len().sort("category").to_dicts(),
        "by_airport_type": audited.group_by("airport_type", "category")
        .len()
        .sort("airport_type", "category")
        .to_dicts(),
        "evidence_rule": "Three points over at least 10 seconds within 2 km and within 300 ft/100 kt or on ground; airport separation at least 10 km; first or last two minutes.",
        "limitation": "Same ADS-B source as the network. Evidence is corroboration, not independent proof, and unsupported or closely spaced airports remain unresolved. No score is relabeled.",
    }
    (out / "network-win-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    run(parser.parse_args().evaluation_dir)
