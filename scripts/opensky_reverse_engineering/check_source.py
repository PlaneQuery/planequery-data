"""Check the src implementation against the saved research holdout.

Reads the existing sampled state cache, never published flight boundaries as
algorithm input. Both catalogs are checked: frozen research catalog for port
parity, shared dated catalog for the source's actual airport behavior. Results
use arrival-day cohorts so they are comparable to the original experiment.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from segments import evaluate

from airports.catalog import AirportCatalogView
from airports.runtime import airport_catalog_for_date
from data_engineering.opensky.reverse_eng_states import STATE_SCHEMA, valid_position
from flights.opensky_reverse_eng import EndpointAirports, reconstruct_arrival_day


class FrozenHistory:
    def __init__(self, path):
        self.frame = pl.read_parquet(path)
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def as_of(self, day):
        return self.frame

    def source_commit_sha(self, day):
        return self.sha

    def aliases_through(self, start, end):
        return pl.DataFrame(
            schema={"airport_id": pl.Int64, "code": pl.String, "code_type": pl.String}
        )


def exported(flights, catalog):
    gps = {airport.id: airport.gps_code for airport in catalog.airports}
    return flights.select(
        pl.col("icao").alias("icao24"),
        pl.col("takeoff_time").dt.epoch("s").alias("firstseen"),
        pl.col("landing_time").dt.epoch("s").alias("lastseen"),
        "callsign",
        pl.col("takeoff_airport_id")
        .replace_strict(gps, default=None, return_dtype=pl.String)
        .alias("estdepartureairport"),
        pl.col("landing_airport_id")
        .replace_strict(gps, default=None, return_dtype=pl.String)
        .alias("estarrivalairport"),
        "first_lat",
        "first_lon",
        "last_lat",
        "last_lon",
    )


def check(directory, target_date):
    aircraft = sorted(
        json.loads((directory / "manifest.json").read_text())["sample_aircraft"]
    )
    files = sorted((directory / "sample_hours").glob("*.parquet"))
    if len(files) != 72:
        raise ValueError("All 72 sampled context hours are required")
    frozen = AirportCatalogView.from_history(
        FrozenHistory(directory / "ourairports_2024-06-01.parquet"),
        target_date,
        target_date,
    )
    catalogs = {"frozen": frozen, "shared": airport_catalog_for_date(target_date)}
    indices = {name: EndpointAirports(catalog) for name, catalog in catalogs.items()}
    parts = {name: [] for name in catalogs}
    for offset in range(0, len(aircraft), 32):
        batch = aircraft[offset : offset + 32]
        states = (
            pl.scan_parquet(files)
            .filter(
                pl.col("icao24").is_between(pl.lit(batch[0]), pl.lit(batch[-1]))
                & pl.col("icao24").is_in(batch)
            )
            .filter(valid_position())
            .rename({"icao24": "icao"})
            .select(list(STATE_SCHEMA))
            .collect(engine="streaming")
        )
        for name, catalog in catalogs.items():
            parts[name].append(
                reconstruct_arrival_day(
                    target_date, states, catalog, airports=indices[name]
                )
            )
        print(
            f"Checked {min(offset + 32, len(aircraft))}/{len(aircraft)} aircraft",
            flush=True,
        )
    compare(directory, target_date, catalogs, parts)


def compare(directory, target_date, catalogs, parts):
    """Score generated flights after canonical callsign normalization."""
    aircraft = json.loads((directory / "manifest.json").read_text())["sample_aircraft"]
    reference = pl.read_parquet(directory / "reference_flights.parquet").filter(
        pl.col("icao24").is_in(aircraft)
    )
    reference = reference.with_columns(
        pl.col("callsign").str.strip_chars().fill_null("")
    )
    old = pl.read_parquet(
        directory / "reconstructed_sample_flights.parquet"
    ).with_columns(pl.col("callsign").str.strip_chars().fill_null(""))
    keys = ["icao24", "firstseen", "lastseen"]
    labels = ["callsign", "estdepartureairport", "estarrivalairport"]
    epoch = int(
        datetime.combine(target_date, datetime.min.time(), timezone.utc).timestamp()
    )
    report = {
        "date": str(target_date),
        "sample_aircraft": len(aircraft),
        "cohort": "arrival day",
        "callsign_normalization": "trim surrounding whitespace; absent callsign becomes empty string",
        "catalogs": {},
    }
    for name, catalog in catalogs.items():
        flights = pl.concat(parts[name]).sort(["icao", "takeoff_time"])
        prediction = exported(flights, catalog)
        metrics, _ = evaluate(prediction, reference, day=epoch)
        exact = prediction.join(reference, on=keys, suffix="_reference")
        for label in labels:
            metrics[f"{label}_on_exact_boundaries"] = exact.select(
                pl.col(label).eq_missing(pl.col(label + "_reference")).sum()
            ).item()
        metrics["timestamps_and_labels"] = exact.filter(
            pl.all_horizontal(
                pl.col(label).eq_missing(pl.col(label + "_reference"))
                for label in labels
            )
        ).height
        metrics["matches_saved_research_boundaries"] = (
            prediction.select(keys).sort(keys).equals(old.select(keys).sort(keys))
        )
        metrics["matches_saved_research_labels"] = (
            prediction.select(keys + labels)
            .sort(keys)
            .equals(old.select(keys + labels).sort(keys))
        )
        metrics["catalog_provenance"] = asdict(catalog.provenance)
        report["catalogs"][name] = metrics
        flights.write_parquet(directory / f"source_validation_{name}.parquet")
    path = directory / "source_validation.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    if not report["catalogs"]["frozen"]["matches_saved_research_boundaries"]:
        raise AssertionError("Source port changed the frozen research boundaries")
    if not report["catalogs"]["frozen"]["matches_saved_research_labels"]:
        raise AssertionError("Source port changed the frozen research labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/mnt/storage2/planequery-data/data/processed/opensky_reverse_engineering"
        ),
    )
    args = parser.parse_args()
    check(args.data_root / str(args.date), args.date)
