"""Audit European disagreements against raw physical evidence; never relabel.

This is an error review, not a corrected accuracy estimate. The same ADS-B
observations used by the predictors cannot supply independent ground truth.
"""

import argparse
import importlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from flights_airport_nn.audit_source_matrix import _CATALOG, evidence
from flights_airport_nn.broader_evaluation import load_catalog


def audit(output):
    protocol = json.loads((output / "protocol.json").read_text())
    reader = importlib.import_module("data_engineering.adsb.read_adsb")
    for source, root in protocol.get("parquet_roots", {}).items():
        setattr(
            reader,
            "ADSBLOL_PARQUET_ROOT" if source == "adsblol" else "ADSBX_PARQUET_ROOT",
            Path(root),
        )
    _CATALOG.update(load_catalog())
    endpoints = pl.read_parquet(output / "endpoint_results.parquet").filter(
        (pl.col("loss_seconds") == 0)
        & (pl.col("continent") == "EU")
        & (pl.col("match_status") == "matched")
        & (~pl.col("reference_time_conflict"))
        & ((~pl.col("nn_correct")) | (~pl.col("nearest_correct")))
    )
    rows = []
    for (day, source), group in endpoints.partition_by(
        ["date", "source"], as_dict=True
    ).items():
        aircraft = group["icao"].unique().sort().to_list()
        for start in range(0, len(aircraft), 64):
            part = group.filter(pl.col("icao").is_in(aircraft[start : start + 64]))
            messages = (
                reader.read_adsb(
                    date.fromisoformat(day),
                    source=source,
                    icaos=part["icao"].unique().to_list(),
                    columns=[
                        "icao",
                        "time",
                        "lat",
                        "lon",
                        "baro_altitude_ft",
                        "ground_speed_kt",
                        "on_ground",
                        "registration",
                    ],
                    lazy=True,
                )
                .filter(
                    pl.col("lat").is_finite()
                    & pl.col("lon").is_finite()
                    & pl.col("time").is_not_null()
                )
                .unique(subset=["icao", "time"])
                .sort(["icao", "time"])
                .collect(engine="streaming")
            )
            tracks = messages.partition_by("icao", as_dict=True)
            for r in part.iter_rows(named=True):
                track = tracks.get((r["icao"],))
                if track is None:
                    continue
                arrays = {
                    "time": track["time"].dt.epoch("ms").to_numpy(),
                    "lat": track["lat"].to_numpy(),
                    "lon": track["lon"].to_numpy(),
                    "alt": track["baro_altitude_ft"].cast(pl.Float64).to_numpy(),
                    "speed": track["ground_speed_kt"].cast(pl.Float64).to_numpy(),
                    "ground": track["on_ground"].fill_null(False).to_numpy(),
                }
                native_time = r[
                    "segment_start" if r["endpoint"] == "takeoff" else "segment_end"
                ]
                # Use naive UTC arithmetic: datetime.timestamp() would depend
                # on the host timezone for the repository's naive datetimes.
                native_ms = np.datetime64(native_time, "ms").astype(np.int64)
                lo, hi = np.searchsorted(
                    arrays["time"], [native_ms - 900_000, native_ms + 900_000]
                )
                out = {
                    k: r[k]
                    for k in [
                        "reference_id",
                        "reference_source",
                        "date",
                        "source",
                        "icao",
                        "registration",
                        "endpoint",
                        "truth",
                        "nn",
                        "nearest",
                        "nn_correct",
                        "nearest_correct",
                        "reference_time",
                        "segment_start",
                        "segment_end",
                    ]
                }
                for algorithm in ("truth", "nn", "nearest"):
                    e = evidence(arrays, r[algorithm], lo, hi)
                    for field in ("strong", "points", "span_s", "closest_km"):
                        out[f"{algorithm}_{field}"] = e[field]
                alternatives = any(
                    out[f"{a}_strong"] and r[a] != r["truth"] for a in ("nn", "nearest")
                )
                if out["truth_strong"]:
                    review = (
                        "reference_and_alternative_supported_nearby"
                        if alternatives
                        else "reference_supported_prediction_error"
                    )
                else:
                    review = (
                        "alternative_supported_reference_needs_review"
                        if alternatives
                        else "insufficient_ground_evidence"
                    )
                out["review_status"] = review
                registrations = (
                    track["registration"].drop_nulls().unique().sort().to_list()
                )
                out["trace_registrations"] = registrations
                normalized = r["registration"].replace("-", "").upper()
                out["registration_disagreement"] = bool(
                    normalized
                    and registrations
                    and normalized
                    not in [s.replace("-", "").upper() for s in registrations]
                )
                rows.append(out)
            print(
                f"Audited {day} {source}: {min(start + 64, len(aircraft))}/{len(aircraft)} aircraft",
                flush=True,
            )
    result = pl.DataFrame(rows, infer_schema_length=None)
    result.write_parquet(output / "disagreement_audit.parquet")
    summary = (
        result.group_by(["source", "review_status"])
        .len()
        .sort(["source", "review_status"])
        .to_dicts()
        if rows
        else []
    )
    (output / "disagreement_audit.json").write_text(
        json.dumps(
            {
                "policy": "All matched European disagreements, no overlap-conflicting references. Ground support within +/-15min of native endpoint: >=3 points spanning >=10s, within 2km airport center, ground flag or <=300ft elevation difference and <=100kt. Registration metadata can be stale. Review flags do not correct labels or remove errors. Nearby airports may both pass; absent ground reception proves neither airport wrong.",
                "counts": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
