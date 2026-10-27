"""Report airport accuracy by exposure to positive training labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from airports.runtime import airport_catalog_for_date
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.models import (
    EVALUATION_METADATA_COLUMNS,
    EVALUATION_STATS_SCHEMA,
    EvaluationResults,
)


AIRPORT_EXPOSURE_SCHEMA = {
    **{name: EVALUATION_STATS_SCHEMA[name] for name in EVALUATION_METADATA_COLUMNS},
    "training_manifest": pl.String,
    "training_manifest_sha256": pl.String,
    "endpoint": pl.String,
    "airport_group": pl.String,
    "gold_endpoint_count": pl.Int64,
    "gold_airport_count": pl.Int64,
    "matched_endpoint_count": pl.Int64,
    "matched_airport_count": pl.Int64,
    "correct_count": pl.Int64,
    "incorrect_count": pl.Int64,
    "null_prediction_count": pl.Int64,
    "airport_accuracy": pl.Float64,
    "airport_macro_accuracy": pl.Float64,
    "flight_match_coverage": pl.Float64,
    "correct_endpoint_recall": pl.Float64,
}


@dataclass(frozen=True)
class TrainingAirports:
    manifest_path: Path
    manifest_sha256: str
    # Counts combine takeoff and landing endpoints that contributed to the loss.
    airport_counts: pl.DataFrame


def _training_airport_counts(metadata: pl.LazyFrame) -> pl.DataFrame:
    eligible = metadata.filter(pl.col("eligible") & (pl.col("target") >= 0))
    if "truth_id" in metadata.collect_schema():
        counts = (
            eligible.filter(pl.col("truth_id").is_not_null())
            .group_by("truth_id")
            .agg(pl.len().cast(pl.Int64).alias("training_endpoint_count"))
            .rename({"truth_id": "airport_id"})
            .collect()
        )
    else:
        # Code labels must be resolved on the training date, before combining
        # counts: identifiers can change or be reassigned between dates.
        labels = (
            eligible.group_by("date", "truth")
            .agg(pl.len().alias("training_endpoint_count"))
            .collect()
        )
        rows = []
        for (day,), group in labels.partition_by("date", as_dict=True).items():
            catalog = airport_catalog_for_date(date.fromisoformat(day))
            for label in group.iter_rows(named=True):
                airport = catalog.get_airport_by_ident(label["truth"])
                if airport is None:
                    raise ValueError(
                        f"Cannot resolve training airport {label['truth']!r} on {day}"
                    )
                rows.append((airport.id, label["training_endpoint_count"]))
        counts = pl.DataFrame(
            rows,
            schema={"airport_id": pl.Int64, "training_endpoint_count": pl.Int64},
            orient="row",
        ).group_by("airport_id").agg(pl.col("training_endpoint_count").sum())
    return counts.with_columns(pl.col("airport_id").cast(pl.Int64)).sort("airport_id")


def load_training_airports(manifest_path: Path) -> TrainingAirports:
    """Read only the training metadata parts named by a training.json manifest.

    Fingerprints are [path, size, mtime_ns] triples. Metadata is stored under
    OUTPUT_DIR/data/processed/flights_airport_nn/<dataset>/<date>/<filename>.
    Only that dataset/date/filename suffix identifies the local file; the
    recorded machine prefix is irrelevant. A relative manifest argument resolves
    under OUTPUT_DIR. Validation parts and NPZ feature arrays are not read.
    """
    manifest_path = (OUTPUT_DIR / Path(manifest_path).expanduser()).resolve()
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    parts = []
    for fingerprint in manifest["inputs"]["train"]:
        recorded = Path(fingerprint[0])
        if recorded.suffix != ".parquet":
            continue
        path = (
            OUTPUT_DIR / "data/processed/flights_airport_nn"
            / recorded.parent.parent.name / recorded.parent.name / recorded.name
        )
        if not path.is_file():
            raise FileNotFoundError(f"Training airport metadata not found: {path}")
        parts.append(path)
    if not parts:
        raise ValueError(f"No training metadata Parquet files in {manifest_path}")

    return TrainingAirports(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(contents).hexdigest(),
        airport_counts=_training_airport_counts(pl.scan_parquet(parts)),
    )


def airport_exposure_report(
    results: EvaluationResults,
    training: TrainingAirports,
) -> pl.DataFrame:
    """Summarize each daily comparison, without rematching or changing its cache.

    Membership uses the gold endpoint ID and the union of training takeoff and
    landing labels. Unknown gold IDs are counted separately and never scored.
    Null predictions are errors for known gold IDs. Empty denominators produce
    null rates. Coverage includes retained gold-only flights, but not test-only
    rows or flights already discarded by the evaluator's boundary grace period.
    """
    rows = []
    training_ids = training.airport_counts.get_column("airport_id").to_list()
    for entry in results.entries:
        stats_row = entry.stats.to_row()
        metadata = {name: stats_row[name] for name in EVALUATION_METADATA_COLUMNS}
        gold = entry.comparison.filter(
            pl.col("match_status").is_in(["both", "df1_only"])
        )
        for endpoint in ("takeoff", "landing"):
            endpoints = gold.select(
                pl.col(f"{endpoint}_airport_id_df1").alias("gold_id"),
                pl.col(f"{endpoint}_airport_id").alias("predicted_id"),
                (pl.col("match_status") == "both").alias("matched"),
            ).with_columns(
                pl.when(pl.col("gold_id").is_null())
                .then(pl.lit("unknown"))
                .when(pl.col("gold_id").is_in(training_ids))
                .then(pl.lit("training"))
                .otherwise(pl.lit("nontraining"))
                .alias("airport_group"),
                (
                    pl.col("matched")
                    & (pl.col("predicted_id") == pl.col("gold_id"))
                ).fill_null(False).alias("correct"),
            )
            for group in ("training", "nontraining", "unknown"):
                subset = endpoints.filter(pl.col("airport_group") == group)
                matched = subset.filter(pl.col("matched"))
                known = group != "unknown"
                correct = int(matched["correct"].sum()) if known else None
                macro_accuracy = None
                if known and matched.height:
                    macro_accuracy = (
                        matched.group_by("gold_id")
                        .agg(pl.col("correct").mean())
                        .get_column("correct")
                        .mean()
                    )
                rows.append({
                    **metadata,
                    "training_manifest": str(training.manifest_path),
                    "training_manifest_sha256": training.manifest_sha256,
                    "endpoint": endpoint,
                    "airport_group": group,
                    "gold_endpoint_count": subset.height,
                    "gold_airport_count": subset["gold_id"].drop_nulls().n_unique(),
                    "matched_endpoint_count": matched.height,
                    "matched_airport_count": matched["gold_id"].drop_nulls().n_unique(),
                    "correct_count": correct,
                    "incorrect_count": matched.height - correct if known else None,
                    "null_prediction_count": matched["predicted_id"].null_count(),
                    "airport_accuracy": (
                        correct / matched.height if known and matched.height else None
                    ),
                    "airport_macro_accuracy": macro_accuracy,
                    "flight_match_coverage": (
                        matched.height / subset.height if subset.height else None
                    ),
                    "correct_endpoint_recall": (
                        correct / subset.height if known and subset.height else None
                    ),
                })
    return pl.DataFrame(rows, schema=AIRPORT_EXPOSURE_SCHEMA)
