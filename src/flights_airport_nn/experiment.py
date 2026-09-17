"""CLI for cache preparation, training and a once-only final test."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from flights_airport_nn.benchmark import evaluate_dates, summarize
from flights_airport_nn.dataset import cache_path, prepare_dates
from flights_airport_nn.features.features import ALL_FEATURE_CONFIG
from flights_airport_nn.train import run_train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument(
        "--validation-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument("--test-dates", nargs="+", type=date.fromisoformat, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=1)
    args = parser.parse_args(argv)
    splits = [set(args.train_dates), set(args.validation_dates), set(args.test_dates)]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        parser.error("Train, validation and test dates must be disjoint")
    if any(
        len(dates) != len(set(dates))
        for dates in [args.train_dates, args.validation_dates, args.test_dates]
    ):
        parser.error("Dates must be unique within each split")
    if not args.prepare_only and (args.output_dir / "model.pt").exists():
        parser.error(
            "Output directory already contains a model; choose a new directory"
        )
    prepare_dates(
        (*args.train_dates, *args.validation_dates, *args.test_dates),
        args.cache_workers,
    )
    if args.prepare_only:
        return
    model = run_train(
        args.train_dates,
        epochs=args.epochs,
        validation_dates=args.validation_dates,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
        prepare=False,
    )
    if args.test_dates:
        final_test(model, args.test_dates, args.train_dates, args.output_dir)


def final_test(
    model, test_dates, train_dates, output_dir, feature_config=ALL_FEATURE_CONFIG
):
    overlap = set(test_dates) & (
        set(train_dates) | set(getattr(model, "validation_dates", ()))
    )
    if overlap:
        raise ValueError(f"Test dates overlap training/validation: {sorted(overlap)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "test_metrics.json").exists():
        raise FileExistsError(
            "Final test metrics already exist; choose a new output directory"
        )
    aircraft, airports = set(), set()
    for day in train_dates:
        with np.load(cache_path(day), allow_pickle=False) as z:
            aircraft.update(s.split("_")[0] for s in z["flight_id"])
            airports.update(z["truth"])
    frame, _, _ = evaluate_dates(
        model,
        test_dates,
        feature_config,
        temperature=getattr(model, "temperature", 1.0),
    )
    frame = frame.with_columns(
        pl.col("icao").is_in(sorted(aircraft)).alias("seen_aircraft"),
        pl.col("truth").is_in(sorted(airports)).alias("seen_airport"),
    )
    frame.write_parquet(output_dir / "test_predictions.parquet")
    metrics = summarize(frame)
    metrics["test_dates"] = [d.isoformat() for d in test_dates]
    metrics["limitations"] = [
        "Labels are BTS/SFDPS agreement on US commercial flights; general aviation and worldwide performance are unmeasured.",
        "Confidence is conditional on candidate retrieval; missing truth is counted as incorrect.",
        "Date splits can share aircraft, airports and routes. Unseen groups are reported separately.",
        "Binomial intervals do not account for correlations between endpoints, aircraft or days.",
    ]
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    main()
