"""Audit transfer to airport identities excluded from all model selection."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from flights_airport_nn.benchmark import evaluate_dates, summarize
from flights_airport_nn.dataset import cache_path, prepare_dates
from flights_airport_nn.features.features import ALL_FEATURE_CONFIG
from flights_airport_nn.train import run_train


def airport_fold(ident, folds=5):
    return int(hashlib.sha256(ident.encode()).hexdigest(), 16) % folds


def run_audit(
    train_dates,
    validation_dates,
    output_dir,
    *,
    epochs=15,
    patience=3,
    folds=5,
    fold=0,
    cache_workers=1,
    prepare=True,
):
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError("Require folds >= 2 and 0 <= fold < folds")
    if set(train_dates) & set(validation_dates):
        raise ValueError("Training and validation dates must be disjoint")
    output_dir = Path(output_dir)
    if (output_dir / "model.pt").exists():
        raise FileExistsError("Choose a new output directory for the audit")
    if prepare:
        prepare_dates((*train_dates, *validation_dates), cache_workers)
    airports = set()
    for day in (*train_dates, *validation_dates):
        with np.load(cache_path(day), allow_pickle=False) as cache:
            airports.update(cache["truth"])
    excluded = sorted(a for a in airports if a and airport_fold(a, folds) == fold)
    model = run_train(
        train_dates,
        validation_dates=validation_dates,
        output_dir=output_dir,
        epochs=epochs,
        patience=patience,
        excluded_airports=excluded,
        prepare=False,
    )
    # Held-out airport endpoints were excluded from fitting, early stopping and calibration.
    frame, _, _ = evaluate_dates(
        model, validation_dates, ALL_FEATURE_CONFIG, temperature=model.temperature
    )
    heldout = frame.filter(pl.col("truth").is_in(excluded))
    if heldout.is_empty():
        raise ValueError("No validation endpoints belong to the held-out airport fold")
    heldout.write_parquet(output_dir / "unseen_airport_predictions.parquet")
    metrics = summarize(heldout)
    metrics.update(
        excluded_airports=excluded,
        airport_folds=folds,
        heldout_fold=fold,
        scope="Previously unseen positive airport labels within the US commercial population; candidate catalogs still contain these airports.",
    )
    (output_dir / "unseen_airport_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument(
        "--validation-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--airport-folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--cache-workers", type=int, default=1)
    args = parser.parse_args(argv)
    run_audit(
        args.train_dates,
        args.validation_dates,
        args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        folds=args.airport_folds,
        fold=args.holdout_fold,
        cache_workers=args.cache_workers,
    )


if __name__ == "__main__":
    main()
