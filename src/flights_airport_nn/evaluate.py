"""Evaluate a saved checkpoint on dates excluded from training and validation."""

import argparse
from datetime import date
from pathlib import Path

from flights_airport_nn.dataset import prepare_day
from flights_airport_nn.experiment import final_test
from flights_airport_nn.inference import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    model, feature_config = load_model(args.model)
    overlap = set(args.dates) & (
        set(model.training_dates) | set(model.validation_dates)
    )
    if overlap:
        parser.error(f"Test dates overlap training/validation: {sorted(overlap)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "test_metrics.json").exists():
        parser.error("Test metrics already exist; choose a new output directory")
    for day in args.dates:
        prepare_day(day)
    final_test(model, args.dates, model.training_dates, args.output_dir, feature_config)


if __name__ == "__main__":
    main()
