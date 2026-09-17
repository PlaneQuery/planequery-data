import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from data_engineering.flights.flight_type import flights_to_flights_df
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.evaluate import evaluate_model
from flights_airport_nn.features.features import ALL_FEATURE_CONFIG, FeatureConfig
from flights_airport_nn.inference import run_inference
from flights_airport_nn.train import run_train
from planequery_data.utils import current_commit_hash


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_dates: tuple[date, ...]
    eval_dates: tuple[date, ...]
    epochs: int
    features: FeatureConfig


EXPERIMENTS_DIR = (
    OUTPUT_DIR / "data" / "experiments" / "flights_airport_nn" / "runs"
)


def experiment_run_dir(experiment_name: str, run_dt: datetime) -> Path:
    timestamp = run_dt.strftime("%Y-%m-%d_%H-%M-%S")
    return EXPERIMENTS_DIR / f"{timestamp}_{experiment_name}_{current_commit_hash()}"


def parse_experiment_config(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test_experiment")
    parser.add_argument(
        "--train-dates",
        type=int,
        default=1,
        help="number of training days starting from 2026-02-01",
    )
    parser.add_argument(
        "--eval-dates",
        type=date.fromisoformat,
        nargs="+",
        default=[date(2026, 3, 1)],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="maximum number of training epochs",
    )
    args = parser.parse_args(argv)
    if args.train_dates < 1:
        parser.error("--train-dates must be a positive integer")
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    train_dates = tuple(
        date(2026, 2, 1) + timedelta(days=offset)
        for offset in range(args.train_dates)
    )
    return ExperimentConfig(
        name=args.name,
        train_dates=train_dates,
        eval_dates=tuple(args.eval_dates),
        epochs=args.epochs,
        features=ALL_FEATURE_CONFIG,
    )


def run_experiment(config: ExperimentConfig):
    model = run_train(
        config.train_dates,
        config.features,
        epochs=config.epochs,
    )
    eval_records = []
    for eval_date in config.eval_dates:
        flights = run_inference(eval_date, model, config.features)
        df_flights = flights_to_flights_df(flights.values())
        eval_stats = evaluate_model(eval_date, df_flights)
        eval_records.append(eval_stats.to_record())

    output_dir = experiment_run_dir(config.name, datetime.now(UTC))
    path = output_dir / "evaluation" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(eval_records)
    df.write_csv(path)
    print(path)
    print(df)


if __name__ == "__main__":
    run_experiment(parse_experiment_config())
