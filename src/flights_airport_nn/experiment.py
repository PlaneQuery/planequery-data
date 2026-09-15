# pick how many days
# maybe have an ExpiermentConfig class, that has name and such. 

from dataclasses import dataclass
from datetime import date, datetime
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
    features: FeatureConfig

EXPERIMENTS_DIR = (
    OUTPUT_DIR / "data" / "experiments" / "flights_airport_nn" / "runs"
)


def experiment_run_dir(experiment_name: str, run_dt: datetime) -> Path:
    timestamp = run_dt.strftime("%Y-%m-%d_%H-%M-%S")
    return EXPERIMENTS_DIR / f"{timestamp}_{experiment_name}_{current_commit_hash()}"


def run_experiment(config: ExperimentConfig):
    model = run_train(config.train_dates, config.features)
    flights = run_inference(config.eval_dates[0], model, config.features)
    df_flights = flights_to_flights_df(flights.values())
    eval_stats = evaluate_model(config.eval_dates[0], df_flights)
    output_dir = experiment_run_dir(config.name, datetime.now())
    path = output_dir / "evaluation" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([eval_stats.to_record()])
    df.write_csv(path)
    print(path)
    print(df)


if __name__ == "__main__":
    run_experiment(
        ExperimentConfig(
            name="all_features_0",
            train_dates=(date(2026, 2, 1),),
            eval_dates=(date(2026, 3, 1),),
            features=ALL_FEATURE_CONFIG,
        )
    )
