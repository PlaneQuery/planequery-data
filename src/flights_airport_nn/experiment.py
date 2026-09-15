# pick how many days
# maybe have an ExpiermentConfig class, that has name and such. 

from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.flights.flight_type import flights_to_flights_df
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.evaluate import evaluate_model
from flights_airport_nn.inference import run_inference
from flights_airport_nn.train import run_train
from planequery_data.utils import current_commit_hash

class ExperimentConfig:
    pass

EXPERIMENTS_DIR = (
    OUTPUT_DIR / "data" / "experiments" / "flights_airport_nn" / "runs"
)


def experiment_run_dir(experiment_name: str, run_dt: datetime) -> Path:
    timestamp = run_dt.strftime("%Y-%m-%d_%H-%M-%S")
    return EXPERIMENTS_DIR / f"{timestamp}_{experiment_name}_{current_commit_hash()}"


def run_experiment():
    experiment_name = "test0"
    train_dates = [date(2026,2,1)]
    eval_dates = [date(2026,3,1)]

    model = run_train(train_dates)
    flights = run_inference(eval_dates[0], model)
    df_flights = flights_to_flights_df(flights.values())
    eval_stats = evaluate_model(eval_dates[0], df_flights)
    output_dir = experiment_run_dir(experiment_name, datetime.now())
    path = output_dir / "evaluation" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([eval_stats.to_record()])
    df.write_csv(path)
    print(path)
    print(df)


if __name__ == "__main__":
    run_experiment()
