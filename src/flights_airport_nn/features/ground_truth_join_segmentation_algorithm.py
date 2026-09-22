import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import date

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.flights.flight_type import with_flight_schema_columns
from flights.evaluation.evaluation import main as run_evaluation


def flights_features_df(test_dt: date):
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            outputs, _ = run_evaluation(
                "algorithm-no-airports",
                "ground_truth",
                test_dates=test_dt,
                adsb_src=ADSBSource.ADSBLOL,
                matching_adsb_src=ADSBSource.ADSBLOL,
                write_outputs=False
            )

    comparison_df = outputs[0]
    comparison_df = comparison_df.filter(
        (pl.col("match_status") == "both").all().over("icao")
    )
    comparison_df = comparison_df.with_columns(pl.col("takeoff_airport_ident_df1").alias("takeoff_airport_ident"))
    comparison_df = comparison_df.with_columns(pl.col("landing_airport_ident_df1").alias("landing_airport_ident"))
    comparison_df = with_flight_schema_columns(
        comparison_df
    )
    return comparison_df
