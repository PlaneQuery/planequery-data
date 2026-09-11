from datetime import date

import polars as pl

from data_engineering.flights.flight_type import with_flight_schema_columns
from flights.evaluation.evaluation import main as run_evaluation

def flights_features_df(test_dt: date):
    outputs, _ = run_evaluation(
        "algorithm-no-airports",
        "ground_truth",
        test_dates=test_dt,
        adsb_src="adsblol",
        matching_adsb_src="adsblol",
    )

    comparison_df = outputs[0]
    comparison_df = comparison_df.with_columns(pl.col("takeoff_airport_ident_df1").alias("takeoff_airport_ident"))
    comparison_df = comparison_df.with_columns(pl.col("landing_airport_ident_df1").alias("landing_airport_ident"))
    df0_flights = with_flight_schema_columns(
        comparison_df.filter(
            pl.col("match_status").is_in(["both", "df0_only"])
        )
    )
    return df0_flights
