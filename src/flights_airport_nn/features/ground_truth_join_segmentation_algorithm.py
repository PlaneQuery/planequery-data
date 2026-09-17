"""Join labels to segmented flights without using labels to match airports."""
from datetime import date, datetime, timedelta
import polars as pl
from data_engineering.flights.flight_type import (
    flights_algorithm_output_path, with_flight_schema_columns,
)
from flights.flights_comparison import df_flights_comparison
from flights.ground_truth.bts_and_sfdps_to_gold import ground_truth_flights_output_path


def flights_features_df(test_dt: date):
    # Require local caches: training must not silently download or rebuild labels.
    truth = pl.read_parquet(ground_truth_flights_output_path(test_dt, 'adsblol'))
    segments = pl.read_parquet(flights_algorithm_output_path(
        test_dt, adsb_src='adsblol', no_airports_model=True,
    )).filter(pl.col('icao').is_in(truth['icao'].unique().implode()))
    start = datetime.combine(test_dt, datetime.min.time())
    comparison = df_flights_comparison(
        segments, truth, start, start + timedelta(days=1), compare_airports=False,
    )
    # Keep individually matched segments, including aircraft with unmatched flights.
    matched = comparison.filter(pl.col('match_status') == 'both').with_columns(
        pl.col('takeoff_airport_ident_df1').alias('takeoff_airport_ident'),
        pl.col('landing_airport_ident_df1').alias('landing_airport_ident'),
    )
    return with_flight_schema_columns(matched)
