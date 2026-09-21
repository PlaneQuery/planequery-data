from datetime import date
import polars as pl

from data_engineering.flights.flight_type import flight_id_expr, get_flights
from data_profiling.metrics.flights import FlightsMetrics
from flights.evaluation.evaluation import main
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day

results, stats = main(
    test_src="algorithm-no-airports",
    gold_src="ground_truth",
    test_dates=date(2026, 3, 1),
    adsb_src="adsblol",  # Internal name for adsb.lol
    write_outputs=False
)
df: pl.DataFrame = results[0]
df = df.with_columns(flight_id_expr())
print(len(df))
evaluatable = df.filter(pl.col("match_status") == "both").get_column("flight_id").unique()
print(len(evaluatable))
df_algo = get_flights(date(2026, 3, 1), adsb_src="adsblol", algorithm="algorithm", no_airports_model=True)
print(FlightsMetrics(df_algo, date(2026, 3, 1)).to_vis())
df_groundtruth = get_ground_truth_flights_for_day(date(2026, 3, 1), adsb_src="adsblol", use_cache=True)
print(FlightsMetrics(df_groundtruth, date(2026, 3, 1)).to_vis())
