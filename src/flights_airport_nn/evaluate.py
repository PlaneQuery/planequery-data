# this would be passing in the filghts is useful so I can reuse ground truth flights here. Although loading them is not that expensive at all and saves ram. 
from datetime import timedelta, datetime
from attr import dataclass
import polars as pl
from flights.flights_comparison import df_flights_comparison, df_flights_comparison_stats
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day

@dataclass
class eval_stats:
    x: int

# I should still have the neareast airport approach in the .csvs just for compariaision. I can save that data to OUTPUT_DIR/cache/

def evaluate_flights_airport_nn(eval_dt, df_flights):
    df_gt = get_ground_truth_flights_for_day(eval_dt)

    df = df_flights_comparison(df_flights, df_gt, datetime(eval_dt.year, eval_dt.month, eval_dt.day), datetime(eval_dt.year, eval_dt.month, eval_dt.day) + timedelta(days=1), compare_airports=False)
    df = df.filter(pl.col("match_status") == "both")
    stats = df_flights_comparison_stats(df)
    stats['takeoff_airport_ident_total_count']
    # output with the flights and then the true takeoff_airport. Maybe having the groundtruth takeoff/landing time would be useful as well? 
    print(stats)

    # Maybe we can cache the accuracy of the ground_truth. 

    # what should evaluation values be: number_correct_takeoffs_ total_number_takeoff takeoff_airport_accuracy. 

if __name__ == "__main__":
    from datetime import date
    from flights_airport_nn.inference import run_inference, load_model

    model = load_model(
        model_path="/Volumes/T2-SSD/planequery/data/models/flights_airport_nn/runs/2026-09-13_17-39_9a999b3_0days_0epochs/model.pt",
    )
    eval_dt = date(2026, 3, 1)
    flights = run_inference(eval_dt, model)
    from data_engineering.flights.flight_type import flights_to_flights_df
    df_flights = flights_to_flights_df(flights.values())
    evaluate_flights_airport_nn(eval_dt, df_flights)