# this would be passing in the filghts is useful so I can reuse ground truth flights here. Although loading them is not that expensive at all and saves ram. 
from datetime import timedelta, datetime
import polars as pl
from data_engineering.flights.flight_type import get_flights
from flights.flights_comparison import df_flights_comparison, df_flights_comparison_stats
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day

class EvalStats:
    def __init__(self, eval_dt, takeoff_airport_total_count: int, takeoff_airport_correct_count:int, nearest_takeoff_airport_correct_count:int):
        self.eval_dt = eval_dt
        self.takeoff_airport_total_count = takeoff_airport_total_count
        self.takeoff_airport_correct_count = takeoff_airport_correct_count
        self.nearest_takeoff_airport_correct_count = nearest_takeoff_airport_correct_count
        # TODO: have modelconfig properites. 

    @property
    def takeoff_airport_incorrect_count(self):
        return self.takeoff_airport_total_count - self.takeoff_airport_correct_count
    # derive percentages here or maybe do this in an __init__ ?
    

    def to_record(self):
        return {
            "eval_dt": self.eval_dt.isoformat(),
            "takeoff_airport_total_count": self.takeoff_airport_total_count,
            "takeoff_airport_correct_count": self.takeoff_airport_correct_count,
            "takeoff_airport_incorrect_count": self.takeoff_airport_incorrect_count,
            "nearest_takeoff_airport_correct_count": self.nearest_takeoff_airport_correct_count,
        }


# I should still have the neareast airport approach in the .csvs just for compariaision. I can save that data to OUTPUT_DIR/cache/
def get_nearest_airport_method_stats(df_gt, eval_dt):
    df_nearest_flights = get_flights(eval_dt, no_airports_model=True)
    df = df_flights_comparison(df_nearest_flights, df_gt, datetime(eval_dt.year, eval_dt.month, eval_dt.day), datetime(eval_dt.year, eval_dt.month, eval_dt.day) + timedelta(days=1), compare_airports=False)
    df = df.filter(pl.col("match_status") == "both")
    stats = df_flights_comparison_stats(df)
    return stats

def evaluate_model(eval_dt, df_flights):
    df_gt = get_ground_truth_flights_for_day(eval_dt)

    df = df_flights_comparison(df_flights, df_gt, datetime(eval_dt.year, eval_dt.month, eval_dt.day), datetime(eval_dt.year, eval_dt.month, eval_dt.day) + timedelta(days=1), compare_airports=False)
    df = df.filter(pl.col("match_status") == "both")
    stats = df_flights_comparison_stats(df)
    stats['takeoff_airport_ident_total_count']
    nearest_airport_stats = get_nearest_airport_method_stats(df_gt, eval_dt)
    eval_stats = EvalStats(
        eval_dt,
        stats["takeoff_airport_ident_total_count"],
        stats["takeoff_airport_ident_match_count"],
        nearest_airport_stats["takeoff_airport_ident_match_count"],
    )
    # output with the flights and then the true takeoff_airport. Maybe having the groundtruth takeoff/landing time would be useful as well? 
    return eval_stats


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
    eval_stats = evaluate_model(eval_dt, df_flights)
    df = pl.DataFrame([eval_stats.to_record()])
    print(df)