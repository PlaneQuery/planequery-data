# evaluate againsit baseline. 

# this would be passing in the filghts is useful so I can reuse ground truth flights here. Although loading them is not that expensive at all and saves ram. 

from datetime import timedelta, datetime

from flights.flights_comparison import df_flights_comparision
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day


def evaluate_flights_airport_nn(eval_dt, df_flights):
    df_gt = get_ground_truth_flights_for_day(eval_dt)

    # TODO: Check if flights are still being "DISCARD"
    df_flights_comparision(df_flights, df_gt, datetime(eval_dt.year, eval_dt.month, eval_dt.day), datetime(eval_dt.year, eval_dt.month, eval_dt.day) + timedelta(days=1))