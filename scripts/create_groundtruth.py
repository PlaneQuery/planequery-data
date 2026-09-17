from datetime import date, timedelta

from flights.ground_truth.bts_and_sfdps_to_gold import (
    get_ground_truth_flights_for_day,
)

start_date = date(2026, 2, 1)
end_date = date(2026, 4, 1)

current_date = start_date

while current_date <= end_date:
    print(f"Processing {current_date}")

    df_gold = get_ground_truth_flights_for_day(
        current_date,
        adsb_src="adsblol",
        use_cache=False,
    )

    current_date += timedelta(days=1)