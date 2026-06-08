import argparse
import os
from datetime import date
from pathlib import Path

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from flights.flights_match_adsb import flights_match_adsb, get_matching_icaos_in_flights

os.environ.setdefault("OUTPUT_DIR", str(Path.cwd()))

import polars as pl

from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from data_engineering.flights.flight_type import get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights_ai_model.validation import flights_df_comparison

DEFAULT_TARGET_DATE = date(2026, 3, 1)
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])


def read_opensky_flights(target_date: date) -> pl.DataFrame:
    return pl.read_parquet(f"/Volumes/T2-SSD/planequery/data/flights/opensky/v1/{target_date:%Y-%m-%d}.parquet")


def default_stats_path(target_date: date) -> Path:
    return OUTPUT_DIR / "data" / "experiments" / "comparison" / f"{target_date:%Y-%m-%d}_stats.csv"


def main(test_src: str, gold_src: str, test_dates: date | list[date]):
    if isinstance(test_dates, date):
        test_dates = [test_dates]
    
    if test_src == "adsbx":
        get_source_flights = get_adsbx_flights_for_day

    if gold_src == "sfdps":
        get_gold_flights = get_sfdps_flights_day # we may need some logic for perfect icao matching. 
        # flights_match_adsb
    # how to deal with takeoff_time and landing_time
    rows = []
    for test_date in test_dates: # actually we may have to do things with matching the takeoff_time and landing_time
        df_source = get_source_flights(test_date)
        df_gold = get_gold_flights(test_date)
        df_adsb = read_adsb(test_date)
        df_gold = get_matching_icaos_in_flights(df_gold, df_adsb)
        df_source = df_source.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
        _, stats = flights_df_comparison(df_source, df_gold)
        rows.append({"test_date": test_date.isoformat(), "test_src": test_src, "gold_src": gold_src, **stats})
        print(f"Stats for {test_src}:{gold_src}: {stats}")

    stats_output = default_stats_path(test_date)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(stats_output)
    print(f"Stats CSV saved to {stats_output}")


if __name__ == "__main__":
    main("adsbx", "sfdps", date(2026,3,1))
