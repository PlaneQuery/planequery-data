from datetime import date, datetime, timedelta

import polars as pl

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.american_evals import continental_usa_to_continental_usa
from flights.flights_comparison import df_flights_comparision
from flights.flights_match_adsb import get_matching_icaos_in_flights


def ground_truth_flights_output_path(target_date: date):
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "ground_truth"
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "flights.parquet"
    )


def get_ground_truth_flights_for_day(
    target_date: date,
    use_cache: bool = True,
) -> pl.DataFrame:
    output_path = ground_truth_flights_output_path(target_date)
    if use_cache and output_path.exists():
        return pl.read_parquet(output_path)

    df_bts_flights = get_bts_flights_for_day(target_date)
    df_sfdps_flights = get_sfdps_flights_day(target_date)
    shared_icaos = set(df_bts_flights["icao"].unique()) & (set(df_sfdps_flights["icao"].unique()))
    df_bts_flights = df_bts_flights.filter(pl.col("icao").is_in(shared_icaos))
    df_sfdps_flights = df_sfdps_flights.filter(pl.col("icao").is_in(shared_icaos))
    df_bts_flights = df_bts_flights.filter(pl.col("landing_time").dt.date() == target_date)
    df_sfdps_flights = df_sfdps_flights.filter(pl.col("landing_time").dt.date() == target_date)
    df_bts_flights = continental_usa_to_continental_usa(df_bts_flights)
    df_sfdps_flights = continental_usa_to_continental_usa(df_sfdps_flights)
    start_dt = datetime(target_date.year, target_date.month, target_date.day)
    end_dt = start_dt + timedelta(days=1)
    original_columns = df_bts_flights.columns
    df = df_flights_comparision(df_bts_flights, df_sfdps_flights, start_dt, end_dt)
    df = df.filter(
    (pl.col("match_status") == "both").all().over("icao")
    )
    df = df.select(original_columns)
    df_adsb = read_adsb(target_date, icaos = df["icao"].unique().to_list(), source = "adsbx")
    df_gold = get_matching_icaos_in_flights(df, df_adsb, start_dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.write_parquet(output_path)
    return df_gold


if __name__ == "__main__":
    target_date = date(2026, 3, 1)
    df_gold = get_ground_truth_flights_for_day(target_date)
    print(df_gold)
