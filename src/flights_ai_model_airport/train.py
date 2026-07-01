from datetime import date, datetime, timedelta
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
from flights.flights_match_adsb import get_matching_icaos_in_flights
from flights_ai_model_airport.features import build_training_data
from flights_ai_model_airport.model import model_training
from flights_ai_model_airport.flights_ai_model_utils import get_model_save_path

def train(training_dates, inference_date = date(2026,3,15)):
    # test_dates = [date(2026,3,1), date(2026,3,15)]
    df = pl.DataFrame()
    for dt in training_dates:
        df_sfpds_flights = get_sfdps_flights_day(dt)
        df_adsb = read_adsb(dt, icaos=df_sfpds_flights.get_column("icao").to_list())
        df_sfdps_flights = get_matching_icaos_in_flights(df_sfpds_flights, df_adsb)
        df_flights = get_flights(dt)
        df = df_flights_comparision(df_flights, df_sfdps_flights, datetime(dt.year, dt.month, dt.day), datetime(dt.year, dt.month, dt.day) + timedelta(days=1), compare_airports=False)
        df = df.filter(pl.col("match_status") == "both")
        df = df.with_columns(pl.col("takeoff_airport_ident_df1").alias("takeoff_airport_ident"))
        df = df.select(FLIGHT_POLARS_SCHEMA.keys())
        df = pl.concat([df, df_flights], how="vertical")

    df = df.with_columns(
        (
            pl.col("icao").cast(pl.Utf8)
            + "_"
            + pl.col("takeoff_time").dt.strftime("%Y-%m-%d_%H-%M")
        ).alias("flight_id")
    )
    print(len(df), "flights for training")
    df = df.filter(pl.col("first_message_time").is_not_null() & pl.col("last_message_time").is_not_null())
    X, y, groups, airport_idents = build_training_data(df)
    print(X)
    model = model_training(X, y, groups, airport_idents)
    import joblib
    path = get_model_save_path(num_days=len(training_dates))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    print(f"  -> saved to {path}")

def evalaution(eval_date = date(2026,6,1)):
    df_test = run_inference(eval_date)
    df_sfpds_flights = get_sfdps_flights_day(eval_date)
    df_adsb = read_adsb(eval_date, icaos=df_sfpds_flights.get_column("icao").to_list())
    df_gold = get_matching_icaos_in_flights(df_sfpds_flights, df_adsb)
    df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
    df = df_flights_comparision(df_test, df_gold, datetime(eval_date.year, eval_date.month, eval_date.day), datetime(eval_date.year, eval_date.month, eval_date.day) + timedelta(days=1), compare_airports=False)
    stats = df_flights_comparison_stats(df)
    print(stats)


if __name__ == "__main__":
    # training_dates = [
    #     date(2026, 6, 1) + timedelta(days=i)
    #     for i in range((date(2026, 6, 10) - date(2026, 6, 1)).days + 1)
    # ]
    training_dates = [date(2026,3,1)]
    train(training_dates)
