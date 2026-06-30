import polars as pl
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from datetime import date, datetime, timedelta

from torch import le
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, flights_df_to_flights, get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from evaluation.flights.evaluation import main
from flights.flights_comparison import df_flights_comparision
from flights.flights_match_adsb import get_matching_icaos_in_flights
from flights_ai_model_airport.features import build_training_data

def model_training(X, y, groups, airport_idents):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        eval_metric="binary_logloss",
    )

    val_scores = model.predict_proba(X[val_idx])[:, 1]

    val_results = pl.DataFrame({
        "flight_id": groups[val_idx],
        "airport_ident": airport_idents[val_idx],
        "label": y[val_idx],
        "score": val_scores,
    })

    top1 = (
        val_results
        .sort(["flight_id", "score"], descending=[False, True])
        .group_by("flight_id")
        .head(1)
    )

    print("Top-1 airport accuracy:", top1["label"].mean())

    return model
def train_model():
    test_dates = [date(2026,3,1)]
    df = pl.DataFrame()
    for test_date in test_dates:
        df_sfpds_flights = get_sfdps_flights_day(test_date)
        df_adsb = read_adsb(test_date, icaos=df_sfpds_flights.get_column("icao").to_list())
        df_sfdps_flights = get_matching_icaos_in_flights(df_sfpds_flights, df_adsb)
        df_flights = get_flights(test_date)
        df = df_flights_comparision(df_flights, df_sfdps_flights, datetime(test_date.year, test_date.month, test_date.day), datetime(test_date.year, test_date.month, test_date.day) + timedelta(days=1), compare_airports=False)
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
    from data_engineering.utils import OUTPUT_DIR
    import joblib
    path = OUTPUT_DIR / "tmp" / "flights_ai_model_airport_2.pkl"
    joblib.dump(model, str(path))

if __name__ == "__main__":
    train_model()