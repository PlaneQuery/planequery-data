from datetime import date, timedelta
from pathlib import Path

import polars as pl
import joblib
from lightgbm import LGBMClassifier

from adsb.adsb_messages_types import adsb_messages_from_parquet_df
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import add_flight_id_col, flights_df_to_flights, get_flights, pia_or_american_ladd_icao_filter
from flights_ai_model_airport.features import build_inference_data, create_flight_airport_features, extract_from_messages

def load_model(model_path: Path) -> LGBMClassifier:
    print(f"Loading airport model from {model_path}")
    return joblib.load(str(model_path))


def inference(df_flights: pl.DataFrame, model: LGBMClassifier):
    df_flights = add_flight_id_col(df_flights)
    X, flight_ids, airport_idents = build_inference_data(df_flights)
    scores = model.predict_proba(X)[:,1]
    results = pl.DataFrame({
        "flight_id": flight_ids,
        "airport_ident": airport_idents,
        "score": scores
    })

    predictions = (
    results
    .sort(["flight_id", "score"], descending=[False, True])
    .group_by("flight_id", maintain_order=True)
    .head(1)
    )
    df_flights = df_flights.join(predictions, on="flight_id", how="left")
    df_flights = df_flights.with_columns(pl.col("airport_ident").alias("takeoff_airport_ident"))
    return df_flights



def run_inference(run_date: date, model_path: Path):
    df_flights = get_flights(run_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    model = load_model(model_path)
    df_flights = inference(df_flights, model)
    return df_flights

MODEL_PATH = '/Volumes/T2-SSD/planequery/data/models/flights_ai_model_airport/runs/2026-07-01_13-35_c02a24e_5days/model.pkl'
def get_airport_model_flights(target_date: date, model_path: Path = Path(MODEL_PATH)):
    df = run_inference(target_date, model_path)
    return df

# probably a run_inference with a date

if __name__ == "__main__":
    pass
    # df_flights = run_inference(date(2026,3,15), )
    # print(df_flights)
