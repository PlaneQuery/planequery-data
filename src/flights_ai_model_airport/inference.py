from datetime import date
import polars as pl
import joblib
from lightgbm import LGBMClassifier

from data_engineering.flights.flight_type import Flight, add_flight_id_col, flights_df_to_flights, get_flights
from flights_ai_model_airport.features import build_inference_data
from data_engineering.utils import OUTPUT_DIR
path = OUTPUT_DIR / "tmp" / "flights_ai_model_airport.txt"
from data_engineering.utils import OUTPUT_DIR
MODEL_PATH =  OUTPUT_DIR / "tmp" / "flights_ai_model_airport.pkl"

def inference(df_flights: pl.DataFrame, model:  LGBMClassifier = joblib.load(str(MODEL_PATH))):
    df_flights = add_flight_id_col(df_flights)
    flights = flights_df_to_flights(df_flights)
    X, flight_ids, airport_idents = build_inference_data(flights)
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



def run_inference(run_date: date = date(2026,6,1)):
    df_flights = get_flights(run_date)
    model = joblib.load(str(MODEL_PATH))
    df_flights = inference(df_flights, model)
    return df_flights

def get_airport_model_flights(target_date: date):
    df_flights = get_flights(target_date)
    df_flights = inference(df_flights)
    return df_flights
# probably a run_inference with a date

if __name__ == "__main__":
    run_inference()