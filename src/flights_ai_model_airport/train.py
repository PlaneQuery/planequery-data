from datetime import date, datetime, timedelta
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
from flights.flights_match_adsb import get_matching_icaos_in_flights
from flights_ai_model_airport.features import build_training_data
from flights_ai_model_airport.flights_ai_model_utils import AIRPORT_MODEL_ENDPOINTS
from flights_ai_model_airport.inference import run_inference
from flights_ai_model_airport.model import model_training
from flights_ai_model_airport.flights_ai_model_utils import get_model_path
from flights.evaluation.evaluation import main

def train(training_dates, validation_date = date(2026,3,15)):
    # test_dates = [date(2026,3,1), date(2026,3,15)]
    models = {}
    for endpoint in AIRPORT_MODEL_ENDPOINTS:
        X, y, groups, airport_idents = build_training_data(training_dates, endpoint=endpoint)
        models[endpoint] = model_training(X, y, groups, airport_idents, endpoint=endpoint)

    # do val
    import joblib
    path = get_model_path(num_days=len(training_dates)) / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models},
        str(path),
    )
    print(f"  -> saved to {path}")
    return path

if __name__ == "__main__":
    training_dates = [
        date(2026, 6, 1) + timedelta(days=i)
        for i in range((date(2026, 6, 8) - date(2026, 6, 1)).days + 1)
    ]
    print(training_dates)
    train(training_dates)
