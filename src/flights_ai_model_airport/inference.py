from datetime import date
from pathlib import Path
from typing import Any
import warnings

import polars as pl
import joblib
from lightgbm import LGBMClassifier

from data_engineering.flights.flight_type import add_flight_id_col, get_flights, pia_or_american_ladd_icao_filter
from flights_ai_model_airport.features import build_inference_data
from flights_ai_model_airport.flights_ai_model_utils import AIRPORT_MODEL_ENDPOINTS

def load_model(model_path: Path):
    print(f"Loading airport model from {model_path}")
    return joblib.load(str(model_path))


def _model_for_endpoint(model_or_bundle: Any, endpoint: str) -> LGBMClassifier:
    if isinstance(model_or_bundle, Path):
        model_or_bundle = load_model(model_or_bundle)

    if isinstance(model_or_bundle, dict) and "models" in model_or_bundle:
        models = model_or_bundle["models"]
        if endpoint not in models:
            raise ValueError(f"Model bundle does not contain a {endpoint} airport model")
        return models[endpoint]

    if isinstance(model_or_bundle, dict) and endpoint in model_or_bundle:
        return model_or_bundle[endpoint]

    if endpoint == "takeoff":
        return model_or_bundle

    raise ValueError("Landing airport inference requires an endpoint model bundle")


def score_airport_candidates(
    df_flights: pl.DataFrame,
    model: LGBMClassifier | dict | Path,
    df_adsb_full: pl.DataFrame,
    endpoint: str = "takeoff",
):
    df_flights = add_flight_id_col(df_flights)
    model = _model_for_endpoint(model, endpoint)
    X, flight_ids, airport_idents = build_inference_data(
        df_flights,
        df_adsb_full,
        endpoint,
    )

    with warnings.catch_warnings(): # TODO: deal with.
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        scores = model.predict_proba(X)[:, 1]

    return (
        pl.concat(
            [
                pl.DataFrame({
                    "flight_id": flight_ids,
                    "airport_ident": airport_idents,
                    "score": scores,
                    "endpoint": endpoint,
                }),
                pl.DataFrame(X),
            ],
            how="horizontal",
        )
        .with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("flight_id")
            .alias("rank")
        )
        .sort(["flight_id", "rank"])
    )


def _join_endpoint_predictions(df_flights: pl.DataFrame, scores: pl.DataFrame, endpoint: str) -> pl.DataFrame:
    predictions = (
        scores
        .filter(pl.col("rank") == 1)
        .select(
            "flight_id",
            pl.col("airport_ident").alias(f"_{endpoint}_airport_ident"),
            pl.col("score").alias(f"{endpoint}_airport_score"),
        )
    )
    return (
        df_flights
        .join(predictions, on="flight_id", how="left")
        .with_columns(pl.col(f"_{endpoint}_airport_ident").alias(f"{endpoint}_airport_ident"))
        .drop(f"_{endpoint}_airport_ident")
    )


def inference(
    df_flights: pl.DataFrame,
    model: LGBMClassifier | dict | Path,
    df_adsb_full: pl.DataFrame,
):
    if isinstance(model, Path):
        model = load_model(model)

    takeoff_scores = score_airport_candidates(
        df_flights,
        model,
        df_adsb_full,
        "takeoff",
    )
    landing_scores = score_airport_candidates(
        df_flights,
        model,
        df_adsb_full,
        "landing",
    )

    df_flights = add_flight_id_col(df_flights)
    df_flights = _join_endpoint_predictions(df_flights, takeoff_scores, "takeoff")
    df_flights = _join_endpoint_predictions(df_flights, landing_scores, "landing")
    df_flights = set_consecutive_airport_idents_from_scores(df_flights, landing_scores, takeoff_scores)
    return df_flights

def run_inference(run_date: date, model_path: Path):
    from data_engineering.adsb.read_adsb import read_adsb

    df_flights = get_flights(run_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb_full = read_adsb(run_date, icaos=icaos)
    model = load_model(model_path)
    df_flights = inference(df_flights, model, df_adsb_full)
    return df_flights

def run_score_canidates(run_date: date, model_path: Path):
    from data_engineering.adsb.read_adsb import read_adsb

    df_flights = get_flights(run_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb_full = read_adsb(run_date, icaos=icaos)
    model = load_model(model_path)
    return {
        endpoint: score_airport_candidates(df_flights, model, df_adsb_full, endpoint)
        for endpoint in AIRPORT_MODEL_ENDPOINTS
    }

def set_consecutive_airport_idents_from_scores(
    df_flights: pl.DataFrame,
    landing_scores: pl.DataFrame,
    takeoff_scores: pl.DataFrame,
    same_airport_score_bonus: float = 0.05,
) -> pl.DataFrame:
    df_flights = add_flight_id_col(df_flights)
    boundary_pairs = (
        df_flights
        .sort(["icao", "takeoff_time"])
        .with_columns(
            pl.col("flight_id").shift(-1).over("icao").alias("next_flight_id")
        )
        .filter(pl.col("next_flight_id").is_not_null())
        .select(
            pl.col("flight_id").alias("landing_flight_id"),
            "next_flight_id",
        )
    )

    landing_candidates = landing_scores.select(
        pl.col("flight_id").alias("landing_flight_id"),
        "airport_ident",
        pl.col("score").alias("landing_score"),
    )
    takeoff_candidates = takeoff_scores.select(
        pl.col("flight_id").alias("next_flight_id"),
        "airport_ident",
        pl.col("score").alias("takeoff_score"),
    )
    best_landing_candidates = (
        landing_candidates
        .sort(["landing_flight_id", "landing_score"], descending=[False, True])
        .group_by("landing_flight_id")
        .head(1)
        .select(
            "landing_flight_id",
            pl.col("landing_score").alias("best_landing_score"),
        )
    )
    best_takeoff_candidates = (
        takeoff_candidates
        .sort(["next_flight_id", "takeoff_score"], descending=[False, True])
        .group_by("next_flight_id")
        .head(1)
        .select(
            "next_flight_id",
            pl.col("takeoff_score").alias("best_takeoff_score"),
        )
    )

    best_boundary_airports = (
        boundary_pairs
        .join(landing_candidates, on="landing_flight_id", how="inner")
        .join(takeoff_candidates, on=["next_flight_id", "airport_ident"], how="inner")
        .join(best_landing_candidates, on="landing_flight_id", how="inner")
        .join(best_takeoff_candidates, on="next_flight_id", how="inner")
        .with_columns((pl.col("landing_score") + pl.col("takeoff_score")).alias("score"))
        .with_columns(
            (
                pl.col("best_landing_score")
                + pl.col("best_takeoff_score")
            ).alias("best_independent_score")
        )
        # Continuity is a small prior, not a hard constraint.
        .filter(pl.col("score") + same_airport_score_bonus >= pl.col("best_independent_score"))
        .sort(
            ["landing_flight_id", "next_flight_id", "score"],
            descending=[False, False, True],
        )
        .group_by(["landing_flight_id", "next_flight_id"])
        .head(1)
        .select("landing_flight_id", "next_flight_id", "airport_ident")
    )

    landing_updates = best_boundary_airports.select(
        pl.col("landing_flight_id").alias("flight_id"),
        pl.col("airport_ident").alias("_landing_airport_ident"),
    )
    takeoff_updates = best_boundary_airports.select(
        pl.col("next_flight_id").alias("flight_id"),
        pl.col("airport_ident").alias("_takeoff_airport_ident"),
    )

    return (
        df_flights
        .join(landing_updates, on="flight_id", how="left")
        .join(takeoff_updates, on="flight_id", how="left")
        .with_columns(
            pl.coalesce("_landing_airport_ident", "landing_airport_ident").alias("landing_airport_ident"),
            pl.coalesce("_takeoff_airport_ident", "takeoff_airport_ident").alias("takeoff_airport_ident"),
        )
        .drop("_landing_airport_ident", "_takeoff_airport_ident")
    )

def get_airport_model_flights(target_date: date, model_path: Path | None = None):
    if model_path is None:
        from flights_ai_model_airport.flights_ai_model_utils import get_latest_airport_model_path

        model_path = get_latest_airport_model_path()
    df = run_inference(target_date, model_path)
    return df

# probably a run_inference with a date

if __name__ == "__main__":
    pass
    # df_flights = run_inference(date(2026,3,15), )
    # print(df_flights)
