from datetime import datetime
import os

import numpy as np
import polars as pl

os.environ.setdefault("OUTPUT_DIR", "/tmp/planequery-test-output")

from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights_ai_model_airport.inference import set_consecutive_airport_idents_from_scores


def _two_consecutive_flights() -> pl.DataFrame:
    return flights_to_flights_df(
        [
            Flight(
                icao="abc123",
                takeoff_time=datetime(2026, 3, 1, 10, 0),
                landing_time=datetime(2026, 3, 1, 11, 0),
                takeoff_airport_ident="KZZZ",
                landing_airport_ident="KAAA",
            ),
            Flight(
                icao="abc123",
                takeoff_time=datetime(2026, 3, 1, 12, 0),
                landing_time=datetime(2026, 3, 1, 13, 0),
                takeoff_airport_ident="KBBB",
                landing_airport_ident="KYYY",
            ),
        ]
    )


def _scores(flight_id: str, rows: list[tuple[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "flight_id": [flight_id for _ in rows],
            "airport_ident": [airport_ident for airport_ident, _ in rows],
            "score": [score for _, score in rows],
        }
    )


def test_set_consecutive_airport_idents_uses_shared_airport_when_score_tradeoff_is_close():
    df_flights = _two_consecutive_flights()
    landing_scores = _scores(
        "abc123_2026-03-01_10-00",
        [
            ("KAAA", 0.95),
            ("KBBB", 0.10),
        ],
    )
    takeoff_scores = _scores(
        "abc123_2026-03-01_12-00",
        [
            ("KBBB", 0.20),
            ("KAAA", 0.18),
        ],
    )

    result = set_consecutive_airport_idents_from_scores(
        df_flights,
        landing_scores,
        takeoff_scores,
    )

    assert result["landing_airport_ident"][0] == "KAAA"
    assert result["takeoff_airport_ident"][1] == "KAAA"


def test_set_consecutive_airport_idents_keeps_different_airports_when_scores_disagree():
    df_flights = _two_consecutive_flights()
    landing_scores = _scores(
        "abc123_2026-03-01_10-00",
        [
            ("KAAA", 0.95),
            ("KBBB", 0.10),
        ],
    )
    takeoff_scores = _scores(
        "abc123_2026-03-01_12-00",
        [
            ("KBBB", 0.90),
            ("KAAA", 0.20),
        ],
    )

    result = set_consecutive_airport_idents_from_scores(
        df_flights,
        landing_scores,
        takeoff_scores,
    )

    assert result["landing_airport_ident"][0] == "KAAA"
    assert result["takeoff_airport_ident"][1] == "KBBB"


def test_inference_passes_preloaded_adsb_to_both_endpoint_feature_builders(monkeypatch):
    import flights_ai_model_airport.inference as inference_module

    df_flights = flights_to_flights_df(
        [
            Flight(
                icao="abc123",
                takeoff_time=datetime(2026, 3, 1, 10, 0),
                landing_time=datetime(2026, 3, 1, 11, 0),
                takeoff_airport_ident="",
                landing_airport_ident="",
            ),
        ]
    )
    df_adsb_full = pl.DataFrame({"icao": ["abc123"]})
    calls = []

    def fake_build_inference_data(df, df_adsb_full, endpoint="takeoff"):
        calls.append((endpoint, df_adsb_full))
        return (
            np.ones((1, len(inference_module.FEATURE_NAMES))),
            np.array(["abc123_2026-03-01_10-00"]),
            np.array(["KAAA"]),
        )

    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9] for _ in range(len(X))])

    monkeypatch.setattr(
        inference_module,
        "build_inference_data",
        fake_build_inference_data,
    )

    inference_module.inference(
        df_flights,
        {"takeoff": FakeModel(), "landing": FakeModel()},
        df_adsb_full=df_adsb_full,
    )

    assert [endpoint for endpoint, _df_adsb in calls] == ["takeoff", "landing"]
    assert all(df_adsb is df_adsb_full for _endpoint, df_adsb in calls)
