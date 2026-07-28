from datetime import datetime, timedelta
import os

import numpy as np
import polars as pl

os.environ.setdefault("OUTPUT_DIR", "/tmp/planequery-test-output")
AIRPORTS_PATH = "/tmp/planequery-test-output/data/raw/ourairports/airports.csv"
os.makedirs(os.path.dirname(AIRPORTS_PATH), exist_ok=True)
if not os.path.exists(AIRPORTS_PATH):
    with open(AIRPORTS_PATH, "w", encoding="utf-8") as airport_csv:
        airport_csv.write(
            "ident,type,latitude_deg,longitude_deg,elevation_ft,iata_code,gps_code,local_code,iso_country,iso_region\n"
            "KAAA,small_airport,0,0,0,AAA,KAAA,AAA,US,US-AA\n"
            "KBBB,small_airport,1,1,0,BBB,KBBB,BBB,US,US-BB\n"
        )

from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from data_engineering.adsb.adsb_messages_types import AdsbMessage
from airports.airport_lookup import Airport, Airport_Types
from flights_ai_model_airport.features import (
    create_flight_airport_features,
    extract_from_messages,
)
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


def test_extract_from_messages_is_invariant_to_input_order():
    start = datetime(2026, 3, 1, 10, 0)
    messages = [
        AdsbMessage(
            time=start + timedelta(minutes=offset),
            icao="abc123",
            lat=float(offset),
            lon=-float(offset),
            baro_altitude_ft=offset * 1_000,
            track_deg=float(offset * 10),
        )
        for offset in (10, 0, 20)
    ]

    takeoff = extract_from_messages(messages, "takeoff")
    landing = extract_from_messages(messages, "landing")

    assert takeoff == (start, 0.0, -0.0, 0, 0.0)
    assert landing == (
        start + timedelta(minutes=20),
        20.0,
        -20.0,
        20_000,
        200.0,
    )


def test_extract_from_messages_keeps_position_and_motion_at_same_timestamp():
    start = datetime(2026, 3, 1, 10, 0)
    messages = [
        AdsbMessage(
            time=start,
            icao="abc123",
            lat=1.0,
            lon=-1.0,
            baro_altitude_ft=None,
            track_deg=10.0,
        ),
        AdsbMessage(
            time=start + timedelta(minutes=1),
            icao="abc123",
            lat=2.0,
            lon=-2.0,
            baro_altitude_ft=1_000,
            track_deg=20.0,
        ),
    ]

    assert extract_from_messages(messages, "takeoff") == (
        start + timedelta(minutes=1),
        2.0,
        -2.0,
        1_000,
        20.0,
    )


def test_candidate_features_keep_existing_nearest_airport(monkeypatch):
    import flights_ai_model_airport.features as features_module

    nearby = Airport(
        ident="KAAA",
        iata="AAA",
        lat=0.0,
        lon=0.0,
        elevation_ft=0,
        type=Airport_Types.SMALL_AIRPORT,
    )
    baseline = Airport(
        ident="KBBB",
        iata="BBB",
        lat=2.0,
        lon=2.0,
        elevation_ft=0,
        type=Airport_Types.SMALL_AIRPORT,
    )

    class FakeAirportLookup:
        def getAirportsWithinRadius(self, lat, lon, radius_km):
            return [nearby]

        def get_Airport_from_airport_ident(self, ident):
            return baseline if ident == baseline.ident else None

    monkeypatch.setattr(features_module, "airport_lookup", FakeAirportLookup())
    flight = Flight(
        icao="abc123",
        takeoff_time=datetime(2026, 3, 1, 10, 0),
        landing_time=datetime(2026, 3, 1, 11, 0),
        takeoff_airport_ident=baseline.ident,
        landing_airport_ident="KAAA",
    )

    rows = create_flight_airport_features(
        lat=0.0,
        lon=0.0,
        baro_altitude_ft=1_000,
        track=90.0,
        flight=flight,
        endpoint="takeoff",
    )

    assert [airport_ident for _features, _label, airport_ident in rows] == [
        nearby.ident,
        baseline.ident,
    ]
    assert [label for _features, label, _airport_ident in rows] == [0, 1]


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
    from flights_ai_model_airport.features import FEATURE_NAMES

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
            np.ones((1, len(FEATURE_NAMES))),
            np.array(["abc123_2026-03-01_10-00"]),
            np.array(["KAAA"]),
        )

    predict_inputs = []

    class FakeModel:
        def predict_proba(self, X):
            predict_inputs.append(X)
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
    assert all(isinstance(X, np.ndarray) for X in predict_inputs)
