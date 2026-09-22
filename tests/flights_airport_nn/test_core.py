from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from airports.airport_lookup import Airport, Airport_Types
from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights_airport_nn import data, features
from flights_airport_nn import derive_day as daily
from flights_airport_nn.evaluate import metrics
from flights_airport_nn.inference import predict_flights
from flights_airport_nn.model import AirportRanker
from flights_airport_nn.train import (
    BUFFER_DAYS,
    EUROPE_VALIDATION_DAYS,
    TEST_DAYS,
    TRAIN_DAYS,
    VALIDATION_DAYS,
    selection_loss,
)


@pytest.fixture
def observation(monkeypatch):
    airport = Airport(1, "TEST", Airport_Types.SMALL_AIRPORT, "Test", 40, -75, 100)

    class Lookup:
        def getClosestAirport(self, lat, lon):
            return airport

        def getAirportsWithinRadius(self, lat, lon, radius):
            return [airport]

    monkeypatch.setattr(features, "AirportLookup", Lookup)
    start = datetime(2026, 2, 1, 12)  # noqa: DTZ001 - source timestamps are naive UTC
    flights = flights_to_flights_df(
        [Flight("abc123", start, start + timedelta(minutes=30), "OLD", "OLD")]
    )
    messages = pl.DataFrame(
        {
            "icao": ["abc123"] * 5,
            "time": [start + timedelta(minutes=i) for i in (0, 1, 2, 29, 30)],
            "lat": [40.0] * 5,
            "lon": [-75.0] * 5,
            "baro_altitude_ft": [100.0, 1000.0, 2000.0, 1000.0, 100.0],
            "ground_speed_kt": [20.0, 100.0, 150.0, 100.0, 20.0],
            "track_deg": [90.0] * 5,
            "baro_rate_fpm": [0.0] * 5,
            "on_ground": [True, False, False, False, True],
        }
    )
    return flights, messages


def test_features_ignore_labels_and_neighboring_flights(observation):
    flights, messages = observation
    original = list(features.trajectories(flights, messages))
    changed = flights.with_columns(pl.lit("LEAK").alias("takeoff_airport_ident"))
    neighbors = messages.with_columns(pl.col("time") + pl.duration(hours=2))
    actual = list(features.trajectories(changed, pl.concat([messages, neighbors])))
    for a, b in zip(original, actual):
        np.testing.assert_array_equal(a.values, b.values)
        assert a.airports == b.airports
    assert original[0].values.shape[1] == 3
    assert original[1].values.shape[1] == 2


def test_missing_observations_clear_labels_and_preserve_schema(observation):
    flights, messages = observation
    model = AirportRanker(8)
    torch.set_num_threads(1)
    predicted = predict_flights(flights, messages, model)
    missing = predict_flights(flights, messages.head(0), model)
    labels = ["takeoff_airport_ident", "landing_airport_ident"]
    assert predicted.schema == flights.schema == missing.schema
    assert predicted.drop(labels).equals(flights.drop(labels))
    assert missing.drop(labels).equals(flights.drop(labels))
    assert predicted.select(labels).row(0) == ("TEST", "TEST")
    assert missing.select(labels).row(0) == (None, None)


def test_daily_cache_tracks_flight_contents_and_model(
    observation, monkeypatch, tmp_path
):
    flights, messages = observation
    checkpoint = tmp_path / "model.pt"
    model = AirportRanker(8)
    payload = {
        "weights": model.state_dict(),
        "hidden_size": 8,
        "feature_version": features.VERSION,
    }
    torch.save(payload, checkpoint)
    raw = tmp_path / "raw/icao_bucket=0"
    raw.mkdir(parents=True)
    (raw / "data.parquet").write_bytes(b"fingerprint only")
    airports = tmp_path / "data/raw/ourairports/airports.csv"
    airports.parent.mkdir(parents=True)
    airports.write_text("fingerprint only")
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(daily, "_parquet_day_dir", lambda _: raw.parent)
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return messages

    monkeypatch.setattr(daily, "read_adsb", read)
    day = date(2026, 2, 1)
    first = daily.derive_day(day, model_path=checkpoint, flights=flights)
    assert pl.read_parquet(first).schema == flights.schema
    assert daily.derive_day(day, model_path=checkpoint, flights=flights) == first
    assert len(calls) == 1
    changed = flights.with_columns(pl.col("landing_time") + pl.duration(minutes=1))
    second = daily.derive_day(day, model_path=checkpoint, flights=changed)
    assert second != first
    with torch.no_grad():
        model.distance_weight.add_(1)
    torch.save(payload, checkpoint)
    third = daily.derive_day(day, model_path=checkpoint, flights=changed)
    assert third != second
    assert len(calls) == 3


def test_ranker_padding_and_candidate_permutation():
    torch.manual_seed(1)
    model = AirportRanker(8).eval()
    # Exercise the learned branch too, which starts with zero output weights.
    torch.nn.init.normal_(model.output[-1].weight)
    x = torch.randn(2, 3, 7, features.INPUT_SIZE)
    mask = torch.ones(2, 3, 7, dtype=torch.bool)
    expected = model(x, mask)
    padded = torch.cat([x, torch.randn(2, 3, 4, features.INPUT_SIZE) * 100], dim=2)
    padded_mask = torch.cat([mask, torch.zeros(2, 3, 4, dtype=torch.bool)], dim=2)
    torch.testing.assert_close(model(padded, padded_mask), expected)
    order = [2, 0, 1]
    torch.testing.assert_close(model(x[:, order], mask[:, order]), expected[:, order])


def test_temporal_matching_rejects_ambiguity():
    start = datetime(2026, 2, 1, 12)  # noqa: DTZ001 - source timestamps are naive UTC
    segments = pl.DataFrame(
        {
            "icao": ["abc"],
            "segment": [0],
            "takeoff_time": [start],
            "landing_time": [start + timedelta(hours=1)],
        }
    )
    refs = segments.drop("segment").with_columns(
        pl.lit("BTS").alias("source"),
        pl.lit("one").alias("reference_id"),
        pl.lit(False).alias("time_conflict"),
    )
    assert data.match_references(segments, refs).height == 1
    duplicate = refs.with_columns(pl.lit("two").alias("reference_id"))
    assert data.match_references(segments, pl.concat([refs, duplicate])).is_empty()
    independent = duplicate.with_columns(pl.lit("SFDPS").alias("source"))
    assert data.match_references(segments, pl.concat([refs, independent])).height == 2
    assert data.match_references(
        segments, refs.with_columns(pl.lit(True).alias("time_conflict"))
    ).is_empty()


def test_faa_codes_require_unambiguous_supported_identifiers(monkeypatch):
    rows = {
        "ONE": {
            "type": "small_airport",
            "iso_country": "US",
            "icao_code": "KONE",
            "gps_code": "KONE",
            "local_code": "ABC",
        },
        "TWO": {
            "type": "small_airport",
            "iso_country": "US",
            "icao_code": "KTWO",
            "gps_code": "KONE",
            "local_code": "1AB",
        },
    }
    monkeypatch.setattr(data, "catalog", lambda: rows)
    data.faa_codes.cache_clear()
    try:
        assert data.faa_codes() == {"KTWO": "TWO", "1AB": "TWO"}
    finally:
        data.faa_codes.cache_clear()


def test_split_is_disjoint_and_february_only():
    all_days = (
        TRAIN_DAYS + VALIDATION_DAYS + EUROPE_VALIDATION_DAYS + BUFFER_DAYS + TEST_DAYS
    )
    assert len(all_days) == len(set(all_days)) == 32
    assert TRAIN_DAYS == [date(2026, 2, d) for d in range(1, 21)]
    assert VALIDATION_DAYS == [date(2026, 2, d) for d in range(21, 28)]
    assert EUROPE_VALIDATION_DAYS == [date(2024, 6, 12), date(2024, 6, 15)]
    assert BUFFER_DAYS == [date(2026, 2, 28)]
    assert TEST_DAYS == [date(2026, 3, 1), date(2024, 6, 1)]
    assert all(data.february(d) == d for d in TRAIN_DAYS)
    for day in [date(2026, 1, 31), date(2026, 3, 1), date(2025, 2, 1)]:
        with pytest.raises(ValueError, match="February 2026"):
            data.february(day)


def test_selection_weights_regions_equally_and_requires_every_day():
    losses = {str(d): 0.1 for d in VALIDATION_DAYS}
    losses.update({str(d): 0.9 for d in EUROPE_VALIDATION_DAYS})
    assert selection_loss(losses) == pytest.approx(0.5)
    losses[str(EUROPE_VALIDATION_DAYS[0])] = float("inf")
    with pytest.raises(ValueError, match="Every validation day"):
        selection_loss(losses)
    del losses[str(EUROPE_VALIDATION_DAYS[0])]
    with pytest.raises(KeyError):
        selection_loss(losses)


def test_retrieval_failures_stay_in_accuracy_denominator():
    frame = pl.DataFrame(
        {
            "truth": ["ONE", "TWO", "THREE"],
            "icao": ["a", "b", "c"],
            "target": [0, -1, 0],
            "eligible": [True, True, False],
            "label_conflict": [False] * 3,
            "source": ["BTS"] * 3,
            "airport_type": ["small_airport"] * 3,
            "nn_correct": [True, False, True],
            "nearest_correct": [False, False, True],
        }
    )
    result = metrics(frame)
    assert result["primary"]["endpoints"] == 2
    assert result["primary"]["nn_accuracy"] == 0.5
    assert result["primary"]["candidate_recall"] == 0.5
