from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from airports.airport_lookup import Airport, Airport_Types
from flights.airport_nn import data, features
from flights.airport_nn import derive_day as daily
from flights.airport_nn.evaluate import metrics
from flights.airport_nn.inference import predict, predict_flights
from flights.airport_nn.model import AirportRanker, default_device
from flights.airport_nn.train import (
    BUFFER_DAYS,
    TEST_DAYS,
    TRAIN_DAYS,
    VALIDATION_DAYS,
    selection_loss,
    train,
)
from flights.flight_type import Flight, flights_to_flights_df


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


def test_prediction_streams_endpoint_batches():
    model = AirportRanker(8)
    original_forward = model.forward
    forward_calls = 0

    def tracked_forward(values, mask):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(values, mask)

    model.forward = tracked_forward

    def endpoint_stream():
        for row in range(5):
            if row == 2:
                assert forward_calls == 1
            yield features.Endpoint(
                row=row,
                endpoint="takeoff",
                airports=["TEST"],
                values=np.zeros((1, 1, features.INPUT_SIZE), dtype=np.float32),
            )

    predictions = predict(endpoint_stream(), model, batch_size=2)

    assert len(predictions) == 5
    assert forward_calls == 3


def test_derive_day_writes_predicted_flights(observation, monkeypatch, tmp_path):
    flights, messages = observation
    checkpoint = tmp_path / "model.pt"
    model = AirportRanker(8)
    payload = {
        "weights": model.state_dict(),
        "hidden_size": 8,
        "feature_version": features.VERSION,
    }
    torch.save(payload, checkpoint)
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return messages

    monkeypatch.setattr(daily, "read_adsb", read)
    output = tmp_path / "data.parquet"
    monkeypatch.setattr(daily, "flight_path", lambda *args: output)
    day = date(2026, 2, 1)
    result = daily.derive_day(day, model_path=checkpoint, flights=flights)

    predicted = pl.read_parquet(result)
    assert result == output
    assert predicted.schema == flights.schema
    assert predicted["takeoff_airport_ident"].to_list() == ["TEST"]
    assert predicted["landing_airport_ident"].to_list() == ["TEST"]
    assert len(calls) == 1


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


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_default_device_priority(
    monkeypatch,
    cuda_available,
    mps_available,
    expected,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert default_device() == torch.device(expected)


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


def test_split_is_disjoint_and_february_only():
    all_days = TRAIN_DAYS + VALIDATION_DAYS + BUFFER_DAYS + TEST_DAYS
    assert len(all_days) == len(set(all_days)) == 30
    assert TRAIN_DAYS == [date(2026, 2, d) for d in range(1, 21)]
    assert VALIDATION_DAYS == [date(2026, 2, d) for d in range(21, 28)]
    assert BUFFER_DAYS == [date(2026, 2, 28)]
    assert TEST_DAYS == [date(2026, 3, 1), date(2024, 6, 1)]
    assert all(data.february(d) == d for d in TRAIN_DAYS)
    for day in [date(2026, 1, 31), date(2026, 3, 1), date(2025, 2, 1)]:
        with pytest.raises(ValueError, match="February 2026"):
            data.february(day)


def test_selection_weights_days_equally_and_requires_every_day():
    losses = {str(d): i / 10 for i, d in enumerate(VALIDATION_DAYS, 1)}
    assert selection_loss(losses) == pytest.approx(0.4)
    losses[str(VALIDATION_DAYS[0])] = float("inf")
    with pytest.raises(ValueError, match="Every validation day"):
        selection_loss(losses)
    del losses[str(VALIDATION_DAYS[0])]
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


def test_train_accepts_one_day_without_validation(tmp_path):
    day = date(2026, 3, 1)
    folder = tmp_path / "cache" / str(day)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        '{"date": "2026-03-01", "network": "adsblol", "parts": 1}'
    )
    x = np.zeros((2, features.CANDIDATES, features.POINTS, features.INPUT_SIZE))
    mask = np.zeros(x.shape[:3], dtype=bool)
    mask[:, :2, :3] = True
    np.savez_compressed(
        folder / "part-0000.npz",
        x=x,
        mask=mask,
        target=np.array([0, 1]),
        eligible=np.array([True, True]),
    )
    pl.DataFrame(
        {
            "date": [str(day), str(day)],
            "airport_type": ["large_airport", "small_airport"],
        }
    ).write_parquet(folder / "part-0000.parquet")

    model_path = train(
        cache=tmp_path / "cache",
        output=tmp_path / "model",
        epochs=1,
        batch_size=2,
        train_days=[day],
        validation_days=[],
    )

    assert model_path == tmp_path / "model/model.pt"
    assert model_path.exists()
