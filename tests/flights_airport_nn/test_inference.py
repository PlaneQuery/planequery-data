from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from airports.airport_lookup import Airport, Airport_Types
from data_engineering.adsb.adsb_messages_types import AdsbMessage
from data_engineering.flights.flight_type import Flight
from flights_airport_nn.dataset import collate_airport_candidates
from flights_airport_nn.features import features as ff
from flights_airport_nn.inference import inference, load_model, predict
from flights_airport_nn.model import AirportGRU


@pytest.fixture
def scene(monkeypatch):
    near = Airport(1, "NEAR", Airport_Types.SMALL_AIRPORT, "near", 40.0, -75.0, 100)
    far = Airport(2, "FAR", Airport_Types.LARGE_AIRPORT, "far", 42.0, -75.0, 100)
    lookup = SimpleNamespace(
        ident_index={"NEAR": near, "FAR": far},
        getAirportsWithinRadius=lambda *args: [near],
    )
    monkeypatch.setattr(ff, "AirportLookup", lambda: lookup)
    start = datetime(2026, 2, 1, 12)
    flight = Flight("abc123", start, start + timedelta(minutes=3), "FAR", "FAR")
    messages = [
        AdsbMessage(
            start + timedelta(seconds=s),
            "abc123",
            40 + s / 100000,
            -75.0,
            baro_altitude_ft=None,
            ground_speed_kt=120,
        )
        for s in range(-60, 601, 10)
    ]
    return flight, messages, near, far


def test_candidates_do_not_depend_on_ground_truth(scene):
    flight, messages, near, far = scene
    before = ff.create_features_for_flight(flight, messages)
    flight.takeoff_airport_ident = "NEAR"
    after = ff.create_features_for_flight(flight, messages)
    assert [c.predicted_airport for c in before.candidates] == [near]
    np.testing.assert_array_equal(
        before.candidates[0].features, after.candidates[0].features
    )
    assert before.candidates[0].true_airport == far


def test_both_endpoints_stay_within_segment_and_handle_missing(scene):
    flight, messages, _, _ = scene
    for endpoint in ("takeoff", "landing"):
        result = ff.create_features_for_flight(flight, messages, endpoint=endpoint)
        x = result.candidates[0].features
        assert len(x) == 19
        assert np.isfinite(x).all()
        assert x[:, 10].max() == pytest.approx(0.3)
        assert (x[:, 18] == 1).all()
        assert (x[:, 17] == int(endpoint == "landing")).all()
    landing = ff.create_features_for_flight(flight, messages, endpoint="landing")
    assert landing.candidates[0].features[0, 0] > 0


def test_empty_invalid_and_duplicate_messages(scene):
    flight, messages, _, _ = scene
    assert not ff.create_features_for_flight(flight, []).candidates
    bad = AdsbMessage(flight.takeoff_time, "abc123", float("nan"), -75)
    result = ff.create_features_for_flight(flight, [bad, messages[6], messages[6]])
    assert len(result.candidates[0].features) == 1


def test_ranker_padding_candidate_order_and_gradients():
    torch.manual_seed(4)
    model = AirportGRU(24, 16, dropout=0).eval()
    # Nonzero output weights exercise the recurrent/attention paths too.
    torch.nn.init.normal_(model.output[-1].weight, std=0.1)
    x = torch.randn(2, 3, 7, 24)
    mask = torch.ones(2, 3, 7, dtype=torch.bool)
    original = model.rank(x, mask)
    padded = torch.cat([x, torch.randn(2, 3, 5, 24)], dim=2)
    padded_mask = torch.cat([mask, torch.zeros(2, 3, 5, dtype=torch.bool)], dim=2)
    torch.testing.assert_close(
        original, model.rank(padded, padded_mask), atol=1e-6, rtol=1e-5
    )
    order = [2, 0, 1]
    torch.testing.assert_close(
        original[:, order], model.rank(x[:, order], mask[:, order])
    )
    loss = torch.nn.functional.cross_entropy(original, torch.tensor([0, 2]))
    loss.backward()
    assert torch.isfinite(model.gru.weight_ih_l0.grad).all()
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0


def test_collation_masks_candidates_and_time():
    x, mask, _y = collate_airport_candidates(
        [(torch.ones(2, 4, 24), 1), (torch.ones(1, 2, 24), 0)]
    )
    assert x.shape == (2, 2, 4, 24)
    assert not mask[1, 1].any()
    assert mask[1, 0].sum() == 2
    scores = AirportGRU(24, 16).rank(x, mask)
    assert scores[1, 1] < -1000


def test_inference_updates_both_and_clears_unavailable(scene):
    flight, messages, near, _ = scene
    takeoff = ff.create_features_for_flight(flight, messages)
    landing = ff.create_features_for_flight(flight, [], endpoint="landing")
    features = SimpleNamespace(
        endpoints=[takeoff, landing], flights_by_id={flight.flight_id: flight}
    )
    output = inference(features, AirportGRU(24, 16))
    assert output[flight.flight_id].takeoff_airport_ident == near.ident
    assert output[flight.flight_id].landing_airport_ident is None
    assert predict(
        AirportGRU(24, 16), np.empty((0, 1, 24)), np.empty((0, 1))
    ).shape == (0,)


def test_checkpoint_roundtrip_and_version_rejection(tmp_path):
    model = AirportGRU(24, 16, dropout=0).eval()
    path = tmp_path / "model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_names": [f.value for f in ff.FeatureName],
        "hidden_size": 16,
        "dropout": 0,
        "feature_version": ff.FEATURE_VERSION,
        "distance_index": 4,
        "temperature": 1.5,
    }
    torch.save(checkpoint, path)
    restored, _config = load_model(path)
    x = np.random.default_rng(1).normal(size=(2, 5, 24)).astype("float32")
    mask = np.ones((2, 5), bool)
    np.testing.assert_allclose(
        predict(model, x, mask), predict(restored, x, mask), atol=1e-5
    )
    assert restored.temperature == 1.5
    checkpoint["feature_version"] = 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(path)


def test_evaluation_counts_candidate_misses():
    from flights_airport_nn.benchmark import summarize

    frame = pl.DataFrame(
        {
            "date": ["2026-03-01"] * 2,
            "endpoint": ["takeoff", "landing"],
            "truth": ["A", "B"],
            "correct": [True, False],
            "retrieved": [True, False],
            "top3": [True, False],
            "nearest_correct": [True, False],
            "confidence": [0.9, 0.8],
            "distance_km": [1.0, 20.0],
        }
    )
    result = summarize(frame)
    assert result["overall"]["accuracy"] == 0.5
    assert result["overall"]["candidate_recall"] == 0.5
    assert result["distant_endpoint"]["accuracy"] == 0


def test_training_rejects_date_leakage():
    from datetime import date

    from flights_airport_nn.train import run_train

    day = date(2026, 2, 1)
    with pytest.raises(ValueError, match="disjoint"):
        run_train([day], validation_dates=[day])
    with pytest.raises(ValueError, match="validation_dates"):
        run_train([day])


def test_train_evaluate_pipeline_and_airport_holdout(tmp_path, monkeypatch):
    from datetime import date

    from flights_airport_nn import dataset
    from flights_airport_nn.benchmark import evaluate_dates
    from flights_airport_nn.train import run_train

    monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "cache")
    days = [date(2026, 2, 1), date(2026, 2, 2)]
    for day in days:
        path = dataset.cache_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        x = np.zeros((4, 2, 3, 24), np.float16)
        x[:, 0, :, 4] = 0.1
        x[:, 1, :, 4] = 0.5
        np.savez(
            path,
            x=x,
            lengths=np.full(4, 3),
            counts=np.full(4, 2),
            target=np.zeros(4, dtype=np.int64),
            airports=np.array([["A", "B"], ["A", "B"], ["HOLD", "B"], ["HOLD", "B"]]),
            truth=np.array(["A", "A", "HOLD", "HOLD"]),
            nearest=np.zeros(4, dtype=np.int64),
            endpoint=np.array(["takeoff", "landing", "takeoff", "landing"]),
            flight_id=np.array(["aaa_1", "aaa_1", "bbb_2", "bbb_2"]),
        )
    training = dataset.AirportCandidateDataset([days[0]], excluded_airports=["HOLD"])
    assert len(list(training)) == 2
    model = run_train(
        [days[0]],
        validation_dates=[days[1]],
        epochs=1,
        output_dir=tmp_path / "run",
        hidden_size=8,
        batch_size=2,
        excluded_airports=["HOLD"],
        prepare=False,
    )
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert checkpoint["epoch"] == 1 and checkpoint["temperature"] > 0
    selected, _, _ = evaluate_dates(
        model, [days[1]], ff.ALL_FEATURE_CONFIG, excluded_airports=["HOLD"]
    )
    assert set(selected["truth"]) == {"A"}
    all_rows, _, _ = evaluate_dates(model, [days[1]], ff.ALL_FEATURE_CONFIG)
    assert all_rows.height == 4


def test_batched_inference_preserves_endpoint_probabilities(scene):
    from flights_airport_nn.inference import predict_endpoints

    flight, messages, near, far = scene
    first = ff.create_features_for_flight(flight, messages)
    short = first.candidates[0].features[:3].copy()
    alternative = short.copy()
    alternative[:, 4] += 1
    second = ff.FlightFeatures(
        "another_flight",
        None,
        [
            ff.AirportCandidateFeatures("another_flight", near, None, short, "landing"),
            ff.AirportCandidateFeatures(
                "another_flight", far, None, alternative, "landing"
            ),
        ],
        "landing",
    )
    features = SimpleNamespace(endpoints=[first, second])
    model = AirportGRU(24, 16, dropout=0).eval()
    torch.nn.init.normal_(model.output[-1].weight, std=0.1)
    individual = predict_endpoints(features, model, endpoint_batch_size=1)
    batched = predict_endpoints(features, model, endpoint_batch_size=16)
    for a, b in zip(individual, batched):
        assert (a["flight_id"], a["endpoint"], a["airport_ident"]) == (
            b["flight_id"],
            b["endpoint"],
            b["airport_ident"],
        )
        assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-6)
        assert sum(p["confidence"] for p in b["alternatives"]) == pytest.approx(1)
