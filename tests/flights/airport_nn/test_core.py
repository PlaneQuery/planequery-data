from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json

import numpy as np
import polars as pl
import pytest
import torch

from airports.models import (
    Airport,
    AirportCatalogProvenance,
    AirportMatch,
    AirportMatchProvenance,
    AirportType,
)
from flights.airport_nn import data, features
from flights.airport_nn import derive_day as daily
from flights.airport_nn.evaluate import metrics, score, write_report
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
from flights.providers.bts import BTSFlightSource
from flights.providers.sfdps import SFDPSFlightSource


@dataclass(frozen=True)
class FakeAirportMatcher:
    airport: Airport

    def _match(self, lat: float, lon: float) -> AirportMatch:
        return AirportMatch(
            id=self.airport.id,
            ident=self.airport.ident,
            distance_km=self.airport.distance_to(lat, lon),
            provenance=AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE,
            source_commit_sha="test-commit",
            airport=self.airport,
        )

    def nearest(self, lat: float, lon: float) -> AirportMatch:
        return self._match(lat, lon)

    def within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> tuple[AirportMatch, ...]:
        return (self._match(lat, lon),)


@dataclass(frozen=True)
class FakeAirportCatalog:
    matcher: FakeAirportMatcher
    provenance: AirportCatalogProvenance

    @property
    def airports(self):
        return (self.matcher.airport,)

    @property
    def id_index(self):
        return {self.matcher.airport.id: self.matcher.airport}


@pytest.fixture
def observation():
    airport = Airport(
        id=1,
        ident="TEST",
        type=AirportType.SMALL_AIRPORT,
        name="Test",
        lat=40,
        lon=-75,
        elevation_ft=100,
    )
    airport_catalog = FakeAirportCatalog(
        matcher=FakeAirportMatcher(airport),
        provenance=AirportCatalogProvenance(
            flight_date=date(2026, 2, 1),
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 3, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 3, 1),
            policy_version="test",
        ),
    )
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
    return flights, messages, airport_catalog


def test_features_ignore_labels_and_neighboring_flights(observation):
    flights, messages, airport_catalog = observation
    original = list(features.trajectories(flights, messages, airport_catalog))
    changed = flights.with_columns(
        pl.lit(999).alias("takeoff_airport_id"),
        pl.lit("LEAK").alias("takeoff_airport_ident"),
    )
    neighbors = messages.with_columns(pl.col("time") + pl.duration(hours=2))
    actual = list(
        features.trajectories(
            changed,
            pl.concat([messages, neighbors]),
            airport_catalog,
        )
    )
    for a, b in zip(original, actual):
        np.testing.assert_array_equal(a.values, b.values)
        assert a.candidate_idents == b.candidate_idents
        assert a.candidate_ids == b.candidate_ids
        assert a.nearest_id == b.nearest_id == 1
    assert original[0].values.shape[1] == 3
    assert original[1].values.shape[1] == 2


def test_missing_observations_clear_labels_and_preserve_schema(observation):
    flights, messages, airport_catalog = observation
    model = AirportRanker(8)
    torch.set_num_threads(1)
    predicted = predict_flights(flights, messages, model, airport_catalog)
    missing = predict_flights(flights, messages.head(0), model, airport_catalog)
    labels = [
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ]
    assert predicted.schema == flights.schema == missing.schema
    assert predicted.drop(labels).equals(flights.drop(labels))
    assert missing.drop(labels).equals(flights.drop(labels))
    assert predicted.select(labels).row(0) == (1, "TEST", 1, "TEST")
    assert missing.select(labels).row(0) == (None, None, None, None)


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
                candidate_ids=[1],
                candidate_idents=["TEST"],
                values=np.zeros((1, 1, features.INPUT_SIZE), dtype=np.float32),
            )

    predictions = predict(endpoint_stream(), model, batch_size=2)

    assert len(predictions) == 5
    assert forward_calls == 3


def test_derive_day_writes_predicted_flights(observation, monkeypatch, tmp_path):
    flights, messages, airport_catalog = observation
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
    monkeypatch.setattr(
        daily,
        "airport_catalog_for_date",
        lambda _day: airport_catalog,
    )
    output = tmp_path / "data.parquet"
    monkeypatch.setattr(daily, "flight_path", lambda *args: output)
    day = date(2026, 2, 1)
    result = daily.derive_day(day, model_path=checkpoint, flights=flights)

    predicted = pl.read_parquet(result)
    assert result == output
    assert predicted.schema == flights.schema
    assert predicted["takeoff_airport_ident"].to_list() == ["TEST"]
    assert predicted["landing_airport_ident"].to_list() == ["TEST"]
    assert predicted["takeoff_airport_id"].to_list() == [1]
    assert predicted["landing_airport_id"].to_list() == [1]
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


def test_references_label_stable_ids_not_idents(observation, monkeypatch, tmp_path):
    _, _, airport_catalog = observation
    day = date(2026, 2, 1)
    start = datetime(2026, 2, 1, 12)
    paths = {}
    frames = {}
    for source in (SFDPSFlightSource(), BTSFlightSource()):
        path = tmp_path / f"{source.name}.parquet"
        paths[source.name] = path
        frames[source.name] = pl.DataFrame(
            {
                "icao": ["abc123", "def456"],
                "takeoff_time": [start, start + timedelta(hours=2)],
                "landing_time": [
                    start + timedelta(hours=1),
                    start + timedelta(hours=3),
                ],
                "takeoff_airport_id": [1, 999],
                "takeoff_airport_ident": [f"{source.name}-ALIAS", "TEST"],
                "landing_airport_id": [1, 999],
                "landing_airport_ident": [f"{source.name}-ALIAS", "TEST"],
            }
        )
        frames[source.name].write_parquet(path)

    monkeypatch.setattr(data, "airport_catalog_for_date", lambda _: airport_catalog)
    monkeypatch.setattr(data, "flight_path", lambda source, _: paths[source.name])
    catalog_calls = []
    monkeypatch.setattr(
        data,
        "get_flights_for_catalog",
        lambda source, requested_day, catalog: (
            catalog_calls.append((source.name, requested_day, catalog))
            or frames[source.name]
        ),
    )

    refs, _ = data.references(day)

    valid = refs.filter(pl.col("icao") == "abc123")
    invalid = refs.filter(pl.col("icao") == "def456")
    assert valid["takeoff_truth_id"].to_list() == [1, 1]
    assert set(valid["takeoff_truth_ident"]) == {"bts-ALIAS", "sfdps-ALIAS"}
    assert invalid["takeoff_truth_id"].null_count() == 2
    assert invalid["takeoff_truth_ident"].to_list() == ["", ""]
    assert catalog_calls == [
        ("sfdps", day, airport_catalog),
        ("bts", day, airport_catalog),
    ]


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


def test_prepare_defaults_workers_from_machine_resources(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        data,
        "default_worker_count",
        lambda: calls.append("default") or 1,
    )
    monkeypatch.setattr(data, "ensure_airport_history", lambda: object())

    data.prepare([], output=tmp_path)

    assert calls == ["default"]


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
            "truth_id": [1, 2, 3],
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


def test_score_and_report_use_ids_for_correctness(tmp_path):
    part = tmp_path / "part-0000.npz"
    x = np.zeros((2, features.CANDIDATES, features.POINTS, features.INPUT_SIZE))
    mask = np.zeros(x.shape[:3], dtype=bool)
    mask[:, 0, 0] = True
    np.savez_compressed(
        part,
        x=x,
        mask=mask,
        target=np.array([0, 0]),
        eligible=np.array([True, True]),
    )
    pl.DataFrame(
        {
            "truth_id": [1, 1],
            "truth_ident": ["SAME", "OLD"],
            "nearest_id": [2, 1],
            "nearest_ident": ["SAME", "NEW"],
            "candidate_ids": [[2], [1]],
            "candidate_idents": [["SAME"], ["NEW"]],
            "target": [0, 0],
            "eligible": [True, True],
            "airport_type": ["small_airport", "small_airport"],
            "icao": ["a", "b"],
        }
    ).write_parquet(part.with_suffix(".parquet"))

    scored, _ = score(AirportRanker(8), [part])

    assert scored["nn_id"].to_list() == [2, 1]
    assert scored["nn_ident"].to_list() == ["SAME", "NEW"]
    assert scored["nn_correct"].to_list() == [False, True]
    assert scored["nearest_correct"].to_list() == [False, True]

    report = tmp_path / "report"
    write_report(scored, report)
    metadata = json.loads((report / "metadata.json").read_text())
    assert metadata["airport_identity"] == "ourairports_id"
    assert metadata["target_definition"] == "index of truth_id in candidate_ids"
    assert metadata["display_only_fields"] == [
        "truth_ident",
        "nn_ident",
        "nearest_ident",
        "candidate_idents",
    ]


def test_train_accepts_one_day_without_validation(tmp_path):
    day = date(2026, 3, 1)
    folder = tmp_path / "cache" / str(day)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        '{"date": "2026-03-01", "network": "adsblol", '
        '"airport_identity": "ourairports_id", "parts": 1}'
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
