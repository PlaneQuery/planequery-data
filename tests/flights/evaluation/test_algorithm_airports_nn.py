from datetime import date, datetime, timedelta

import pytest

from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights.evaluation import evaluation
from flights_airport_nn import derive_day


def test_algorithm_airports_nn_source_uses_unlabeled_segments_and_checkpoint(
    monkeypatch, tmp_path
):
    day = date(2026, 3, 1)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    start = datetime(2026, 3, 1, 12)  # noqa: DTZ001 - source timestamps are naive
    original = Flight(
        "abc123", start, start + timedelta(hours=1), "OLD_START", "OLD_END"
    )
    later = start + timedelta(hours=2)
    missing = Flight(
        "def456", later, later + timedelta(hours=1), "OLD_START", "OLD_END"
    )
    segments = flights_to_flights_df([original, missing])
    calls = []

    def get_segments(target_date, adsb_src, no_airports_model=False):
        calls.append((target_date, adsb_src, no_airports_model))
        return segments

    def derive(target_date, source, model_path, flights):
        assert target_date == day
        assert source == "adsblol"
        assert model_path == checkpoint
        assert flights.equals(segments)
        predicted = Flight(
            "abc123", start, start + timedelta(hours=1), "KAAA", "KBBB"
        )
        unavailable = Flight(
            "def456", later, later + timedelta(hours=1), None, None
        )
        path = tmp_path / "flights.parquet"
        flights_to_flights_df([predicted, unavailable]).write_parquet(path)
        return path

    monkeypatch.setattr(evaluation, "get_algorithm_flights_for_adsb_src", get_segments)
    monkeypatch.setattr(derive_day, "derive_day", derive)

    reader = evaluation.source_flights_reader(
        "algorithm-airports-nn",
        adsb_src="adsblol",
        airport_nn_model_path=checkpoint,
    )
    result = reader(day)
    assert calls == [(day, "adsblol", True)]
    assert result["takeoff_airport_ident"].to_list() == ["KAAA", None]
    assert result["landing_airport_ident"].to_list() == ["KBBB", None]
    assert result.columns == segments.columns
    assert evaluation.adsb_srcs_for_test_source(
        "algorithm-airports-nn", ["adsblol", "adsbx"]
    ) == ["adsblol", "adsbx"]


def test_algorithm_airports_nn_requires_checkpoint(tmp_path):
    reader = evaluation.source_flights_reader(
        "algorithm-airports-nn", airport_nn_model_path=tmp_path / "missing.pt"
    )
    with pytest.raises(FileNotFoundError, match="--airport-nn-model"):
        reader(date(2026, 3, 1))
