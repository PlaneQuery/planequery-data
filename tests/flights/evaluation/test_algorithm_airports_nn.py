from datetime import date, datetime, timedelta

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation import evaluation
from flights.airport_nn import derive_day
from flights.flight_type import Flight, flights_to_flights_df


def test_algorithm_nn_source_uses_checkpoint(
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
    def derive(target_date, source, model_path):
        assert target_date == day
        assert source == ADSBSource.ADSBLOL
        assert model_path == checkpoint
        predicted = Flight(
            "abc123", start, start + timedelta(hours=1), "KAAA", "KBBB"
        )
        unavailable = Flight(
            "def456", later, later + timedelta(hours=1), None, None
        )
        path = tmp_path / "flights.parquet"
        flights_to_flights_df([predicted, unavailable]).write_parquet(path)
        return path

    monkeypatch.setattr(derive_day, "derive_day", derive)

    reader = evaluation.source_flights_reader(
        "algorithm-nn",
        adsb_src=ADSBSource.ADSBLOL,
        airport_nn_model_path=checkpoint,
    )
    result = reader(day)
    assert result["takeoff_airport_ident"].to_list() == ["KAAA", None]
    assert result["landing_airport_ident"].to_list() == ["KBBB", None]
    assert result.columns == segments.columns
    assert evaluation.adsb_srcs_for_test_source(
        "algorithm-nn", [ADSBSource.ADSBLOL, ADSBSource.ADSBX]
    ) == [ADSBSource.ADSBLOL, ADSBSource.ADSBX]


def test_algorithm_nn_requires_checkpoint(tmp_path):
    reader = evaluation.source_flights_reader(
        "algorithm-nn", airport_nn_model_path=tmp_path / "missing.pt"
    )
    with pytest.raises(FileNotFoundError, match="--airport-nn-model"):
        reader(date(2026, 3, 1))
