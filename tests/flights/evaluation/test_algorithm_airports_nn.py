from datetime import date, datetime, timedelta

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.evaluation import evaluation
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import ADSBX_SOURCE, BTS_SOURCE, OPENSKY_SOURCE


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

    def read_flights(source, target_date, adsb_source):
        assert isinstance(source, AlgorithmNNFlightSource)
        assert target_date == day
        assert adsb_source == ADSBSource.ADSBLOL
        assert source.model_path == checkpoint
        predicted = Flight(
            "abc123", start, start + timedelta(hours=1), "KAAA", "KBBB"
        )
        unavailable = Flight(
            "def456", later, later + timedelta(hours=1), None, None
        )
        return flights_to_flights_df([predicted, unavailable])

    monkeypatch.setattr(evaluation, "get_flights", read_flights)
    parquet = tmp_path / "data.parquet"
    parquet.touch()
    monkeypatch.setattr(evaluation, "flight_path", lambda *args: parquet)

    source = AlgorithmNNFlightSource(model_path=checkpoint)
    reader = evaluation.source_flights_reader(
        source,
        adsb_src=ADSBSource.ADSBLOL,
    )
    result = reader(day)
    assert result["takeoff_airport_ident"].to_list() == ["KAAA", None]
    assert result["landing_airport_ident"].to_list() == ["KBBB", None]
    assert result.columns == segments.columns
    assert evaluation.adsb_srcs_for_test_source(
        source, [ADSBSource.ADSBLOL, ADSBSource.ADSBX]
    ) == [ADSBSource.ADSBLOL, ADSBSource.ADSBX]


def test_algorithm_nn_requires_checkpoint(tmp_path):
    reader = evaluation.source_flights_reader(
        AlgorithmNNFlightSource(model_path=tmp_path / "missing.pt")
    )
    with pytest.raises(FileNotFoundError, match="--airport-nn-model"):
        reader(date(2026, 3, 1))


def test_canonical_name_resolves_to_flight_source(monkeypatch, tmp_path):
    day = date(2026, 3, 1)
    calls = []

    def read_flights(source, target_date, adsb_source):
        calls.append((source, target_date, adsb_source))
        return flights_to_flights_df([])

    monkeypatch.setattr(evaluation, "get_flights", read_flights)
    parquet = tmp_path / "data.parquet"
    parquet.touch()
    monkeypatch.setattr(evaluation, "flight_path", lambda *args: parquet)

    evaluation.source_flights_reader(BTS_SOURCE.name)(day)

    assert calls == [(BTS_SOURCE, day, None)]
    assert evaluation.adsb_srcs_for_test_source(
        BTS_SOURCE,
        [ADSBSource.ADSBLOL, ADSBSource.ADSBX],
    ) == [evaluation.DEFAULT_ADSB_SRC_FOR_FLIGHTS]


@pytest.mark.parametrize("source", [ADSBX_SOURCE, OPENSKY_SOURCE])
def test_precomputed_source_is_evaluated_once_across_adsb_sources(source):
    assert evaluation.resolve_source(source.name) is source
    assert evaluation.adsb_srcs_for_test_source(
        source,
        [ADSBSource.ADSBLOL, ADSBSource.OPENSKY, ADSBSource.ADSBX],
    ) == [evaluation.DEFAULT_ADSB_SRC_FOR_FLIGHTS]


def test_source_flights_reader_writes_missing_parquet(monkeypatch, tmp_path):
    day = date(2026, 3, 1)
    parquet = tmp_path / "missing.parquet"
    expected = flights_to_flights_df([])
    calls = []

    monkeypatch.setattr(evaluation, "flight_path", lambda *args: parquet)

    def write(source, target_date, adsb_source):
        calls.append((source, target_date, adsb_source))
        return expected

    monkeypatch.setattr(evaluation, "write_flights", write)
    monkeypatch.setattr(
        evaluation,
        "get_flights",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("a missing parquet must be written")
        ),
    )

    result = evaluation.source_flights_reader(
        BTS_SOURCE,
        adsb_src=ADSBSource.ADSBX,
    )(day)

    assert result.equals(expected)
    assert calls == [(BTS_SOURCE, day, None)]
