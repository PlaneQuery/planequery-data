from datetime import date, datetime, timedelta

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.airport_nn import AlgorithmNNFlightSource, OpenSkyNNFlightSource
from flights.evaluation import evaluation
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import (
    ADSBX_SOURCE,
    BTS_SOURCE,
    BTS_VERIFIED_SOURCE,
    EUROCONTROL_SOURCE,
    EUROCONTROL_VERIFIED_SOURCE,
    OPENSKY_NN_SOURCE,
    OPENSKY_NEAREST_SOURCE,
    OPENSKY_SOURCE,
    SFDPS_SOURCE,
    SFDPS_VERIFIED_SOURCE,
)
from flights.verified.source import VerifiedFlightSource


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

    source = AlgorithmNNFlightSource(model_path=checkpoint)
    reader = evaluation.source_flights_reader(
        source,
        adsb_source=ADSBSource.ADSBLOL,
    )
    result = reader(day)
    assert result["takeoff_airport_ident"].to_list() == ["KAAA", None]
    assert result["landing_airport_ident"].to_list() == ["KBBB", None]
    assert result.columns == segments.columns
    assert evaluation.adsb_sources_for_evaluation(
        source,
        BTS_SOURCE,
        [ADSBSource.ADSBLOL, ADSBSource.ADSBX],
    ) == [ADSBSource.ADSBLOL, ADSBSource.ADSBX]


@pytest.mark.parametrize("source_type", [AlgorithmNNFlightSource, OpenSkyNNFlightSource])
def test_airport_nn_requires_checkpoint(tmp_path, source_type):
    reader = evaluation.source_flights_reader(
        source_type(model_path=tmp_path / "missing.pt")
    )
    with pytest.raises(FileNotFoundError, match="--airport-nn-model"):
        reader(date(2026, 3, 1))


def test_canonical_name_resolves_to_flight_source(monkeypatch):
    day = date(2026, 3, 1)
    calls = []

    def read_flights(source, target_date, adsb_source):
        calls.append((source, target_date, adsb_source))
        return flights_to_flights_df([])

    monkeypatch.setattr(evaluation, "get_flights", read_flights)

    evaluation.source_flights_reader(BTS_SOURCE.name)(day)

    assert calls == [(BTS_SOURCE, day, None)]
    assert evaluation.adsb_sources_for_evaluation(
        BTS_SOURCE,
        BTS_SOURCE,
        [ADSBSource.ADSBLOL, ADSBSource.ADSBX],
    ) == [None]


@pytest.mark.parametrize(
    ("verified_source", "wrapped_source"),
    [
        (BTS_VERIFIED_SOURCE, BTS_SOURCE),
        (SFDPS_VERIFIED_SOURCE, SFDPS_SOURCE),
        (EUROCONTROL_VERIFIED_SOURCE, EUROCONTROL_SOURCE),
    ],
)
@pytest.mark.parametrize("target_day_adsb", [False, True])
@pytest.mark.parametrize("provider", [None, *ADSBSource])
def test_verified_source_always_uses_adsblol(
    monkeypatch,
    verified_source,
    wrapped_source,
    target_day_adsb,
    provider,
):
    day = date(2026, 3, 1)
    calls = []

    def read_flights(source, target_date, adsb_source):
        calls.append((source, target_date, adsb_source))
        return flights_to_flights_df([])

    monkeypatch.setattr(evaluation, "get_flights", read_flights)

    evaluation.source_flights_reader(
        evaluation._with_target_day_adsb([verified_source], target_day_adsb)[0],
        provider,
    )(day)

    configured_source, target_date, source_adsb = calls[0]
    assert isinstance(configured_source, VerifiedFlightSource)
    if wrapped_source is EUROCONTROL_SOURCE:
        assert configured_source.source.callsign_source is ADSBSource.ADSBLOL
    else:
        assert configured_source.source is wrapped_source
    assert configured_source.adsb_source is ADSBSource.ADSBLOL
    assert configured_source.target_day_adsb is target_day_adsb
    assert target_date == day
    assert source_adsb is None
    assert evaluation.adsb_sources_for_evaluation(
        BTS_SOURCE,
        verified_source,
        [ADSBSource.ADSBLOL, ADSBSource.OPENSKY],
    ) == [None]


@pytest.mark.parametrize("source", [ADSBX_SOURCE, OPENSKY_SOURCE, OPENSKY_NN_SOURCE, OPENSKY_NEAREST_SOURCE])
def test_independent_source_is_evaluated_once_across_adsb_sources(source):
    assert evaluation.resolve_source(source.name) is source
    assert evaluation.adsb_sources_for_evaluation(
        source,
        BTS_SOURCE,
        [ADSBSource.ADSBLOL, ADSBSource.OPENSKY, ADSBSource.ADSBX],
    ) == [None]


def test_opensky_nn_evaluation_uses_custom_checkpoint_and_fixed_provider(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    sources = evaluation._with_airport_nn_model(
        [OPENSKY_NN_SOURCE, BTS_SOURCE], checkpoint,
    )
    assert sources[1] is BTS_SOURCE
    source = sources[0]
    assert source.model_path == checkpoint
    assert source.enrichment_adsb_source() is ADSBSource.OPENSKY
    calls = []
    monkeypatch.setattr(
        evaluation, "get_flights",
        lambda *args: calls.append(args) or flights_to_flights_df([]),
    )

    evaluation.source_flights_reader(source, ADSBSource.ADSBX)(date(2026, 3, 1))

    assert calls == [(source, date(2026, 3, 1), None)]
    assert evaluation.parse_args(["--test-source", "opensky-nn"]).test_source == [
        "opensky-nn"
    ]
