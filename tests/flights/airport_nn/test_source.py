from datetime import date

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights import algorithm
from flights.airport_nn import inference, model
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.flight_source import ADSBFlightSource, flight_path


def test_algorithm_nn_is_canonical_adsb_flight_source(monkeypatch, tmp_path):
    from flights import flight_source

    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    source = AlgorithmNNFlightSource(model_path=tmp_path / "model.pt")

    assert isinstance(source, ADSBFlightSource)
    assert source.name == "algorithm-nn"
    assert "baro_rate_fpm" in source.adsb_columns()
    assert "geom_altitude_ft" in source.adsb_columns()
    assert flight_path(source, date(2026, 3, 1), ADSBSource.ADSBX) == (
        tmp_path
        / "data/flights/source=algorithm-nn/adsb_source=adsbx/version=1"
        / "year=2026/month=03/day=01/data.parquet"
    )


def test_algorithm_nn_segments_then_replaces_airports(monkeypatch, tmp_path):
    source = AlgorithmNNFlightSource(model_path=tmp_path / "model.pt")
    adsb = pl.DataFrame({"icao": ["abc123"]})
    segmented = pl.DataFrame({"icao": ["abc123"]})
    predicted = pl.DataFrame({"icao": ["abc123"], "takeoff_airport_ident": ["KJFK"]})

    monkeypatch.setattr(
        algorithm.AlgorithmNearestFlightSource,
        "create_flights_for_day",
        lambda self, dt, df: segmented,
    )
    model_loads = []
    monkeypatch.setattr(
        model,
        "load_model",
        lambda path: model_loads.append(path) or "model",
    )
    monkeypatch.setattr(
        inference,
        "predict_flights",
        lambda flights, messages, loaded_model: predicted,
    )

    assert source.create_flights_for_day(date(2026, 3, 1), adsb).equals(predicted)
    assert source.create_flights_for_day(date(2026, 3, 1), adsb).equals(predicted)
    assert model_loads == [tmp_path / "model.pt"]
