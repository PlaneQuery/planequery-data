from dataclasses import dataclass
from datetime import date

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights import algorithm
from flights.airport_nn import inference, model
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.flight_source import ADSBFlightSource, flight_path


@dataclass(frozen=True)
class FakeAirportCatalog:
    name: str = "test-catalog"


def test_algorithm_nn_is_canonical_adsb_flight_source(monkeypatch, tmp_path):
    from flights import flight_source

    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    source = AlgorithmNNFlightSource(model_path=model_path)

    assert isinstance(source, ADSBFlightSource)
    assert source.name == "algorithm-nn"
    assert "baro_rate_fpm" in source.adsb_columns()
    assert "geom_altitude_ft" in source.adsb_columns()
    assert flight_path(source, date(2026, 3, 1), ADSBSource.ADSBX) == (
        tmp_path
        / "data/flights/source=algorithm-nn/adsb_source=adsbx"
        / f"version={source.version}"
        / "year=2026/month=03/day=01/data.parquet"
    )


def test_algorithm_nn_model_does_not_change_flight_path(monkeypatch, tmp_path):
    from flights import flight_source

    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path / "output")
    first_model = tmp_path / "first" / "model.pt"
    second_model = tmp_path / "second" / "model.pt"
    first_model.parent.mkdir()
    second_model.parent.mkdir()
    first_model.write_bytes(b"first model")
    second_model.write_bytes(b"other model")

    first_path = flight_path(
        AlgorithmNNFlightSource(model_path=first_model),
        date(2026, 3, 1),
        ADSBSource.ADSBX,
    )
    second_path = flight_path(
        AlgorithmNNFlightSource(model_path=second_model),
        date(2026, 3, 1),
        ADSBSource.ADSBX,
    )

    assert first_path == second_path


def test_algorithm_nn_segments_then_replaces_airports(monkeypatch, tmp_path):
    source = AlgorithmNNFlightSource(model_path=tmp_path / "model.pt")
    adsb = pl.DataFrame({"icao": ["abc123"]})
    segmented = pl.DataFrame({"icao": ["abc123"]})
    predicted = pl.DataFrame({"icao": ["abc123"], "takeoff_airport_ident": ["KJFK"]})
    airport_catalog = FakeAirportCatalog()

    monkeypatch.setattr(
        algorithm.AlgorithmNearestFlightSource,
        "create_flights_for_day",
        lambda self, dt, df, catalog: segmented,
    )
    model_loads = []
    monkeypatch.setattr(
        model,
        "load_model",
        lambda path: model_loads.append(path) or "model",
    )
    catalogs = []
    monkeypatch.setattr(
        inference,
        "predict_flights",
        lambda flights, messages, loaded_model, catalog: (
            catalogs.append(catalog) or predicted
        ),
    )

    assert source.create_flights_for_day(
        date(2026, 3, 1), adsb, airport_catalog
    ).equals(predicted)
    assert source.create_flights_for_day(
        date(2026, 3, 1), adsb, airport_catalog
    ).equals(predicted)
    assert model_loads == [tmp_path / "model.pt"]
    assert catalogs == [airport_catalog, airport_catalog]
