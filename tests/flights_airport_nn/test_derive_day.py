from datetime import date, datetime, timedelta

import polars as pl

from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights_airport_nn import derive_day as builder


def test_day_cache_resumes_and_invalidates_with_inputs(monkeypatch, tmp_path):
    day = date(2026, 3, 1)
    start = datetime(2026, 3, 1, 12)  # noqa: DTZ001 - source times are naive
    original = Flight("abc123", start, start + timedelta(hours=1), None, None)
    flights = flights_to_flights_df([original])
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    output = tmp_path / "algorithm-nn" / "flights.parquet"
    version = [1]
    calls = []

    monkeypatch.setattr(builder, "derived_path", lambda *_args: output)
    monkeypatch.setattr(
        builder,
        "source_signature",
        lambda *_args: {"version": version[0]},
    )
    monkeypatch.setattr(builder, "load_model", lambda *_args: ("model", "config"))

    def score(*_args, **_kwargs):
        calls.append(version[0])
        predicted = Flight(
            "abc123", start, start + timedelta(hours=1), f"K{version[0]}", None
        )
        return {predicted.flight_id: predicted}

    monkeypatch.setattr(builder, "run_inference", score)
    first = builder.derive_day(day, model_path=model_path, flights=flights)
    assert first == output
    assert pl.read_parquet(first)["takeoff_airport_ident"].to_list() == ["K1"]
    builder.derive_day(day, model_path=model_path, flights=flights)
    assert calls == [1]
    version[0] = 2
    builder.derive_day(day, model_path=model_path, flights=flights)
    assert calls == [1, 2]
    assert pl.read_parquet(output)["takeoff_airport_ident"].to_list() == ["K2"]
