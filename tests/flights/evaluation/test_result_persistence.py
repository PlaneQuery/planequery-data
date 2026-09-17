import json
from datetime import date, datetime, timedelta

import polars as pl

from flights.evaluation import evaluation


def test_evaluation_saves_full_comparisons_with_source_index(monkeypatch, tmp_path):
    flight = {
        "icao": "abc123",
        "takeoff_time": datetime.fromisoformat("2026-03-01T12:00:00"),
        "landing_time": datetime.fromisoformat("2026-03-01T12:00:00") + timedelta(hours=1),
        "takeoff_airport_ident": "KJFK",
        "landing_airport_ident": "KBOS",
    }
    source_frames = {
        "test_a": pl.DataFrame([flight]),
        "test_b": pl.DataFrame([{**flight, "landing_airport_ident": "KBDL"}]),
        "gold": pl.DataFrame([flight]),
    }
    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda src, adsb_src="adsblol": lambda _: source_frames[src].clone(),
    )
    monkeypatch.setattr(
        evaluation,
        "default_stats_path",
        lambda: tmp_path / "run.csv",
    )

    results, stats = evaluation.main(
        test_src=["test_a", "test_b"],
        gold_src="gold",
        test_dates=date(2026, 3, 1),
        use_all_icaos=True,
    )

    output_dir = tmp_path / "run"
    index = json.loads((output_dir / "index.json").read_text())
    assert index["stats_csv"] == "../run.csv"
    assert [entry["test_src"] for entry in index["comparisons"]] == [
        "test_a", "test_b"
    ]
    for entry, result in zip(index["comparisons"], results, strict=True):
        assert entry["test_date"] == "2026-03-01"
        assert entry["gold_src"] == "gold"
        assert entry["rows"] == result.height
        assert {"match_status", "takeoff_airport_ident", "takeoff_airport_ident_df1"} <= set(result.columns)
        assert pl.read_parquet(output_dir / entry["path"]).equals(result)
    assert pl.read_csv(tmp_path / "run.csv").height == stats.height


def test_evaluation_without_outputs_does_not_write_files(monkeypatch, tmp_path):
    flight = pl.DataFrame([{
        "icao": "abc123",
        "takeoff_time": datetime.fromisoformat("2026-03-01T12:00:00"),
        "landing_time": datetime.fromisoformat("2026-03-01T13:00:00"),
        "takeoff_airport_ident": "KJFK",
        "landing_airport_ident": "KBOS",
    }])
    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda src, adsb_src="adsblol": lambda _: flight.clone(),
    )
    monkeypatch.setattr(evaluation, "default_stats_path", lambda: tmp_path / "run.csv")

    results, _ = evaluation.main(
        test_src="test",
        gold_src="gold",
        test_dates=date(2026, 3, 1),
        use_all_icaos=True,
        write_outputs=False,
    )

    assert len(results) == 1
    assert not list(tmp_path.iterdir())
