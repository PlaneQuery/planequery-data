from datetime import date, datetime

import polars as pl

from data_engineering.adsbx import published_flights


def _published_flights(rows: list[tuple[datetime, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "icao": [f"ABC{index:03d}" for index in range(len(rows))],
            "callsign": [None] * len(rows),
            "registration": [f"N{index}" for index in range(len(rows))],
            "takeoff_time": [takeoff for takeoff, _ in rows],
            "takeoff_airport_ident": ["KAAA"] * len(rows),
            "landing_time": [landing for _, landing in rows],
            "landing_airport_ident": ["KBBB"] * len(rows),
        },
        schema_overrides={
            "callsign": pl.String,
            "takeoff_time": pl.Datetime("ms"),
            "landing_time": pl.Datetime("ms"),
        },
    )


def test_same_day_flights_use_only_current_arrival_partition(monkeypatch):
    target_date = date(2026, 3, 1)
    arrivals = _published_flights(
        [
            (datetime(2026, 2, 28, 23), datetime(2026, 3, 1, 1)),
            (datetime(2026, 3, 1, 10), datetime(2026, 3, 1, 12)),
        ]
    )
    requested_dates = []

    def read_arrivals(arrival_date):
        requested_dates.append(arrival_date)
        return arrivals

    monkeypatch.setattr(
        published_flights,
        "read_adsbx_published_arrivals",
        read_arrivals,
    )

    result = published_flights.read_adsbx_published_same_day_flights(
        target_date
    )

    assert requested_dates == [target_date]
    assert result.get_column("takeoff_time").to_list() == [
        datetime(2026, 3, 1, 10),
    ]


def test_legacy_arrivals_artifact_remains_readable(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    monkeypatch.setattr(published_flights, "OUTPUT_DIR", tmp_path)
    legacy_path = published_flights.legacy_adsbx_published_arrivals_path(
        target_date
    )
    legacy_path.parent.mkdir(parents=True)
    legacy = _published_flights(
        [(datetime(2026, 3, 1, 10), datetime(2026, 3, 1, 12))]
    ).with_columns(pl.lit(True).alias("pia"))
    legacy.write_parquet(legacy_path)

    result = published_flights.read_adsbx_published_arrivals(target_date)

    assert result.columns == list(published_flights.PUBLISHED_FLIGHT_SCHEMA)
    assert result.get_column("icao").to_list() == ["abc000"]
    assert not published_flights.adsbx_published_arrivals_path(
        target_date
    ).exists()


def test_raw_arrivals_are_normalized_into_intermediate_artifact(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    monkeypatch.setattr(published_flights, "OUTPUT_DIR", tmp_path)
    raw_path = published_flights._adsbx_raw_arrivals_path(target_date)
    raw_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "hex": ["ABC123"],
            "reg": ["N123AB"],
            "orig": ["KAAA"],
            "dest": ["KBBB"],
            "depTime": ["2026-03-01 10:00:00"],
            "arrTime": ["2026-03-01 12:00:00"],
        }
    ).write_csv(raw_path)

    result = published_flights.read_adsbx_published_arrivals(target_date)

    assert result.columns == list(published_flights.PUBLISHED_FLIGHT_SCHEMA)
    assert result.select(
        "icao",
        "takeoff_time",
        "landing_time",
    ).row(0) == (
        "abc123",
        datetime(2026, 3, 1, 10),
        datetime(2026, 3, 1, 12),
    )
    assert published_flights.adsbx_published_arrivals_path(
        target_date
    ).is_file()
