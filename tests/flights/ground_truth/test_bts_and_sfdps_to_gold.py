from datetime import date, datetime

import polars as pl

import flights.ground_truth.bts_and_sfdps_to_gold as ground_truth


TARGET_DATE = date(2026, 3, 1)


def test_ground_truth_cache_reads_existing_parquet(monkeypatch, tmp_path):
    cache_path = tmp_path / "flights.parquet"
    expected = pl.DataFrame({"icao": ["abc123"]})
    expected.write_parquet(cache_path)

    monkeypatch.setattr(
        ground_truth,
        "ground_truth_flights_output_path",
        lambda _: cache_path,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_bts_flights_for_day",
        lambda _: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )

    result = ground_truth.get_ground_truth_flights_for_day(TARGET_DATE)

    assert result.equals(expected)


def test_ground_truth_generation_writes_daily_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "flights.parquet"
    flights = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
            "takeoff_airport_ident": ["KJFK"],
            "landing_airport_ident": ["KLAX"],
        }
    )

    monkeypatch.setattr(
        ground_truth,
        "ground_truth_flights_output_path",
        lambda _: cache_path,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_bts_flights_for_day",
        lambda _: flights.clone(),
    )
    monkeypatch.setattr(
        ground_truth,
        "get_sfdps_flights_day",
        lambda _: flights.clone(),
    )
    monkeypatch.setattr(
        ground_truth,
        "continental_usa_to_continental_usa",
        lambda df: df,
    )
    monkeypatch.setattr(
        ground_truth,
        "df_flights_comparision",
        lambda df_bts, *_args, **_kwargs: df_bts.with_columns(
            pl.lit("both").alias("match_status")
        ),
    )
    monkeypatch.setattr(
        ground_truth,
        "read_adsb",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    monkeypatch.setattr(
        ground_truth,
        "get_matching_icaos_in_flights",
        lambda df, *_args: df,
    )

    result = ground_truth.get_ground_truth_flights_for_day(
        TARGET_DATE,
        use_cache=False,
    )

    assert result.equals(flights)
    assert pl.read_parquet(cache_path).equals(flights)
