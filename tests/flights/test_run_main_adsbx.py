from datetime import date, datetime
import sys

import polars as pl


def _set_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    airports_path = tmp_path / "data" / "interim" / "airports.csv"
    airports_path.parent.mkdir(parents=True, exist_ok=True)
    airports_path.write_text(
        "ident,type,latitude_deg,longitude_deg,elevation_ft,iata_code,gps_code,local_code,iso_country,iso_region\n"
        "KAAA,small_airport,0,0,0,AAA,KAAA,AAA,US,US-AA\n",
        encoding="utf-8",
    )


def test_run_main_use_adsbx_reads_adsbx_and_writes_adsbx_output(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import data_engineering.adsb.read_adsb as read_adsb_module
    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    read_calls = []
    scan_calls = []
    inference_adsb_heights = []

    def fake_read_adsbx(current_day, columns, icaos=None, pia_or_american_ladd_only=False):
        read_calls.append(
            {
                "current_day": current_day,
                "columns": columns,
                "icaos": icaos,
                "pia_or_american_ladd_only": pia_or_american_ladd_only,
            }
        )
        return pl.DataFrame(
            {
                "time": [datetime.combine(current_day, datetime.min.time())],
                "icao": ["abc123"],
                "lat": [0.0],
                "lon": [0.0],
                "ground_speed_kt": [0.0],
                "on_ground": [True],
                "baro_altitude_ft": [0],
                "geom_altitude_ft": [0],
                "pia": [True],
                "ladd": [False],
                "military": [False],
                "interesting": [False],
                "aircraft_type": [""],
                "owner": [""],
                "aircraft_description": [""],
                "category": [""],
            }
        )

    def fake_scan_adsb(*args, **kwargs):
        scan_calls.append((args, kwargs))
        return pl.DataFrame().lazy()

    output_path = tmp_path / "flights.parquet"
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
            "pia": [True],
            "ladd": [False],
        }
    )

    monkeypatch.setattr(read_adsb_module, "read_adsbx", fake_read_adsbx)
    monkeypatch.setattr(read_adsb_module, "scan_adsb", fake_scan_adsb)
    monkeypatch.setattr(
        flights_algorithm_main,
        "_collect_segments_from_adsb_lf",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_segments_to_flights_df",
        lambda _segments: flights_df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_filter_short_same_airport_flights",
        lambda df: df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_run_airport_model_inference",
        lambda df, df_adsb_full: (
            inference_adsb_heights.append(df_adsb_full.height),
            df,
        )[1],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "adsbx_flights_airport_model_output_path",
        lambda _target_date: output_path,
    )

    df = flights_algorithm_main.run_main(
        target_date=target_date,
        icaos=["abc123"],
        pia_or_american_ladd_only=True,
        use_adsbx=True,
        max_workers=1,
    )

    assert df.equals(flights_df)
    assert output_path.exists()
    assert scan_calls == []
    assert [call["current_day"] for call in read_calls] == [
        date(2026, 2, 28),
        date(2026, 3, 1),
        date(2026, 3, 2),
    ]
    assert inference_adsb_heights == [2]
    assert all(call["icaos"] == ["abc123"] for call in read_calls)
    assert all(call["pia_or_american_ladd_only"] for call in read_calls)


def test_cli_forwards_optional_icaos(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    calls = []

    def fake_run_main(**kwargs):
        calls.append(kwargs)
        return pl.DataFrame({"icao": kwargs["icaos"]})

    monkeypatch.setattr(flights_algorithm_main, "run_main", fake_run_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flights_algorithm.main",
            "2026-03-01",
            "2026-03-03",
            "--icaos",
            "ABC123",
            "def456",
            "--max-workers",
            "1",
            "--no-airports-model",
        ],
    )

    flights_algorithm_main.cli()

    assert calls == [
        {
            "target_date": date(2026, 3, 1),
            "pia_or_american_ladd_only": False,
            "sfdps_or_bts_only": False,
            "use_adsbx": False,
            "max_workers": 1,
            "icaos": ["abc123", "def456"],
            "no_airports_model": True,
        },
        {
            "target_date": date(2026, 3, 2),
            "pia_or_american_ladd_only": False,
            "sfdps_or_bts_only": False,
            "use_adsbx": False,
            "max_workers": 1,
            "icaos": ["abc123", "def456"],
            "no_airports_model": True,
        },
    ]


def test_run_main_no_airports_model_skips_inference_and_writes_base_output(
    tmp_path,
    monkeypatch,
):
    _set_output_dir(tmp_path, monkeypatch)

    import data_engineering.adsb.read_adsb as read_adsb_module
    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    inference_calls = []

    def fake_scan_adsb(current_day, columns, icaos=None, pia_or_american_ladd_only=False):
        return pl.DataFrame(
            {
                "time": [datetime.combine(current_day, datetime.min.time())],
                "icao": ["abc123"],
                "lat": [0.0],
                "lon": [0.0],
                "ground_speed_kt": [0.0],
                "on_ground": [True],
                "baro_altitude_ft": [0],
                "geom_altitude_ft": [0],
                "pia": [False],
                "ladd": [False],
                "military": [False],
                "interesting": [False],
                "aircraft_type": [""],
                "owner": [""],
                "aircraft_description": [""],
                "category": [""],
            }
        ).lazy()

    output_path = tmp_path / "no_airports_model" / "flights.parquet"
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
        }
    )

    monkeypatch.setattr(read_adsb_module, "scan_adsb", fake_scan_adsb)
    monkeypatch.setattr(
        flights_algorithm_main,
        "_collect_segments_from_adsb_lf",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_segments_to_flights_df",
        lambda _segments: flights_df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_filter_short_same_airport_flights",
        lambda df: df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_run_airport_model_inference",
        lambda df, df_adsb_full: inference_calls.append((df, df_adsb_full)),
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "adsblol_flights_output_path",
        lambda _target_date: output_path,
    )

    df = flights_algorithm_main.run_main(
        target_date=target_date,
        no_airports_model=True,
        max_workers=1,
    )

    assert df.equals(flights_df)
    assert inference_calls == []
    assert output_path.exists()


def test_run_main_no_airports_model_rejects_adsbx():
    import pytest
    import flights_algorithm.main as flights_algorithm_main

    with pytest.raises(ValueError, match="ADSB.lol"):
        flights_algorithm_main.run_main(use_adsbx=True, no_airports_model=True)


def test_run_main_sfdps_or_bts_only_limits_adsb_reads(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import data_engineering.adsb.read_adsb as read_adsb_module
    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    read_calls = []
    inference_adsb_heights = []

    def fake_read_adsbx(current_day, columns, icaos=None, pia_or_american_ladd_only=False):
        read_calls.append(
            {
                "current_day": current_day,
                "columns": columns,
                "icaos": icaos,
                "pia_or_american_ladd_only": pia_or_american_ladd_only,
            }
        )
        return pl.DataFrame(
            {
                "time": [datetime.combine(current_day, datetime.min.time())],
                "icao": ["abc123"],
                "lat": [0.0],
                "lon": [0.0],
                "ground_speed_kt": [0.0],
                "on_ground": [True],
                "baro_altitude_ft": [0],
                "geom_altitude_ft": [0],
                "pia": [False],
                "ladd": [False],
                "military": [False],
                "interesting": [False],
                "aircraft_type": [""],
                "owner": [""],
                "aircraft_description": [""],
                "category": [""],
            }
        )

    output_path = tmp_path / "flights.parquet"
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
        }
    )

    monkeypatch.setattr(read_adsb_module, "read_adsbx", fake_read_adsbx)
    monkeypatch.setattr(
        flights_algorithm_main,
        "_limit_icaos_to_sfdps_or_bts",
        lambda _target_date, icaos: [icao for icao in icaos if icao == "abc123"],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_collect_segments_from_adsb_lf",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_segments_to_flights_df",
        lambda _segments: flights_df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_filter_short_same_airport_flights",
        lambda df: df,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "_run_airport_model_inference",
        lambda df, df_adsb_full: (
            inference_adsb_heights.append(df_adsb_full.height),
            df,
        )[1],
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "adsbx_flights_airport_model_output_path",
        lambda _target_date: output_path,
    )

    flights_algorithm_main.run_main(
        target_date=target_date,
        icaos=["abc123", "def456"],
        sfdps_or_bts_only=True,
        use_adsbx=True,
        max_workers=1,
    )

    assert [call["current_day"] for call in read_calls] == [
        date(2026, 2, 28),
        date(2026, 3, 1),
        date(2026, 3, 2),
    ]
    assert inference_adsb_heights == [2]
    assert all(call["icaos"] == ["abc123"] for call in read_calls)
