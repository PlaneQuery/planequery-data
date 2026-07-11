from datetime import date, datetime
import sys

import polars as pl


def _set_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    for module_name in (
        "data_engineering.utils",
        "data_engineering.flights.flight_type",
        "flights_algorithm.main",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path, raising=False)
    airports_path = tmp_path / "data" / "interim" / "airports.csv"
    airports_path.parent.mkdir(parents=True, exist_ok=True)
    airports_path.write_text(
        "ident,type,latitude_deg,longitude_deg,elevation_ft,iata_code,gps_code,local_code,iso_country,iso_region\n"
        "KAAA,small_airport,0,0,0,AAA,KAAA,AAA,US,US-AA\n",
        encoding="utf-8",
    )


def test_flights_algorithm_output_path_uses_algorithm_source_partitions(
    tmp_path,
    monkeypatch,
):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    monkeypatch.setattr(flights_algorithm_main, "OUTPUT_DIR", tmp_path)

    assert flights_algorithm_main.flights_algorithm_output_path(
        date(2026, 3, 1),
        "adsblol",
    ) == (
        tmp_path
        / "data"
        / "flights"
        / "algorithm"
        / "v1"
        / "adsblol"
        / "year=2026"
        / "month=03"
        / "day=01"
        / "flights.parquet"
    )
    assert "adsbx" in str(
        flights_algorithm_main.flights_algorithm_output_path(
            date(2026, 3, 1),
            "adsbx",
        )
    )


def test_run_main_use_adsbx_writes_adsbx_algorithm_output(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    process_calls = []
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
            "pia": [True],
            "ladd": [False],
        }
    )

    monkeypatch.setattr(
        flights_algorithm_main,
        "OUTPUT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "get_icaos_in_adsb",
        lambda _target_date: ["abc123"],
    )

    def fake_process_icaos(
        start_dt,
        end_dt,
        icaos,
        no_airports_model=False,
        use_adsbx=False,
        target_date=None,
    ):
        process_calls.append(
            {
                "start_dt": start_dt,
                "end_dt": end_dt,
                "icaos": icaos,
                "no_airports_model": no_airports_model,
                "use_adsbx": use_adsbx,
                "target_date": target_date,
            }
        )
        return flights_df

    monkeypatch.setattr(flights_algorithm_main, "process_icaos", fake_process_icaos)

    df = flights_algorithm_main.run_main(
        target_date=target_date,
        icaos=["abc123"],
        pia_or_american_ladd_only=True,
        use_adsbx=True,
        max_workers=1,
    )

    assert df.equals(flights_df)
    assert process_calls == [
        {
            "start_dt": datetime(2026, 3, 1),
            "end_dt": datetime(2026, 3, 2),
            "icaos": ["abc123"],
            "no_airports_model": False,
            "use_adsbx": True,
            "target_date": target_date,
        }
    ]
    assert (
        tmp_path
        / "data"
        / "flights"
        / "algorithm"
        / "v1"
        / "adsbx"
        / "year=2026"
        / "month=03"
        / "day=01"
        / "flights.parquet"
    ).exists()


def test_run_main_single_worker_processes_icaos_in_serial_chunks(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    icaos = [f"abc{i:03x}" for i in range(501)]
    process_calls = []

    monkeypatch.setattr(flights_algorithm_main, "OUTPUT_DIR", tmp_path)

    def fake_process_icaos(
        start_dt,
        end_dt,
        icaos,
        no_airports_model=False,
        use_adsbx=False,
        target_date=None,
    ):
        process_calls.append(icaos)
        return pl.DataFrame(
            {
                "icao": [icaos[0]],
                "takeoff_time": [datetime(2026, 3, 1, 1)],
                "landing_time": [datetime(2026, 3, 1, 2)],
            }
        )

    monkeypatch.setattr(flights_algorithm_main, "process_icaos", fake_process_icaos)

    flights_algorithm_main.run_main(
        target_date=target_date,
        icaos=icaos,
        max_workers=1,
    )

    assert process_calls == [icaos[:500], icaos[500:]]


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


def test_cli_forwards_max_workers(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    calls = []

    def fake_run_main(**kwargs):
        calls.append(kwargs)
        return pl.DataFrame({"icao": ["abc123"]})

    monkeypatch.setattr(flights_algorithm_main, "run_main", fake_run_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flights_algorithm.main",
            "2026-03-01",
            "2026-03-02",
            "--max-workers",
            "3",
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
            "max_workers": 3,
            "icaos": None,
            "no_airports_model": True,
        }
    ]


def test_run_main_no_airports_model_skips_inference_and_writes_base_output(
    tmp_path,
    monkeypatch,
):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    process_calls = []
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
        }
    )

    monkeypatch.setattr(
        flights_algorithm_main,
        "OUTPUT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        flights_algorithm_main,
        "get_icaos_in_adsb",
        lambda _target_date: ["abc123"],
    )

    def fake_process_icaos(
        start_dt,
        end_dt,
        icaos,
        no_airports_model=False,
        use_adsbx=False,
        target_date=None,
    ):
        process_calls.append(no_airports_model)
        return flights_df

    monkeypatch.setattr(flights_algorithm_main, "process_icaos", fake_process_icaos)

    df = flights_algorithm_main.run_main(
        target_date=target_date,
        no_airports_model=True,
        max_workers=1,
    )

    assert df.equals(flights_df)
    assert process_calls == [True]
    assert (
        tmp_path
        / "data"
        / "flights"
        / "algorithm-no-airports"
        / "v1"
        / "adsblol"
        / "year=2026"
        / "month=03"
        / "day=01"
        / "flights.parquet"
    ).exists()


def test_run_main_no_airports_model_rejects_adsbx():
    import pytest
    import flights_algorithm.main as flights_algorithm_main

    with pytest.raises(ValueError, match="ADSB.lol"):
        flights_algorithm_main.run_main(use_adsbx=True, no_airports_model=True)


def test_run_main_sfdps_or_bts_only_limits_process_icaos(tmp_path, monkeypatch):
    _set_output_dir(tmp_path, monkeypatch)

    import flights_algorithm.main as flights_algorithm_main

    target_date = date(2026, 3, 1)
    process_calls = []
    flights_df = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
        }
    )

    monkeypatch.setattr(flights_algorithm_main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        flights_algorithm_main,
        "_limit_icaos_to_sfdps_or_bts",
        lambda _target_date, icaos: [icao for icao in icaos if icao == "abc123"],
    )

    def fake_process_icaos(
        start_dt,
        end_dt,
        icaos,
        no_airports_model=False,
        use_adsbx=False,
        target_date=None,
    ):
        process_calls.append(icaos)
        return flights_df

    monkeypatch.setattr(flights_algorithm_main, "process_icaos", fake_process_icaos)

    flights_algorithm_main.run_main(
        target_date=target_date,
        icaos=["abc123", "def456"],
        sfdps_or_bts_only=True,
        use_adsbx=True,
        max_workers=1,
    )

    assert process_calls == [["abc123"]]
