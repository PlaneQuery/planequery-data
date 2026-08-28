from datetime import date

import polars as pl

import pia.ground_truth as ground_truth
import pia.identify_ladd_planes.pia_broadcast_adsb as pia_broadcast_adsb
import pia.identify_ladd_planes.pia_callsign_switch as pia_callsign_switch
from pia.create_website_csv import create_pia_mappings_dataframe


def test_cached_pia_broadcast_csv_is_enriched_with_faa_data(
    tmp_path, monkeypatch, capsys
):
    cache_path = tmp_path / "pia_broadcast_adsb.csv"
    pl.DataFrame(
        {
            "pia_icao": [" PIA001 ", "pia002"],
            "registration": [" n100 ", "N200"],
        }
    ).write_csv(cache_path)
    faa = pl.DataFrame(
        {
            "registration_number": ["N100"],
            "transponder_code_hex": ["A0B001"],
        }
    )
    monkeypatch.setattr(pia_broadcast_adsb, "PIA_BROADCAST_CACHE_PATH", cache_path)
    monkeypatch.setattr(pia_broadcast_adsb, "get_openairframes_faa_df", lambda: faa)

    result = pia_broadcast_adsb.get_pia_to_reg_broadcast_icaos(use_cache=True)

    assert result.to_dicts() == [
        {
            "pia_icao": "pia001",
            "registration": "N100",
            "real_icao": "a0b001",
        }
    ]
    assert "N200" in capsys.readouterr().out


def test_cached_callsign_switch_csv_is_enriched_with_openairframes(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "real_and_pia_icao_equal_callsign_close.csv"
    pl.DataFrame(
        {
            "pia_icao": ["pia001", "pia001"],
            "ladd_icao": ["A0B001", "A0B002"],
            "callsign": ["TEST1", "TEST2"],
            "pia_time": ["2025-01-01", "2025-01-02"],
            "ladd_time": ["2025-01-01", "2025-01-02"],
            "time_difference": [1, 2],
        }
    ).write_csv(cache_path)
    openairframes = pl.DataFrame(
        {
            "icao": ["a0b001", "a0b002"],
            "registration": ["n100", "n200"],
        }
    ).lazy()
    monkeypatch.setattr(
        pia_callsign_switch,
        "PIA_CALLSIGN_SWITCH_CACHE_PATH",
        cache_path,
    )
    monkeypatch.setattr(
        pia_callsign_switch,
        "scan_latest_icao_info_for_join",
        lambda: openairframes,
    )

    result = pia_callsign_switch.get_pia_callsign_switch_icaos(use_cache=True)

    assert sorted(result.to_dicts(), key=lambda row: row["real_icao"]) == [
        {"pia_icao": "pia001", "registration": "N100", "real_icao": "a0b001"},
        {"pia_icao": "pia001", "registration": "N200", "real_icao": "a0b002"},
    ]


def test_cached_callsign_switch_excludes_unmatched_openairframes_rows(
    tmp_path, monkeypatch, capsys
):
    cache_path = tmp_path / "real_and_pia_icao_equal_callsign_close.csv"
    pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "ladd_icao": ["a0b001"],
            "callsign": ["TEST1"],
            "pia_time": ["2025-01-01"],
            "ladd_time": ["2025-01-01"],
            "time_difference": [1],
        }
    ).write_csv(cache_path)
    monkeypatch.setattr(
        pia_callsign_switch,
        "PIA_CALLSIGN_SWITCH_CACHE_PATH",
        cache_path,
    )
    monkeypatch.setattr(
        pia_callsign_switch,
        "scan_latest_icao_info_for_join",
        lambda: pl.DataFrame(
            {"icao": [], "registration": []},
            schema={"icao": pl.String, "registration": pl.String},
        ).lazy(),
    )

    result = pia_callsign_switch.get_pia_callsign_switch_icaos(use_cache=True)

    assert result.is_empty()
    assert "a0b001" in capsys.readouterr().out


def test_cached_observed_pia_csv_avoids_parquet_scan(tmp_path, monkeypatch):
    cache_path = tmp_path / "all_pias_observed.csv"
    pl.DataFrame({"icao": [" A0B002 ", "a0b001", "a0b001"]}).write_csv(
        cache_path
    )
    monkeypatch.setattr(ground_truth, "ALL_PIA_ICAOS_CACHE_PATH", cache_path)
    monkeypatch.setattr(ground_truth, "assert_pia_icaos_not_in_faa", lambda _: None)

    result = ground_truth.get_unique_pia_icaos_from_adsblol(use_cache=True)

    assert result.to_list() == ["a0b001", "a0b002"]


def test_ground_truth_propagates_use_cache_to_mapping_sources(
    monkeypatch, capsys
):
    calls = []
    ground_control = pl.DataFrame(
        {
            "pia_icao": ["pia000"],
            "registration": ["N000"],
            "real_icao": ["a00000"],
            "date": [date(2025, 1, 1)],
        }
    )
    source_mapping = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N100"],
            "real_icao": ["a0b001"],
        }
    )
    monkeypatch.setattr(
        ground_truth,
        "load_ground_control_mappings",
        lambda: ground_control,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_to_reg_broadcast_icaos",
        lambda use_cache: calls.append(("broadcast", use_cache)) or source_mapping,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_callsign_switch_icaos",
        lambda use_cache: calls.append(("callsign", use_cache)) or source_mapping,
    )
    monkeypatch.setattr(
        ground_truth,
        "load_graph_solve_mappings",
        lambda: source_mapping.clear(),
    )

    ground_truth.get_ground_truth_mappings(use_cache=True)

    assert calls == [("broadcast", True), ("callsign", True)]
    assert "Mappings found only in ground control: 1" in capsys.readouterr().out


def test_ground_truth_reports_cross_source_disagreements(monkeypatch, capsys):
    ground_control = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N100"],
            "real_icao": ["a0b001"],
            "date": [date(2025, 1, 1)],
        }
    )
    broadcast = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N200"],
            "real_icao": ["a0b002"],
        }
    )
    empty = broadcast.clear()
    monkeypatch.setattr(
        ground_truth, "load_ground_control_mappings", lambda: ground_control
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_to_reg_broadcast_icaos",
        lambda use_cache: broadcast,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_callsign_switch_icaos",
        lambda use_cache: empty,
    )
    monkeypatch.setattr(
        ground_truth,
        "load_graph_solve_mappings",
        lambda: empty,
    )

    result = ground_truth.get_ground_truth_mappings(use_cache=True)

    assert result.to_dicts() == [
        {
            "pia_icao": "pia001",
            "registration": "N200",
            "real_icao": "a0b002",
            "source": "callsign",
        }
    ]
    output = capsys.readouterr().out
    assert "ground_control" in output
    assert "pia_broadcast_adsb" in output


def test_callsign_switch_has_priority_over_ground_control(monkeypatch):
    ground_control = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N100"],
            "real_icao": ["a0b001"],
            "date": [date(2025, 1, 1)],
        }
    )
    callsign_switch = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N200"],
            "real_icao": ["a0b002"],
        }
    )
    empty = callsign_switch.clear()
    monkeypatch.setattr(
        ground_truth, "load_ground_control_mappings", lambda: ground_control
    )
    monkeypatch.setattr(
        ground_truth, "load_graph_solve_mappings", lambda: empty
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_to_reg_broadcast_icaos",
        lambda use_cache: empty,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_callsign_switch_icaos",
        lambda use_cache: callsign_switch,
    )

    result = ground_truth.get_ground_truth_mappings(use_cache=True)

    assert result.to_dicts() == [
        {
            "pia_icao": "pia001",
            "registration": "N200",
            "real_icao": "a0b002",
            "source": "callsign",
        }
    ]


def test_graph_solve_has_highest_priority_and_reports_disagreements(
    monkeypatch, capsys
):
    graph_solve = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N300"],
            "real_icao": ["a0b003"],
        }
    )
    ground_control = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N100"],
            "real_icao": ["a0b001"],
            "date": [date(2025, 1, 1)],
        }
    )
    broadcast = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N200"],
            "real_icao": ["a0b002"],
        }
    )
    monkeypatch.setattr(
        ground_truth, "load_graph_solve_mappings", lambda: graph_solve
    )
    monkeypatch.setattr(
        ground_truth, "load_ground_control_mappings", lambda: ground_control
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_to_reg_broadcast_icaos",
        lambda use_cache: broadcast,
    )
    monkeypatch.setattr(
        ground_truth,
        "get_pia_callsign_switch_icaos",
        lambda use_cache: broadcast.clear(),
    )

    result = ground_truth.get_ground_truth_mappings(use_cache=True)

    assert result.to_dicts() == [
        {
            "pia_icao": "pia001",
            "registration": "N300",
            "real_icao": "a0b003",
            "source": "graph-solve",
        }
    ]
    output = capsys.readouterr().out
    assert "graph_solve" in output
    assert "ground_control" in output
    assert "pia_broadcast_adsb" in output


def test_website_csv_retains_icaos_without_a_joined_mapping():
    all_pia_icaos = pl.Series("icao", ["pia001", "pia002"])
    ground_truth_mappings = pl.DataFrame(
        {
            "pia_icao": ["pia001"],
            "registration": ["N100"],
            "real_icao": ["a0b001"],
            "source": ["callsign"],
        }
    )

    result = create_pia_mappings_dataframe(
        all_pia_icaos,
        ground_truth_mappings,
    )

    assert result.to_dicts() == [
        {
            "icao": "pia001",
            "real_icao": "a0b001",
            "registration": "N100",
            "source": "callsign",
        },
        {
            "icao": "pia002",
            "real_icao": None,
            "registration": None,
            "source": None,
        },
    ]
