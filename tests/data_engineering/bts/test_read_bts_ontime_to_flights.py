import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

import polars as pl
import pytest

from flights.providers import bts as bts_module
from flights.providers.bts import (
    _bts_month_paths,
    _compute_bts_timestamps,
    _create_flights_from_bts_month,
    create_flights_from_bts,
    iata_airport_mapping,
)


@dataclass(frozen=True)
class FakeAirportCatalog:
    name: str = "test-catalog"


def test_bts_month_paths_include_both_adjacent_months(monkeypatch, tmp_path):
    monkeypatch.setattr(bts_module, "OUTPUT_DIR", tmp_path)

    paths = _bts_month_paths(2026, 1)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "data/raw/bts/v2/year=2025/month=12/T_ONTIME_REPORTING.csv",
        "data/raw/bts/v2/year=2026/month=01/T_ONTIME_REPORTING.csv",
        "data/raw/bts/v2/year=2026/month=02/T_ONTIME_REPORTING.csv",
    ]


def test_iata_airport_mapping_omits_unknown_non_null_code():
    class UnknownAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            assert code_priority == ("iata_code",)
            return None

    result = iata_airport_mapping(UnknownAirportCatalog(), ["ZZZ", "ZZZ"])

    assert result.is_empty()
    assert result.schema == {
        "iata": pl.String,
        "airport_id": pl.Int64,
        "ident": pl.String,
    }


def test_iata_airport_mapping_omits_ambiguous_code():
    class AmbiguousAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            raise ValueError(f"ambiguous airport code: {code}")

    result = iata_airport_mapping(AmbiguousAirportCatalog(), ["AAA"])

    assert result.is_empty()


def test_diversion_report_recognizes_positive_counts_without_dropping_zero_or_null():
    result = pl.DataFrame(
        {
            "diverted": [0, 0, 0, 1],
            "div_airport_landings": [None, 0, 9, 1],
        }
    ).select(bts_module._diversion_report_expr().alias("is_diversion"))

    assert result.get_column("is_diversion").to_list() == [
        False,
        False,
        True,
        True,
    ]


def test_bts_callsign_uses_current_faa_carrier_designators():
    result = pl.DataFrame(
        {
            "op_unique_carrier": ["AA", "AS", "MQ", "OH", "YX", "XX", None],
            "op_carrier_fl_num": [123, 825, 450, 12, 3456, 9, 10],
        }
    ).select(bts_module._bts_callsign_expr())

    assert result.get_column("callsign").to_list() == [
        "AAL123",
        "ASA825",
        "ENY450",
        "JIA12",
        "RPA3456",
        None,
        None,
    ]


def test_compute_bts_timestamps_does_not_roll_one_minute_early_departure_to_next_day():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 2, 28),
            "origin_timezone": "America/Boise",
            "wheels_off": 524,
            "crs_dep_time": 525,
            "air_time": 33,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 2, 28, 12, 24),
        "time_landing": datetime(2026, 2, 28, 12, 57),
    }


def test_compute_bts_timestamps_rolls_true_midnight_wrap_to_next_day():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 2, 28),
            "origin_timezone": "America/Boise",
            "wheels_off": 5,
            "crs_dep_time": 2355,
            "air_time": 30,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 1, 7, 5),
        "time_landing": datetime(2026, 3, 1, 7, 35),
    }


def test_compute_bts_timestamps_uses_arrival_delay_for_delayed_next_day_flight():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 3, 1),
            "origin_timezone": "America/Chicago",
            "dest_timezone": "America/Chicago",
            "wheels_off": 1319,
            "wheels_on": 1406,
            "crs_dep_time": 2234,
            "crs_arr_time": 2348,
            "arr_delay": 864,
            "air_time": 47,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 2, 19, 19),
        "time_landing": datetime(2026, 3, 2, 20, 6),
    }


def test_compute_bts_timestamps_uses_arrival_delay_for_same_day_flight():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 3, 1),
            "origin_timezone": "America/New_York",
            "dest_timezone": "America/New_York",
            "wheels_off": 600,
            "wheels_on": 703,
            "crs_dep_time": 549,
            "crs_arr_time": 703,
            "arr_delay": 0,
            "air_time": 63,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 1, 11, 0),
        "time_landing": datetime(2026, 3, 1, 12, 3),
    }


@pytest.mark.parametrize(
    "missing_fields",
    [
        (),
        ("AIR_TIME",),
        ("WHEELS_ON",),
        ("ARR_DELAY",),
        ("AIR_TIME", "WHEELS_ON", "ARR_DELAY"),
    ],
)
def test_create_flights_from_bts_month_keeps_short_flights_and_drops_incomplete_rows(
    monkeypatch, tmp_path, missing_fields,
):
    source_path = tmp_path / "T_ONTIME_REPORTING.csv"
    source_df = pl.DataFrame(
        {
            "FL_DATE": ["03/01/2026 12:00:00 AM"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "OP_CARRIER_FL_NUM": [123],
            "ORIGIN": ["AAA"],
            "DEST": ["BBB"],
            "WHEELS_OFF": [600],
            "WHEELS_ON": [620],
            "CANCELLED": [0],
            "AIR_TIME": [20],
            "DIVERTED": [0],
            "CRS_DEP_TIME": [600],
            "CRS_ARR_TIME": [620],
            "ARR_DELAY": [0],
            "TAIL_NUM": ["N12345"],
        }
    )
    if missing_fields:
        incomplete_df = source_df.with_columns(
            pl.lit(456, dtype=pl.Int64).alias("OP_CARRIER_FL_NUM"),
            *[
                pl.lit(None, dtype=source_df.schema[field]).alias(field)
                for field in missing_fields
            ],
        )
        source_df = pl.concat([source_df, incomplete_df])
    source_df.write_csv(source_path)

    monkeypatch.setattr(
        bts_module,
        "_bts_month_paths",
        lambda _year, _month: [source_path],
    )
    monkeypatch.setattr(
        bts_module,
        "get_registration_to_icao",
        lambda at_date=None: pl.DataFrame(
            {"registration": ["N12345"], "icao": ["ABC123"]}
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "iata_airport_mapping",
        lambda _catalog, _codes: pl.DataFrame(
            {
                "iata": ["AAA", "BBB"],
                "airport_id": [1, 2],
                "ident": ["KAAA", "KBBB"],
            }
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "airport_timezone_mapping",
        lambda _catalog: pl.DataFrame(
            {"airport_id": [1, 2], "timezone": ["UTC", "UTC"]}
        ),
    )

    result = _create_flights_from_bts_month(
        2026,
        3,
        FakeAirportCatalog(),
    )

    assert result.height == 1
    assert result.select(
        "icao",
        "callsign",
        "takeoff_time",
        "landing_time",
    ).row(0) == (
        "abc123",
        "AAL123",
        datetime(2026, 3, 1, 6),
        datetime(2026, 3, 1, 6, 20),
    )


def test_diverted_flight_rows_create_each_leg_with_departing_tail():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "DDD",
            "tail_num": "N111AA",
            "crs_dep_time": 2100,
            "dep_delay": 0,
            "wheels_off": 2110,
            "wheels_on": 130,
            "div_reached_dest": 1.0,
            "div1_airport": "BBB",
            "div1_wheels_on": 2200,
            "div1_wheels_off": 2300,
            "div1_tail_num": "N222BB",
            "div2_airport": "CCC",
            "div2_wheels_on": "2350",
            "div2_wheels_off": "0030",
            "div2_tail_num": "N333CC",
        },
        {code: "UTC" for code in ("AAA", "BBB", "CCC", "DDD")},
    )

    assert rows == [
        {
            "registration": "N111AA",
            "origin": "AAA",
            "dest": "BBB",
            "takeoff_time": datetime(2026, 3, 1, 21, 10),
            "landing_time": datetime(2026, 3, 1, 22),
        },
        {
            "registration": "N222BB",
            "origin": "BBB",
            "dest": "CCC",
            "takeoff_time": datetime(2026, 3, 1, 23),
            "landing_time": datetime(2026, 3, 1, 23, 50),
        },
        {
            "registration": "N333CC",
            "origin": "CCC",
            "dest": "DDD",
            "takeoff_time": datetime(2026, 3, 2, 0, 30),
            "landing_time": datetime(2026, 3, 2, 1, 30),
        },
    ]


def test_diverted_flight_rows_do_not_invent_unreached_destination_leg():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "CCC",
            "tail_num": "N111AA",
            "crs_dep_time": 1700,
            "dep_delay": 0,
            "wheels_off": 1710,
            "wheels_on": None,
            "div_reached_dest": 0.0,
            "div1_airport": "BBB",
            "div1_wheels_on": 1800,
            "div1_wheels_off": None,
            "div1_tail_num": None,
        },
        {code: "UTC" for code in ("AAA", "BBB", "CCC")},
    )

    assert [(row["origin"], row["dest"]) for row in rows] == [
        ("AAA", "BBB")
    ]


def test_diverted_flight_rows_anchor_long_delay_to_actual_departure_day():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "CCC",
            "tail_num": "N111AA",
            "crs_dep_time": 2200,
            "dep_delay": 900,
            "wheels_off": 1315,
            "wheels_on": None,
            "div_reached_dest": 0.0,
            "div1_airport": "BBB",
            "div1_wheels_on": 1400,
            "div1_wheels_off": None,
            "div1_tail_num": None,
        },
        {code: "UTC" for code in ("AAA", "BBB", "CCC")},
    )

    assert rows[0]["takeoff_time"] == datetime(2026, 3, 2, 13, 15)
    assert rows[0]["landing_time"] == datetime(2026, 3, 2, 14)


def test_diverted_flight_rows_reject_inconsistent_anchored_final_leg():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 11),
            "origin": "CVG",
            "dest": "RSW",
            "tail_num": "N111AA",
            "crs_dep_time": 530,
            "dep_delay": -5,
            "wheels_off": 536,
            "wheels_on": 1100,
            "diverted": 1,
            "cancelled": 0,
            "div_airport_landings": 1,
            "div_reached_dest": 1.0,
            "div_actual_elapsed_time": 342,
            "div1_airport": "MIA",
            "div1_wheels_on": 928,
            "div1_wheels_off": 1100,
            "div1_tail_num": "N111AA",
        },
        {
            "CVG": "America/New_York",
            "MIA": "America/New_York",
            "RSW": "America/New_York",
        },
    )

    assert rows == []


def test_diverted_flight_rows_salvage_post_return_departure():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 2),
            "origin": "SEA",
            "dest": "MSP",
            "tail_num": "N111AA",
            "crs_dep_time": 1530,
            "dep_delay": 179,
            "wheels_off": 1853,
            "wheels_on": 2340,
            "diverted": 1,
            "cancelled": 0,
            "div_airport_landings": 1,
            "div_reached_dest": 1.0,
            "div_actual_elapsed_time": 198,
            "div1_airport": "SEA",
            "div1_wheels_on": 1705,
            "div1_wheels_off": 1853,
            "div1_tail_num": "N222BB",
        },
        {
            "SEA": "America/Los_Angeles",
            "MSP": "America/Chicago",
        },
    )

    assert rows == [
        {
            "registration": "N222BB",
            "origin": "SEA",
            "dest": "MSP",
            "takeoff_time": datetime(2026, 3, 3, 2, 53),
            "landing_time": datetime(2026, 3, 3, 5, 40),
        }
    ]


def test_diverted_flight_rows_assign_extra_day_to_ground_stop():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "CCC",
            "tail_num": "N111AA",
            "crs_dep_time": 850,
            "dep_delay": 0,
            "wheels_off": 900,
            "wheels_on": 1200,
            "diverted": 1,
            "cancelled": 0,
            "div_airport_landings": 1,
            "div_reached_dest": 1.0,
            "div_actual_elapsed_time": 1630,
            "div1_airport": "BBB",
            "div1_wheels_on": 1000,
            "div1_wheels_off": 1100,
            "div1_tail_num": "N222BB",
        },
        {code: "UTC" for code in ("AAA", "BBB", "CCC")},
    )

    assert rows[1]["takeoff_time"] == datetime(2026, 3, 2, 11)
    assert rows[1]["landing_time"] == datetime(2026, 3, 2, 12)


def test_diverted_flight_rows_create_completed_fly_return_legs():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "BBB",
            "tail_num": "N111AA",
            "crs_dep_time": 850,
            "dep_delay": 0,
            "actual_elapsed_time": 200,
            "wheels_off": 900,
            "wheels_on": 1210,
            "diverted": 0,
            "cancelled": 0,
            "div_airport_landings": 9,
            "div_reached_dest": None,
            "div1_airport": "AAA",
            "div1_wheels_on": 930,
            "div1_wheels_off": 1030,
            "div1_tail_num": "N222BB",
        },
        {"AAA": "UTC", "BBB": "UTC"},
    )

    assert [
        (row["registration"], row["origin"], row["dest"])
        for row in rows
    ] == [
        ("N111AA", "AAA", "AAA"),
        ("N222BB", "AAA", "BBB"),
    ]


def test_diverted_flight_rows_keep_airborne_cancelled_fly_return():
    rows = bts_module._diverted_flight_rows(
        {
            "fl_date": date(2026, 3, 1),
            "origin": "AAA",
            "dest": "BBB",
            "tail_num": "N111AA",
            "crs_dep_time": 850,
            "dep_delay": 0,
            "wheels_off": 900,
            "wheels_on": None,
            "diverted": 0,
            "cancelled": 1,
            "div_airport_landings": 9,
            "div_reached_dest": None,
            "div1_airport": "AAA",
            "div1_wheels_on": 930,
            "div1_wheels_off": None,
            "div1_tail_num": None,
        },
        {"AAA": "UTC", "BBB": "UTC"},
    )

    assert len(rows) == 1
    assert (rows[0]["origin"], rows[0]["dest"]) == ("AAA", "AAA")


def test_create_flights_from_bts_month_includes_diversion_tail_swap(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "T_ONTIME_REPORTING.csv"
    pl.DataFrame(
        {
            "FL_DATE": ["03/01/2026 12:00:00 AM"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "OP_CARRIER_FL_NUM": [123],
            "ORIGIN": ["AAA"],
            "DEST": ["CCC"],
            "WHEELS_OFF": [2310],
            "WHEELS_ON": [130],
            "CANCELLED": [0],
            "AIR_TIME": [None],
            "DIVERTED": [1],
            "CRS_DEP_TIME": [2300],
            "CRS_ARR_TIME": [100],
            "ARR_DELAY": [None],
            "TAIL_NUM": ["N111AA"],
            "DEP_DELAY": [0],
            "DIV_AIRPORT_LANDINGS": [1],
            "DIV_REACHED_DEST": [1],
            "DIV1_AIRPORT": ["BBB"],
            "DIV1_WHEELS_ON": [2350],
            "DIV1_WHEELS_OFF": [10],
            "DIV1_TAIL_NUM": ["N222BB"],
        }
    ).write_csv(source_path)

    monkeypatch.setattr(
        bts_module,
        "_bts_month_paths",
        lambda _year, _month: [source_path],
    )
    monkeypatch.setattr(
        bts_module,
        "get_registration_to_icao",
        lambda at_date=None: pl.DataFrame(
            {
                "registration": ["N111AA", "N222BB"],
                "icao": ["ABC111", "ABC222"],
            }
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "iata_airport_mapping",
        lambda _catalog, _codes: pl.DataFrame(
            {
                "iata": ["AAA", "BBB", "CCC"],
                "airport_id": [1, 2, 3],
                "ident": ["KAAA", "KBBB", "KCCC"],
            }
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "airport_timezone_mapping",
        lambda _catalog: pl.DataFrame(
            {
                "airport_id": [1, 2, 3],
                "timezone": ["UTC", "UTC", "UTC"],
            }
        ),
    )

    result = _create_flights_from_bts_month(
        2026,
        3,
        FakeAirportCatalog(),
    ).sort("takeoff_time")

    assert result.select(
        "icao",
        "callsign",
        "registration",
        "takeoff_airport_ident",
        "landing_airport_ident",
        "takeoff_time",
        "landing_time",
    ).rows() == [
        (
            "abc111",
            "AAL123",
            "N111AA",
            "KAAA",
            "KBBB",
            datetime(2026, 3, 1, 23, 10),
            datetime(2026, 3, 1, 23, 50),
        ),
        (
            "abc222",
            "AAL123",
            "N222BB",
            "KBBB",
            "KCCC",
            datetime(2026, 3, 2, 0, 10),
            datetime(2026, 3, 2, 1, 30),
        ),
    ]


def test_create_flights_from_bts_returns_only_target_day(monkeypatch):
    df = pl.DataFrame(
        {
            "icao": ["outside-before", "abc123", "abc123", "def456", "outside-after"],
            "takeoff_time": [
                datetime(2026, 2, 28, 23),
                datetime(2026, 3, 1, 10),
                datetime(2026, 3, 1, 10, 30),
                datetime(2026, 3, 1, 12),
                datetime(2026, 4, 1),
            ],
            "landing_time": [
                datetime(2026, 3, 1),
                datetime(2026, 3, 1, 11),
                datetime(2026, 3, 1, 11, 30),
                datetime(2026, 3, 1, 13),
                datetime(2026, 4, 1, 1),
            ],
        }
    )
    airport_catalog = FakeAirportCatalog()

    def create_month(year, month, catalog, openairframes_date=None):
        assert (year, month) == (2026, 3)
        assert catalog is airport_catalog
        assert openairframes_date == date(2026, 3, 1)
        return df

    monkeypatch.setattr(bts_module, "_create_flights_from_bts_month", create_month)
    result = create_flights_from_bts(date(2026, 3, 1), airport_catalog)

    assert result.get_column("icao").to_list() == ["def456"]
    assert bts_module.BTSFlightSource().version == 14
