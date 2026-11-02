"""Category cohorts must retain false positives and aircraft with no matches."""
from datetime import date, datetime
from pathlib import Path
import sys

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/paper"))
import category_performance_table as table


def test_reference_categories_control_unmatched_test_flights_and_missing_codes():
    gold = pl.DataFrame({"icao": ["a", "a", "b", "c", "d"],
                         "category": [" a1 ", "A1", None, "A5", "A4"]})
    lookup = table.category_lookup(gold)
    comparison = pl.DataFrame({
        "icao": ["a", "a", "b", "c", "d"],
        "category": ["A3"] * 5,  # Test labels must never determine the cohort.
        "category_df1": ["A1", None, None, "A5", "A4"],
        "match_status": ["both", "df0_only", "df1_only", "both", "df1_only"],
        "same_takeoff_airport_id": [True, False, False, False, False],
        "same_landing_airport_id": [True, False, False, True, False],
        "same_airport_id": [True, False, False, False, False],
        "takeoff_airport_id": [1, 1, None, None, None],
        "landing_airport_id": [2, 2, None, 2, None],
        "takeoff_time": [datetime(2026, 3, 1, 12)] * 5,
        "takeoff_time_df1": [datetime(2026, 3, 1, 12)] * 5,
        "landing_time": [datetime(2026, 3, 1, 13)] * 5,
        "landing_time_df1": [datetime(2026, 3, 1, 13)] * 5,
    })
    metadata = dict(test_date=date(2026, 3, 1), adsb_source=None,
                    test_source="opensky", gold_source="bts")
    rows = {r["category_group"]: r for r in table.summarize(comparison, lookup, {"a", "c"}, metadata)}
    assert rows["A1"]["true_positive"] == 1
    assert rows["A1"]["false_positive"] == 1
    assert rows["A1"]["precision"] == .5
    assert rows["Other"]["true_positive"] == 1
    assert rows["Other"]["false_negative"] == 1
    assert rows["Other"]["gold_icao_missing_from_test_count"] == 1
    assert rows["A4"]["gold_icao_missing_from_test_count"] == 1
    assert rows["A2"]["gold_icao_count"] == 0
    assert sum(r["true_positive"] + r["false_positive"] + r["false_negative"] for r in rows.values()) == 5


def test_conflicting_reference_categories_and_unknown_aircraft_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        table.category_lookup(pl.DataFrame({"icao": ["a", "a"], "category": ["A1", "A2"]}))
    lookup = table.category_lookup(pl.DataFrame({"icao": ["a"], "category": ["A1"]}))
    with pytest.raises(ValueError, match="outside"):
        table.categorize(pl.DataFrame({"icao": ["b"]}), lookup)


def test_airport_size_uses_gold_endpoint_and_matched_denominators():
    comparison = pl.DataFrame({
        "match_status": ["both", "both", "both", "df1_only", "df0_only"],
        "takeoff_airport_id_df1": [1, 2, 999, 1, None],
        "landing_airport_id_df1": [3, 1, 3, 1, None],
        "takeoff_airport_id": [3, 2, None, None, 1],
        "landing_airport_id": [3, None, 2, None, 1],
        "same_takeoff_airport_id": [False, True, False, False, False],
        "same_landing_airport_id": [True, False, False, False, False],
    })
    rows = table.summarize_airport_sizes(comparison,
        {1: "small_airport", 2: "medium_airport", 3: "large_airport"}, {})
    lookup = {(r["airport_type"], r["endpoint"]): r for r in rows}
    assert lookup["small_airport", "takeoff"]["matched_endpoint_count"] == 1
    assert lookup["small_airport", "takeoff"]["accuracy_percent"] == 0
    assert lookup["medium_airport", "takeoff"]["accuracy_percent"] == 100
    assert lookup["small_airport", "landing"]["null_prediction_count"] == 1
    assert lookup["small_airport", "landing"]["accuracy_percent"] == 0
    assert lookup["large_airport", "landing"]["matched_endpoint_count"] == 2
    assert lookup["large_airport", "landing"]["accuracy_percent"] == 50
    assert lookup["large_airport", "takeoff"]["accuracy_percent"] is None
    assert lookup["unknown", "takeoff"]["matched_endpoint_count"] == 1
    assert sum(r["matched_endpoint_count"] for r in rows) == 6


def test_latex_keeps_categories_in_separate_groups():
    metrics = tuple(m for m in table.TABLES["reference_summary"].metrics if m.key == "test_flight_count")
    rows = [dict(gold_src="bts", test_date="2026-03-01", target_day_adsb="false",
                 category_group=category, test_src="opensky", adsb_src="",
                 gold_icao_count=n, gold_flight_count=n, test_flight_count=n)
            for category, n in (("A1", 3), ("A2", 5))]
    latex = table.summary_latex(rows, metrics, "test", 2)
    assert "A1 | Gold ICAOs: 3 | Gold flights: 3" in latex
    assert "A2 | Gold ICAOs: 5 | Gold flights: 5" in latex
