#use  ac9113. Do conversion to text 

# flight takeoff_time, landing_time fixed, 

from datetime import date, datetime, time, timedelta
from data_engineering.flights.flight_type import Flight
from flights.flights_comparison import Match, df_flights_comparision, df_flights_comparison_stats, flights_comparison
import polars as pl
import pytest

def create_flight(takeoff_time: datetime):
    landing_time = takeoff_time + timedelta(hours=1)
    flight = Flight("a", takeoff_time,landing_time, "A", "B")
    return flight

flights_0 = [
   create_flight(datetime(2026,1,1,0,28))
]
flights_1 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,2,0)),
    create_flight(datetime(2026,1,1,3,0)),
    create_flight(datetime(2026,1,1,4,0)),
]

flights_2 = [
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
]
flights_3 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
]

flights_4 = [
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
    create_flight(datetime(2026,1,1,23,58))
]

flights_5 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
    create_flight(datetime(2026,1,1,23,58))
]

start_dt = datetime.combine(date(2026,1,1), time.min)
end_dt = datetime.combine(date(2026,1,2), time.min)
flights_comparison(flights_0,flights_0, start_dt, end_dt)

def test_flights_comparison():
    f0_match, f1_match = flights_comparison(flights_0, flights_0, start_dt, end_dt)
    assert f0_match == [0]
    assert f1_match == [0]

    f0_match, f1_match = flights_comparison(flights_1, flights_1, start_dt, end_dt)
    matches = [0, 1, 2, 3]
    assert f0_match == matches
    assert f1_match == matches

    f0_match, f1_match = flights_comparison(flights_0, flights_1, start_dt, end_dt)
    assert f0_match == [0]
    assert f1_match == [0] + [None] * (len(f1_match) - 1)

    # check both ways 
    f0_match, f1_match = flights_comparison(flights_2, flights_3, start_dt, end_dt)
    assert f0_match == [1,2]
    assert f1_match == ['DISCARD',0,1]

    f0_match, f1_match = flights_comparison(flights_2, flights_4, start_dt, end_dt)
    assert f0_match == [0,1]
    assert f1_match == [0,1, 'DISCARD']

    f0_match, f1_match = flights_comparison(flights_2, flights_5, start_dt, end_dt)
    assert f0_match == [1,2]
    assert f1_match == ["DISCARD",0,1, 'DISCARD']

    f0_match, f1_match = flights_comparison(flights_5, flights_2 , start_dt, end_dt)
    assert f0_match == ["DISCARD",0,1, 'DISCARD']
    assert f1_match == [1,2]

# I can have a fixed dataset here. 

def test_df_flights_comparision_with_icao_only_in_one_df():
    df_0 = pl.DataFrame(
        {
            "icao": ["a"],
            "takeoff_time": [datetime(2026, 1, 1, 3)],
            "landing_time": [datetime(2026, 1, 1, 4)],
            "takeoff_airport_ident": ["A"],
            "landing_airport_ident": ["B"],
        }
    )
    df_1 = pl.DataFrame(
        {
            "icao": ["b"],
            "takeoff_time": [datetime(2026, 1, 1, 3)],
            "landing_time": [datetime(2026, 1, 1, 4)],
            "takeoff_airport_ident": ["A"],
            "landing_airport_ident": ["B"],
        }
    )

    comparison = df_flights_comparision(df_0, df_1, start_dt, end_dt)

    assert sorted(comparison.get_column("match_status").to_list()) == ["df0_only", "df1_only"]

def test_df_flights_comparison_stats():
    comparison = pl.DataFrame(
        {
            "match_status": ["both", "both", "df0_only", "df1_only", "df1_only"],
        }
    )

    stats = df_flights_comparison_stats(comparison)

    assert stats["true_positive"] == 2
    assert stats["false_positive"] == 1
    assert stats["false_negative"] == 2
    assert stats["precision"] == pytest.approx(2 / 3)
    assert stats["recall"] == pytest.approx(1 / 2)
    assert stats["f1"] == pytest.approx(4 / 7)

def test_df_flights_comparison_stats_with_no_matches():
    comparison = pl.DataFrame({"match_status": []}, schema={"match_status": pl.String})

    stats = df_flights_comparison_stats(comparison)

    assert stats == {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

def test_df_test_df_bts_flights():
    from datetime import date, datetime, time
    from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
    from data_engineering.adsbx.readsb import read_adsbx_readsb
    from flights.flights_match_adsb import get_matching_icaos_in_flights
    import polars as pl

    test_date = date(2026,3,1)
    df_adsb = read_adsbx_readsb()
    df_gold = get_bts_flights_for_day(test_date)
    df_gold = get_matching_icaos_in_flights(df_gold, df_adsb)
    df_test = pl.read_parquet('/Volumes/T2-SSD/planequery/data/flights/trad-model/2026-03-01.parquet')
    expr = pl.col("takeoff_time").dt.date() == test_date 
    df_gold = df_gold.filter(expr)
    df_test = df_test.filter(expr)
    df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").unique().implode()))
    from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
    df = df_flights_comparision(df_test, df_gold, datetime.combine(test_date, time.min), datetime.combine(test_date, time.max), compare_airports=False)
    stats = df_flights_comparison_stats(df)
    assert stats['f1'] > 0.99
    assert stats['precision'] > 0.99
    assert stats['recall'] > 0.99