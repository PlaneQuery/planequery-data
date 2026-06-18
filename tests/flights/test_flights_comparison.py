#use  ac9113. Do conversion to text 

# flight takeoff_time, landing_time fixed, 

from datetime import date, datetime, time, timedelta
from data_engineering.flights.flight_type import Flight
from flights.flights_comparison import Match, flights_comparison
import unittest

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


start_dt = datetime.combine(date(2026,1,1), time.min)
end_dt = datetime.combine(date(2026,1,2), time.min)
flights_comparison(flights_0,flights_0, start_dt, end_dt)

def test_flights_comparison():
    f0_match, f1_match = flights_comparison(flights_0, flights_0, start_dt, end_dt)
    assert f0_match == [Match.TRUE]
    assert f1_match == [Match.TRUE]

    f0_match, f1_match = flights_comparison(flights_1, flights_1, start_dt, end_dt)
    matches = [Match.TRUE] * (len(flights_1))
    assert f0_match == matches
    assert f1_match == matches

    f0_match, f1_match = flights_comparison(flights_0, flights_1, start_dt, end_dt)
    assert f0_match == [Match.TRUE]
    assert f1_match == [Match.TRUE] + [Match.FALSE] * (len(f1_match) - 1)

    # check both ways 
    f0_match, f1_match = flights_comparison(flights_2, flights_3, start_dt, end_dt)
    f0_matches = [Match.TRUE] * len(flights_2)
    print(f0_matches)
    f1_matches = [Match.DISCARD] + f0_matches
    assert f0_match == f0_matches
    assert f1_match == f1_matches
