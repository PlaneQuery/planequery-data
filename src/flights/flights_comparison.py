
from datetime import timedelta
from enum import Enum
from data_engineering.flights.flight_type import Flight

class Match(Enum):
    FALSE = 0
    TRUE = 1
    DISCARD = 2
# do I want to have seperate rejections for takeoff/landing or airport ident and maybe I can pick a particular one? 
# maybe I can return a list like [True, False, True, False] "matched", "matched", "airport_ident x != y",
MAX_TIME_DIFF = timedelta(hours=1)
def compare_flight(f0: Flight, f1: Flight, compare_airports = True):
    times_match = False
    if abs(f0.takeoff_time - f1.takeoff_time) > MAX_TIME_DIFF or abs(f0.landing_time - f1.landing_time) > MAX_TIME_DIFF:
        return False
    if compare_airports:
        if f0.takeoff_airport_ident != f1.takeoff_airport_ident or f0.landing_airport_ident != f1.landing_airport_ident:
            return False
    return True

def flights_comparison(flights_0: list[Flight], flights_1: list[Flight], start_dt, end_dt, grace_period = timedelta(minutes=30), compare_airports= True):
    '''
    we assume flights are assigned to days by takeoff_time. Only 1 flight in grace period at either end.
    '''
    i = 0
    f0_match = [Match.FALSE for _ in flights_0]
    f1_match = [Match.FALSE for _ in flights_1]

    while i < min(len(flights_0), len(flights_1)):
        f0 = flights_0[i]
        f1 = flights_1[i]
        flight_match =  compare_flight(f0,f1, compare_airports)
        if flight_match:
            f0_match[i] = Match.TRUE
            f1_match[i] = Match.TRUE
            i+=1
            continue
        flights_len_match = len(flights_0) == len(flights_1)
        if not flights_len_match:
            f0_in_grace = f0.takeoff_time - start_dt <= grace_period
            f1_in_grace = f1.takeoff_time -start_dt <= grace_period
            if i == 0:
                if f0_in_grace and not f1_in_grace:
                    f0_match[0] = Match.DISCARD
                    flights_0 = flights_0[1:]
                    continue
                if not f0_in_grace and f1_in_grace:
                    f1_match[0] = Match.DISCARD
                    flights_1 = flights_1[1:]
                    continue
        i += 1
    
    if len(flights_0) != len(flights_1):            
        if len(flights_0) +1 == len(flights_1):
            f0 = flights_0[-1]
            f0_in_grace = end_dt - f0.takeoff_time <= grace_period
            f0_match[-1] = Match.DISCARD
        elif len(flights_1) + 1 == len(flights_0):
            f1 = flights_1[-1]
            f1_in_grace = end_dt - f1.takeoff_time <= grace_period
            f1_match[-1] = Match.DISCARD
    return (f0_match, f1_match)

'''
let's go through scenarios 
first flight of flights_0 dosen't match first flight of flights_1 or first flight of flights_1 dosen't match that of flights_0 and there is uneuqla number of flights. 
last flight of flights_0 dosen't match last flight of flights_1

flights_0 has 1 or 2 more flights than flights_1.
flights_0 has 
flights_1 has more flights than flights_1
what about when they both have a seperate but diffrent flight from cutoff. 

Maybe we go through check with all flights and then check if the flights that have no match could be eliminated 
'''