from dataclasses import dataclass
from datetime import date

from data_engineering.flights.flight_type import get_flights


@dataclass
class CategoryDistribution:
    pass

@dataclass
class AirportDistribution:
    airport_types: str
    has_icao: bool

    # will have properties that get percentage 
    pass

@dataclass
class FlightsMetrics():
    day: date
    number: int

adsb_network = "adsblol"
source = "algorithm"
dt = date(2026, 3, 1)
get_flights(dt)