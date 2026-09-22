# GOAL GET NUMBER OF FLIGHTS FROM THE VARIUS SOURCES: Algorihtms, SFDPS, BTS, Eurocontrol per day. Then I can go futher in on raw messages. 
from datetime import date

from data_engineering.flights.get_flights import FlightSource, get_flights
from planequery_data.utils import date_range
start_date = date(2026,3,1)
end_date = date(2026,6,1)
source = FlightSource.SFDPS

for dt in date_range(start_date, end_date):
    print(f"Getting flights for {dt} from {source}")
    df = get_flights(dt, source, adsb_src=None)
    print(f"Got {len(df)} flights for {dt} from {source}")