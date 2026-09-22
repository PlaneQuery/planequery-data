# GOAL GET NUMBER OF FLIGHTS FROM THE VARIUS SOURCES: Algorihtms, SFDPS, BTS, Eurocontrol per day. Then I can go futher in on raw messages. 
from datetime import date

from flights.flight_source import get_flights
from flights.providers.sfdps import SFDPSFlightSource
from planequery_data.utils import date_range
start_date = date(2026,3,1)
end_date = date(2026,6,1)
source = SFDPSFlightSource()

for dt in date_range(start_date, end_date):
    print(f"Getting flights for {dt} from {source}")
    df = get_flights(source, dt)
    print(f"Got {len(df)} flights for {dt} from {source}")
