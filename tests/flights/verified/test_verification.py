'''
TODO: make tests for these 2 icaos on 2026-03-01
a008c5 should succeed
a19a21 should fail
'''

from datetime import date

from adsb.adsb_callsign_history import get_callsign_history
from data_engineering.adsb.canonical import ADSBSource
from data_engineering.adsb.read_adsb import read_adsb
from flights.flight_source import get_flights
from flights.sources import BTS_SOURCE
from flights.verified.verification import _validate_callsign_history

def test_0():
    # 2026-03-01
    target_dt = date(2026, 3, 1)
    df_flights = get_flights(BTS_SOURCE, target_dt)
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb = read_adsb(target_dt, source= ADSBSource.ADSBLOL, icaos=icaos)
    df_history = get_callsign_history(df_adsb)
    icaos = _validate_callsign_history(df_flights, df_history)
    assert "a008c5" in icaos
    assert "a19a21" not in icaos
