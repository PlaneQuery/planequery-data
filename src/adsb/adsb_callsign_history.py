import polars as pl
from aircraft.callsign import is_valid_callsign
from datetime import date
import polars as pl

from flights.flight_source import get_flights
from flights.sources import BTS_SOURCE

def get_callsign_history(df):
    assert df.select(["icao", "time"]).equals(
        df.sort(["icao", "time"]).select(["icao", "time"])
    )

    df_history = (
        df
        .with_columns(
            pl.when(is_valid_callsign().fill_null(False))
            .then(pl.col("callsign"))
            .otherwise(None)
            .alias("callsign")
        )
        .with_columns(
            pl.col("callsign")
            .forward_fill()
            .backward_fill()
            .over("icao")
        )
        # Identify a new callsign interval within each ICAO.
        .with_columns(
            (
                pl.col("callsign")
                != pl.col("callsign").shift().over("icao")
            )
            .fill_null(True)
            .alias("new_range")
        )
        .with_columns(
            pl.col("new_range")
            .cum_sum()
            .over("icao")
            .alias("range_id")
        )
        .group_by(["icao", "range_id"], maintain_order=True)
        .agg(
            pl.col("time").first().alias("start_time"),
            pl.col("time").last().alias("last_seen_time"),
            pl.col("callsign").first(),
        )
        .drop("range_id")
    )

    return df_history



def validation(df_history: pl.DataFrame, df_flights: pl.DataFrame): # TODO: move to bts verfied/refernce file
    df_history = df_history.select(["icao", "callsign"])
    df_flights = df_flights.select(["icao", "callsign"])
    icaos = df_flights.get_column("icao").unique().to_list()
    valid_icaos = []
    for icao in icaos:
        df_0 = df_history.filter(pl.col("icao") == icao)
        df_1 = df_flights.filter(pl.col("icao") == icao)
        if len(df_0) != len(df_1):
            continue
        if df_0.get_column("callsign").to_list() != df_1.get_column("callsign").to_list():
            continue
        valid_icaos.append(icao)
    return valid_icaos
if __name__ == "__main__":
    from data_engineering.adsb.read_adsb import read_adsb
    df = read_adsb(date(2026, 3, 1))
    df_history = get_callsign_history(df)
    df_flights = get_flights(BTS_SOURCE, date(2026, 3, 1))
    df_flights_icao_num = df_flights.select("icao").unique().height
    print(f"df_flights_icao_num: {df_flights_icao_num}")
    valid_icaos = validation(df_history, df_flights)
    valid_icaos_num = len(valid_icaos)
    print(f"valid_icaos_num: {valid_icaos_num}")
    print("a008c5" in valid_icaos) 
    print("a19a21" in valid_icaos)


'''
TODO: make tests for these 2 icaos on 2026-03-01
a008c5 should succeed
a19a21 should fail
'''