import polars as pl
from collections.abc import Callable
from adsb.adsb_callsign_history import get_callsign_history

def _validate_callsign_history( # may do testing on this with the 2 icaos. 
    df_flights: pl.DataFrame,
    df_history: pl.DataFrame,
) -> list[str]:  # TODO: move to bts verfied/refernce file
    df_history = df_history.select(["icao", "callsign"])
    df_flights = df_flights.select(["icao", "callsign"])
    icaos = df_flights.get_column("icao").unique().to_list()
    valid_icaos = []
    for icao in icaos:
        df_0 = df_history.filter(pl.col("icao") == icao)
        df_1 = df_flights.filter(pl.col("icao") == icao)
        if len(df_0) != len(df_1):
            continue
        if len(df_1) < 4:
            continue
        if df_0.get_column("callsign").to_list() != df_1.get_column("callsign").to_list():
            continue
        valid_icaos.append(icao)
    return valid_icaos

def _check_consecutive_airports(df: pl.DataFrame):
    valid_icaos = (
    df
    .group_by("icao", maintain_order=True)
    .agg(
        (
            pl.col("takeoff_airport_ident").is_not_null()
            & (pl.col("takeoff_airport_ident") != "")
            & pl.col("landing_airport_ident").is_not_null()
            & (pl.col("landing_airport_ident") != "")
        ).all().alias("known_airports"),
        pl.col("landing_airport_ident")
        .shift(1)
        .eq_missing(pl.col("takeoff_airport_ident"))
        .slice(1)
        .all()
        .alias("is_valid")
    )
    .filter(pl.col("is_valid") & pl.col("known_airports"))
    .get_column("icao")
    .to_list()
    )
    return valid_icaos

def _filter_icaos(
    df_flights: pl.DataFrame,
    func: Callable[..., list[str]],
    *args: object,
) -> pl.DataFrame:
    icaos = df_flights.get_column("icao").unique().to_list()
    valid_icaos = func(df_flights, *args)
    df_flights = df_flights.filter(pl.col("icao").is_in(valid_icaos))
    print(f"Filtered {len(icaos) - len(valid_icaos)} ICAOs out of {len(icaos)} due to {func.__name__}")
    return df_flights

def verify_flights(df_flights, df_adsb):
    df_flights = _filter_icaos(df_flights, _check_consecutive_airports)
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb = df_adsb.filter(pl.col("icao").is_in(icaos))
    df_history = get_callsign_history(df_adsb)
    df_flights = _filter_icaos(df_flights, _validate_callsign_history, df_history)
    return df_flights
