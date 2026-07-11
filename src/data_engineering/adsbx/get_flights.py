import polars as pl
from datetime import date
from data_engineering.openairframes.read import add_latest_icao_info
from data_engineering.utils import OUTPUT_DIR


def get_adsbx_flights_for_day(current_date: date) -> pl.DataFrame:
    '''
    returns with takeoff_times on previous day. arrivals on current_date
    '''
    current_date_str = current_date.strftime("%Y%m%d")
    path = f"{OUTPUT_DIR}/data/raw/adsb-exchange/flights-ax-v2/flights-ax-v2_ax_arrivals_{current_date_str}.csv"
    df = pl.read_csv(path)
    df = (
        df
        .rename({"hex": "icao", "reg": "registration", "orig": "takeoff_airport_ident", "dest": "landing_airport_ident"})
        .with_columns(
            pl.lit(None).cast(pl.String).alias("callsign"),
            pl.col("depTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms").alias("takeoff_time"),
            pl.col("arrTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms").alias("landing_time"),
        )
    )
    df = add_latest_icao_info(df)
    df = df.select(["icao", "callsign", "registration", "takeoff_time", "takeoff_airport_ident", "landing_time", "landing_airport_ident", "pia", "ladd", "military", "interesting"])
    return df
