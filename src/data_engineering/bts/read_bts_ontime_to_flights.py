from datetime import date, datetime, timedelta
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import polars as pl
from timezonefinder import TimezoneFinder
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA
from airports.airportLookup import AirportLookup
from data_engineering.flights.flights_and_adsb_match import get_icaos_with_perfect_match
from data_engineering.adsb.read_adsb import read_adsb
airport_lookup = AirportLookup()
tf = TimezoneFinder()

COLUMNS = set([
    "fl_date", "op_unique_carrier", "op_carrier_fl_num",
    "origin", "dest", "wheels_off", "wheels_on",
    "cancelled", "air_time", "diverted", "crs_dep_time", "tail_num",
])


OPENFLIGHTS_AIRLINES_PATH = "/Volumes/T2-SSD/planequery/data/raw/openflights/airlines.dat.txt"
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

def ident_to_timezone_mapping():
    airport_lookup.ident_index
    ident_to_timezone = {}
    for ident, airport in airport_lookup.ident_index.items():
        tz = tf.timezone_at(lng=airport.lon, lat=airport.lat)
        if tz:
            ident_to_timezone[ident] = tz
    lookup = pl.DataFrame({"ident": list(ident_to_timezone.keys()), "timezone": list(ident_to_timezone.values())})
    return lookup
def iata_to_ident_mapping():
    airport_lookup.iata_to_ident_index
    lookup = pl.DataFrame({"iata": list(airport_lookup.iata_to_ident_index.keys()), "ident": list(airport_lookup.iata_to_ident_index.values())})
    return lookup
def get_registration_to_icao() -> pl.DataFrame:
    df_faa = pl.read_csv(OUTPUT_DIR/ "data/raw" /"openairframes"/"openairframes_faa_2023-08-16_2026-03-25.csv",ignore_errors=True)
    df_faa = df_faa.select(["registration_number", "transponder_code_hex"])
    df_faa = df_faa.rename({"registration_number": "registration", "transponder_code_hex": "icao"})
    df_faa = df_faa.unique(subset=["registration"])
    return df_faa

def create_flights_from_bts(year: int, month: int) -> pl.DataFrame:
    '''
    On 2025, 10 for example some flights from UTC day 2025-11-01 and 2025-09-30 are included
    '''
    # bts raw data is in OUTPUT_DIR / 'data/raw/bts/year=2025/month=10/T_ONTIME_REPORTING.csv'
    bts_path = OUTPUT_DIR / "data" / "raw" / "bts" / f"year={year}" / f"month={month:02d}" / "T_ONTIME_REPORTING.csv"
    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1
    prev_bts_path = OUTPUT_DIR / "data" / "raw" / "bts" / f"year={prev_year}" / f"month={prev_month:02d}" / "T_ONTIME_REPORTING.csv"
    dfs = []
    for path in [prev_bts_path, bts_path]:
        if path.exists():
            df_tmp = pl.read_csv(path)
            df_tmp = df_tmp.rename({c: c.lower() for c in df_tmp.columns})
            df_tmp = df_tmp.select(COLUMNS)
            dfs.append(df_tmp)
    df = pl.concat(dfs).unique()
    # filter out canclled, divered, airprots we can't identify. Tail_Numbers we can't idetnify.
    df = df.filter((pl.col("cancelled") == 0) & (pl.col("diverted") == 0))
    df = df.filter(pl.col("air_time") > 30)
    df = df.rename({"tail_num": "registration"})
    mapping = get_registration_to_icao()
    df = df.join(mapping, on="registration", how="inner")
    mapping = iata_to_ident_mapping()
    df = df.join(mapping, left_on="origin", right_on="iata", how="inner").rename({"ident": "takeoff_airport_ident"})
    df = df.join(mapping, left_on="dest", right_on="iata", how="inner").rename({"ident": "landing_airport_ident"})
    mapping = ident_to_timezone_mapping()
    df = df.join(mapping, left_on="takeoff_airport_ident", right_on="ident", how="inner").rename({"timezone": "origin_timezone"})
    df = df.join(mapping, left_on="landing_airport_ident", right_on="ident", how="inner").rename({"timezone": "dest_timezone"})
    df = df.with_columns(
        pl.col("fl_date")
        .str.to_datetime(format="%m/%d/%Y %I:%M:%S %p", strict=False)
        .dt.strftime("%Y-%m-%d")
        .alias("fl_date")
    )

    df = df.with_columns(
        pl.col("fl_date").str.to_date(format="%Y-%m-%d").alias("fl_date")
    )
    def compute_timestamps(row: dict) -> dict:
        tz_name = row["origin_timezone"]
        fl_date = row["fl_date"]  # datetime.date from polars Date column
        day = datetime(fl_date.year, fl_date.month, fl_date.day, tzinfo=ZoneInfo(tz_name))
        if row["wheels_off"] < row["crs_dep_time"]:
            day += timedelta(days=1)
        takeoff = day + timedelta(hours=row["wheels_off"] // 100, minutes=row["wheels_off"] % 100)
        takeoff_utc = takeoff.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        landing_utc = takeoff_utc + timedelta(minutes=row["air_time"])
        return {"time_takeoff": takeoff_utc, "time_landing": landing_utc}

    df = (
        df.with_columns(
            pl.struct("fl_date", "origin_timezone", "wheels_off", "crs_dep_time", "air_time")
            .map_elements(
                compute_timestamps,
                return_dtype=pl.Struct({"time_takeoff": pl.Datetime("us"), "time_landing": pl.Datetime("us")}),
            )
            .struct.unnest()
        )
    )
    df = df.with_columns(
        pl.lit(None).cast(pl.String).alias("callsign"),
        pl.lit(False).alias("pia"),
        pl.lit(False).alias("ladd"),
        pl.lit(False).alias("military"),
        pl.lit(False).alias("interesting"),
        pl.lit(None).cast(pl.String).alias("aircraft_type"),
        pl.lit(None).cast(pl.String).alias("owner"),
        pl.lit(None).cast(pl.String).alias("aircraft_description"),
        pl.lit("A3").cast(pl.String).alias("category"),
    )
    df = df.rename({"time_takeoff": "takeoff_time", "time_landing": "landing_time"})
    df = df.select(list(FLIGHT_POLARS_SCHEMA.keys()))
    for col_name, dtype in FLIGHT_POLARS_SCHEMA.items():
        df = df.with_columns(pl.col(col_name).cast(dtype))
    df = df.with_columns(pl.col("icao").str.to_lowercase())
    flights_output_path = OUTPUT_DIR / "data" / "flights" / "bts" / "v1" / f"year={year}" / f"month={month:02d}" / "bts_flights.parquet"
    os.makedirs(flights_output_path.parent, exist_ok=True)
    df.write_parquet(flights_output_path)
    return df

def get_bts_flights_for_day(target_date: date, use_cache = True) -> pl.DataFrame: # TODO: raise error if a particular date has no data. 
    path = OUTPUT_DIR / "data" / "flights" / "bts" / "v1" / f"year={target_date.year}" / f"month={target_date.month:02d}" / "bts_flights.parquet"
    if use_cache == False or not path.exists():
        create_flights_from_bts(target_date.year, target_date.month)
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("takeoff_time").dt.date() == target_date) 
        # & (pl.col("landing_time").dt.date() == current_date.date())
    )
    overlaps = (
    df.sort(["icao", "takeoff_time"])
    .with_columns(
        pl.col("landing_time").shift(1).over("icao").alias("prev_landing_time")
    )
    .filter(pl.col("takeoff_time") < pl.col("prev_landing_time"))
    )
    print(len(overlaps), "overlapping flights found for", target_date)
    if len(overlaps) > 0:
        df = df.filter(~pl.col("icao").is_in(overlaps["icao"].implode()))
    df = df.sort(["icao", "takeoff_time"])
    return df

if __name__ == "__main__":
    df = create_flights_from_bts(2025, 10)
    print(df)
