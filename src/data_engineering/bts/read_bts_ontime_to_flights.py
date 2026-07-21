from datetime import date, datetime, timedelta
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import polars as pl
from timezonefinder import TimezoneFinder
from data_engineering.flights.flight_type import with_flight_schema_columns
from airports.airport_lookup import AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from data_engineering.utils import OUTPUT_DIR
airport_lookup = AirportLookup()
tf = TimezoneFinder()

COLUMNS = set([
    "fl_date", "op_unique_carrier", "op_carrier_fl_num",
    "origin", "dest", "wheels_off", "wheels_on",
    "cancelled", "air_time", "diverted", "crs_dep_time", "crs_arr_time",
    "arr_delay", "tail_num",
])


OPENFLIGHTS_AIRLINES_PATH = "/Volumes/T2-SSD/planequery/data/raw/openflights/airlines.dat.txt"

BTS_FLIGHTS_VERSION = "v3"
BTS_OVERNIGHT_ROLLOVER_THRESHOLD_MINUTES = 12 * 60


def _hhmm_to_minutes(value: int) -> int:
    return (value // 100) * 60 + (value % 100)


def _has_value(value) -> bool:
    return value is not None and value == value


def _local_datetime(local_date: date, hhmm: int, timezone_name: str) -> datetime:
    return datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        tzinfo=ZoneInfo(timezone_name),
    ) + timedelta(minutes=_hhmm_to_minutes(hhmm))


def _closest_local_datetime(reference: datetime, hhmm: int, timezone_name: str) -> datetime:
    local_reference = reference.astimezone(ZoneInfo(timezone_name))
    base = _local_datetime(local_reference.date(), hhmm, timezone_name)
    candidates = [base - timedelta(days=1), base, base + timedelta(days=1)]
    return min(candidates, key=lambda candidate: abs(candidate - reference))


def _scheduled_arrival_datetime(row: dict) -> datetime:
    scheduled_departure = _local_datetime(
        row["fl_date"],
        row["crs_dep_time"],
        row["origin_timezone"],
    )
    scheduled_arrival = _local_datetime(
        row["fl_date"],
        row["crs_arr_time"],
        row["dest_timezone"],
    )
    while scheduled_arrival.astimezone(ZoneInfo("UTC")) <= scheduled_departure.astimezone(ZoneInfo("UTC")):
        scheduled_arrival += timedelta(days=1)
    return scheduled_arrival


def _compute_bts_timestamps_from_arrival_delay(row: dict) -> dict | None:
    required_fields = [
        "dest_timezone",
        "crs_arr_time",
        "arr_delay",
        "wheels_on",
    ]
    if any(not _has_value(row.get(field)) for field in required_fields):
        return None

    scheduled_arrival = _scheduled_arrival_datetime(row)
    actual_gate_arrival = scheduled_arrival + timedelta(minutes=row["arr_delay"])
    landing = _closest_local_datetime(
        actual_gate_arrival,
        row["wheels_on"],
        row["dest_timezone"],
    )
    landing_utc = landing.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    takeoff_utc = landing_utc - timedelta(minutes=row["air_time"])
    return {"time_takeoff": takeoff_utc, "time_landing": landing_utc}


def _compute_bts_timestamps(row: dict) -> dict:
    timestamps = _compute_bts_timestamps_from_arrival_delay(row)
    if timestamps is not None:
        return timestamps

    tz_name = row["origin_timezone"]
    fl_date = row["fl_date"]
    day = datetime(fl_date.year, fl_date.month, fl_date.day, tzinfo=ZoneInfo(tz_name))

    wheels_off_minutes = _hhmm_to_minutes(row["wheels_off"])
    scheduled_departure_minutes = _hhmm_to_minutes(row["crs_dep_time"])
    if (
        wheels_off_minutes < scheduled_departure_minutes
        and scheduled_departure_minutes - wheels_off_minutes > BTS_OVERNIGHT_ROLLOVER_THRESHOLD_MINUTES
    ):
        day += timedelta(days=1)

    takeoff = day + timedelta(minutes=wheels_off_minutes)
    takeoff_utc = takeoff.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    landing_utc = takeoff_utc + timedelta(minutes=row["air_time"])
    return {"time_takeoff": takeoff_utc, "time_landing": landing_utc}


def _bts_flights_parquet_path(year: int, month: int) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "bts"
        / BTS_FLIGHTS_VERSION
        / f"year={year}"
        / f"month={month:02d}"
        / "bts_flights.parquet"
    )

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
    return (
        scan_latest_faa_registration_icao_for_join(real_icao_column="icao")
        .collect()
    )

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
    df = (
        df.with_columns(
            pl.struct(
                "fl_date",
                "origin_timezone",
                "dest_timezone",
                "wheels_off",
                "wheels_on",
                "crs_dep_time",
                "crs_arr_time",
                "arr_delay",
                "air_time",
            )
            .map_elements(
                _compute_bts_timestamps,
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
    df = with_flight_schema_columns(df)
    df = df.with_columns(pl.col("icao").str.to_lowercase())
    flights_output_path = _bts_flights_parquet_path(year, month)
    os.makedirs(flights_output_path.parent, exist_ok=True)
    df.write_parquet(flights_output_path)
    return df

def get_bts_flights_for_day(target_date: date, use_cache = True) -> pl.DataFrame: # TODO: raise error if a particular date has no data. 
    path = _bts_flights_parquet_path(target_date.year, target_date.month)
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
    df = create_flights_from_bts(2026, 3)
    print(df)
