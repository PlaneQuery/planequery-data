from data_engineering.flights.flight_type import flights_df_to_flights
import polars as pl
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from utils import haversine

MAX_AIRPORT_DISTANCE_KM = 10
MAX_FLIGHT_BOUNDARY_TIME_GAP = timedelta(minutes=20)
MIN_IN_FLIGHT_DISTANCE_KM = 10
FLIGHT_INTERIOR_BUFFER = timedelta(minutes=30)
MIN_INTERIOR_DWELL_TIME = timedelta(minutes=45)
MAX_INTERIOR_DWELL_DISTANCE_KM = 10
MIN_INTERIOR_GAP_TIME = timedelta(minutes=45)
MAX_INTERIOR_GAP_AVERAGE_SPEED_KT = 60
READSB_HIST_PARQUET_ROOT = Path(
    "/Volumes/T2-SSD/planequery/data/processed/adsb-exchange/readsb-hist/parquet"
)
ADSB_MATCH_COLUMNS = ["time", "icao", "lat", "lon"]
_airport_lookup = None
_airport_lookup_loaded = False


@dataclass
class _AdsbRow:
    time: datetime
    lat: float
    lon: float

from bisect import bisect_left, bisect_right

def closest_time_row_idx(adsb_rows, target_time, lte=True):
    times = [row.time for row in adsb_rows]

    idx = (
        bisect_right(times, target_time) - 1
        if lte
        else bisect_left(times, target_time)
    )

    if idx < 0 or idx >= len(adsb_rows):
        return None

    return idx

def flights_match_adsb(df_flights: pl.DataFrame, df_adsb: pl.DataFrame) -> bool: 
    if df_flights.is_empty() or df_adsb.is_empty():
        return False

    airport_lookup = _get_airport_lookup()
    flights = flights_df_to_flights(df_flights.sort("takeoff_time"))
    adsb_rows = [
        _AdsbRow(time=row["time"], lat=row["lat"], lon=row["lon"])
        for row in df_adsb.sort("time").select(["time", "lat", "lon"]).iter_rows(named=True)
    ]
    assert any(flight.landing_time <= flight.takeoff_time for flight in flights) == False

    previous_flight = None
    previous_landing_row_idx = None
    for flight in flights:
        boundary_row_idxs = _boundary_rows_for_flight(adsb_rows, flight)
        if boundary_row_idxs is None:
            return False
        takeoff_row_idx, landing_row_idx = boundary_row_idxs

        if not _airport_ident_matches_adsb_row(
            airport_lookup,
            flight.takeoff_airport_ident,
            adsb_rows[takeoff_row_idx],
        ):
            return False
        if not _airport_ident_matches_adsb_row(
            airport_lookup,
            flight.landing_airport_ident,
            adsb_rows[landing_row_idx],
        ):
            return False

        if previous_flight:
            if previous_flight.landing_airport_ident != flight.takeoff_airport_ident:
                return False
            previous_landing_row = adsb_rows[previous_landing_row_idx]
            if _time_gap_exceeds(previous_landing_row.time, previous_flight.landing_time, MAX_FLIGHT_BOUNDARY_TIME_GAP):
                return False
        else:
            previous_landing_row_idx = 0

        airport_lat = adsb_rows[takeoff_row_idx].lat
        airport_lon = adsb_rows[takeoff_row_idx].lon

        if previous_flight:
            landing_row = adsb_rows[previous_landing_row_idx]
            airport_lat = landing_row.lat
            airport_lon = landing_row.lon

        in_between_rows = adsb_rows[previous_landing_row_idx:takeoff_row_idx]
        for row in in_between_rows:
            distance = haversine(row.lat, row.lon, airport_lat, airport_lon)
            if distance > MAX_AIRPORT_DISTANCE_KM:
                return False

        previous_flight = flight
        previous_landing_row_idx = landing_row_idx

    landing_row_idx = previous_landing_row_idx

    landing_row = adsb_rows[landing_row_idx]
    airport_lat = landing_row.lat
    airport_lon = landing_row.lon

    for row in adsb_rows[landing_row_idx:]:
        distance = haversine(row.lat, row.lon, airport_lat, airport_lon)
        if distance > MAX_AIRPORT_DISTANCE_KM:
            return False
    return True


def _readsb_hist_parquet_day_dir(
    target_date: date,
    parquet_root: Path = READSB_HIST_PARQUET_ROOT,
) -> Path:
    return (
        Path(parquet_root)
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
    )


def _scan_readsb_hist_day_parquet(
    target_date: date,
    parquet_root: Path = READSB_HIST_PARQUET_ROOT,
) -> pl.LazyFrame:
    day_dir = _readsb_hist_parquet_day_dir(target_date, parquet_root)
    parquet_glob = day_dir / "part*.parquet"
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end = day_start + timedelta(days=1)
    return (
        pl.scan_parquet(str(parquet_glob))
        .select(ADSB_MATCH_COLUMNS)
        .filter((pl.col("time") >= day_start) & (pl.col("time") < day_end))
    )


def _target_dates_from_flights(df_flights_full: pl.DataFrame) -> list[date]:
    if df_flights_full.is_empty():
        return []
    return (
        df_flights_full
        .select(pl.col("takeoff_time").dt.date().alias("target_date"))
        .unique()
        .sort("target_date")
        .get_column("target_date")
        .to_list()
    )


def _read_readsb_hist_adsb_for_flights(
    df_flights_full: pl.DataFrame,
    *,
    target_date: date | None = None,
    parquet_root: Path = READSB_HIST_PARQUET_ROOT,
) -> pl.DataFrame:
    icaos = df_flights_full.get_column("icao").unique().to_list()
    if not icaos:
        return pl.DataFrame(schema={"time": pl.Datetime("ms"), "icao": pl.String, "lat": pl.Float64, "lon": pl.Float64})

    target_dates = [target_date] if target_date is not None else _target_dates_from_flights(df_flights_full)
    if not target_dates:
        return pl.DataFrame(schema={"time": pl.Datetime("ms"), "icao": pl.String, "lat": pl.Float64, "lon": pl.Float64})

    return (
        pl.concat(
            [
                _scan_readsb_hist_day_parquet(day, parquet_root)
                for day in target_dates
            ]
        )
        .filter(pl.col("icao").is_in(icaos))
        .collect()
    )


def get_matching_icaos_in_flights(
    df_flights_full: pl.DataFrame,
    df_adsb_full: pl.DataFrame | None = None,
    *,
    use_readsb_parquet: bool = False,
    target_date: date | None = None,
    readsb_parquet_root: Path = READSB_HIST_PARQUET_ROOT,
):
    if use_readsb_parquet:
        df_adsb_full = _read_readsb_hist_adsb_for_flights(
            df_flights_full,
            target_date=target_date,
            parquet_root=readsb_parquet_root,
        )
    elif df_adsb_full is None:
        raise ValueError("df_adsb_full is required unless use_readsb_parquet=True")

    # TODO: this could be faster if you only load the icaos you need for df_adsb instead of full thing. 
    icaos = set()
    for key, df_flights in df_flights_full.group_by("icao"):
        icao = key[0]
        df_adsb = df_adsb_full.filter(pl.col("icao") == icao).sort("time")
        if flights_match_adsb(df_flights, df_adsb):
            icaos.add(icao)
    df = df_flights_full.filter(pl.col("icao").is_in(icaos))
    return df


    
