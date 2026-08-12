from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import polars as pl

from data_engineering.utils import FLIGHTS_VERSION, OUTPUT_DIR
from utils import haversine


FLIGHT_POLARS_SCHEMA = {
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "takeoff_airport_ident": pl.String,
    "landing_time": pl.Datetime("ms"),
    "landing_airport_ident": pl.String,
    "first_message_time": pl.Datetime("ms"),
    "first_lat": pl.Float64,
    "first_lon": pl.Float64,
    "first_baro_altitude_ft": pl.Int64,
    "first_geom_altitude_ft": pl.Int64,
    "last_message_time": pl.Datetime("ms"),
    "last_lat": pl.Float64,
    "last_lon": pl.Float64,
    "last_baro_altitude_ft": pl.Int64,
    "last_geom_altitude_ft": pl.Int64,
    "pia": pl.Boolean,
    "ladd": pl.Boolean,
    "military": pl.Boolean,
    "interesting": pl.Boolean,
    "aircraft_type": pl.String,
    "owner": pl.String,
    "aircraft_description": pl.String,
    "category": pl.String,
}

FLIGHT_COLUMN_DEFAULTS = {
    "callsign": "",
    "registration": "",
    "pia": False,
    "ladd": False,
    "military": False,
    "interesting": False,
    "aircraft_type": "",
    "owner": "",
    "aircraft_description": "",
    "category": "",
}

def with_flight_schema_columns(df: pl.DataFrame) -> pl.DataFrame:
    missing_columns = [
        pl.lit(FLIGHT_COLUMN_DEFAULTS.get(column_name), dtype=dtype).alias(column_name)
        for column_name, dtype in FLIGHT_POLARS_SCHEMA.items()
        if column_name not in df.columns
    ]
    if missing_columns:
        df = df.with_columns(missing_columns)
    return df.select(
        pl.col(column_name).cast(dtype).alias(column_name)
        for column_name, dtype in FLIGHT_POLARS_SCHEMA.items()
    )

def add_flight_duration_col(df: pl.DataFrame):
    df = df.with_columns((pl.col("landing_time") - pl.col("takeoff_time")).alias("flight_duration"))
    return df

@dataclass(eq=False)
class Flight:
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_airport_ident: str
    landing_airport_ident: str
    first_message_time: datetime | None = None
    first_lat: float | None = None
    first_lon: float | None = None
    first_baro_altitude_ft: int | None = None
    first_geom_altitude_ft: int | None = None
    last_message_time: datetime | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    last_baro_altitude_ft: int | None = None
    last_geom_altitude_ft: int | None = None
    callsign: str = ""
    registration: str = ""
    pia: bool = False
    ladd: bool = False
    military: bool = False
    interesting: bool = False
    aircraft_type: str = ""
    owner: str = ""
    aircraft_description: str = ""
    category: str = ""


    @property
    def flight_id(self) -> str:
        return self.icao + "_" + self.takeoff_time.strftime("%Y-%m-%d_%H-%M")

    def __hash__(self) -> int:
        return hash(self.flight_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Flight):
            return NotImplemented
        return self.flight_id == other.flight_id


class FlightIndex:

    def __init__(
        self,
        flights: list[Flight],
        airport_continuity_tolerance: float | None = None,

    ) -> None:
        """Index flights by ICAO, with PIA and LADD time indexes."""
        self.airport_continuity_tolerance_km = airport_continuity_tolerance
        self._flights_by_id = {flight.flight_id: flight for flight in flights}
        self.flights_by_icao: dict[str, list[Flight]] = {}

        pia_flights = []
        ladd_flights = []
        for flight in flights:
            self.flights_by_icao.setdefault(flight.icao, []).append(flight)
            if flight.ladd:
                ladd_flights.append(flight)
            if flight.pia:
                pia_flights.append(flight)

        for icao_flights in self.flights_by_icao.values():
            icao_flights.sort(key=lambda flight: flight.takeoff_time)

        self._pia_flights = sorted(pia_flights, key=lambda flight: flight.takeoff_time)
        self._pia_takeoff_times = [flight.takeoff_time for flight in self._pia_flights]
        self._ladd_flights = sorted(ladd_flights, key=lambda flight: flight.takeoff_time)
        self._ladd_takeoff_times = [flight.takeoff_time for flight in self._ladd_flights]

    def get_flight_by_id(self, flight_id: str) -> Flight:
        """Return the flight for ``flight_id``, or ``None`` when it is absent."""
        return self._flights_by_id[flight_id]

    def get_flights_ids(self) -> list[str]:
        """Return the IDs of all indexed flights."""
        return list(self._flights_by_id)

    def get_pia_flights(self) -> list[Flight]:
        """Return all PIA flights in takeoff-time order."""
        return self._pia_flights.copy()

    def get_ladd_flights(self) -> list[Flight]:
        """Return all LADD flights in takeoff-time order."""
        return self._ladd_flights.copy()

    def pia_flights_after(
        self,
        time: datetime,
        lat: float,
        lon: float,
        airport_ident: str,
        before_time: datetime | None = None,
    ) -> list[Flight]:
        """Return nearby PIA flights taking off at or after ``time``.

        When ``before_time`` is provided, only flights that land strictly
        before it are returned.

        A flight is nearby when it takes off from ``airport_ident`` or its first
        reported position is within :attr:`PIA_LOCATION_RADIUS_KM` of ``lat``,
        ``lon``.
        """
        return self._nearby_flights_after(
            self._pia_flights,
            self._pia_takeoff_times,
            time,
            lat,
            lon,
            airport_ident,
            before_time,
        )

    def ladd_flights_after(
        self,
        time: datetime,
        lat: float,
        lon: float,
        airport_ident: str,
        before_time: datetime | None = None,
    ) -> list[Flight]:
        """Return nearby LADD flights taking off at or after ``time``.

        When ``before_time`` is provided, only flights that land strictly
        before it are returned.
        """
        return self._nearby_flights_after(
            self._ladd_flights,
            self._ladd_takeoff_times,
            time,
            lat,
            lon,
            airport_ident,
            before_time,
        )

    def _nearby_flights_after(
        self,
        flights: list[Flight],
        takeoff_times: list[datetime],
        time: datetime,
        lat: float,
        lon: float,
        airport_ident: str,
        before_time: datetime | None,
    ) -> list[Flight]:
        index = bisect_left(takeoff_times, time)
        return [
            flight
            for flight in flights[index:]
            if (before_time is None or flight.landing_time < before_time)
            and (
                flight.takeoff_airport_ident == airport_ident
                or (
                    self.airport_continuity_tolerance_km is not None
                    and flight.first_lat is not None
                    and flight.first_lon is not None
                    and haversine(lat, lon, flight.first_lat, flight.first_lon)
                    <= self.airport_continuity_tolerance_km
                )
            )
        ]


def flights_df_to_flights(df: pl.DataFrame) -> list[Flight]:
    df = df.select([col for col in FLIGHT_POLARS_SCHEMA if col in df.columns])
    return [
        Flight(**row)
        for row in df.iter_rows(named=True)
    ]

def flights_to_flights_df(flights: list[Flight]) -> pl.DataFrame:
    df = pl.DataFrame([asdict(flight) for flight in flights], schema=FLIGHT_POLARS_SCHEMA)
    return df.select(list(FLIGHT_POLARS_SCHEMA.keys()))

FLIGHT_NUMBER_REGEX = r"^[A-Z]{3}\d{1,4}[A-Z]?$"
def flight_number_validation(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(
    pl.col("flight")
    .str.contains(FLIGHT_NUMBER_REGEX)
    )
    return df

def add_flight_id_col(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
    (
        pl.col("icao").cast(pl.Utf8)
        + "_"
        + pl.col("takeoff_time").dt.strftime("%Y-%m-%d_%H-%M")
    ).alias("flight_id")
    )
    return df


def flights_parquet_path(target_date: date):
    """Public data-lake path retained by planequery-data publishing jobs."""

    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / FLIGHTS_VERSION
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "part-0.parquet"
    )


ALGORITHM_ADSB_SOURCES = ("adsblol", "opensky", "adsbx")
FLIGHT_ALGORITHMS = ("algorithm", "opensky", "adsbx")


def _flights_algorithm_source_dir(adsb_src: str) -> str:
    if adsb_src not in ALGORITHM_ADSB_SOURCES:
        raise ValueError(
            f"Unknown flights algorithm ADS-B source: {adsb_src!r}. "
            f"Expected one of {ALGORITHM_ADSB_SOURCES!r}"
        )
    return adsb_src


def flights_algorithm_output_path(
    target_date: date,
    adsb_src: str = "adsblol",
    no_airports_model: bool = False,
):
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / ("algorithm-no-airports" if no_airports_model else "algorithm")
        / "v3"
        / _flights_algorithm_source_dir(adsb_src)
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "flights.parquet"
    )


def _date_range(start_date: date, end_date: date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


def _get_algorithm_flights_for_day(
    target_date: date,
    no_airports_model: bool = False,
    adsb_src: str = "adsblol",
):
    path = flights_algorithm_output_path(
        target_date,
        adsb_src=adsb_src,
        no_airports_model=no_airports_model,
    )
    df = pl.read_parquet(path)
    return with_flight_schema_columns(df)


def _get_flights_for_day(
    target_date: date,
    no_airports_model: bool = False,
    adsb_src: str = "adsblol",
    algorithm: str = "algorithm",
):
    if algorithm == "algorithm":
        return _get_algorithm_flights_for_day(
            target_date,
            no_airports_model=no_airports_model,
            adsb_src=adsb_src,
        )
    if algorithm == "opensky":
        from data_engineering.opensky.create_flights_from_trino_parquets import read_opensky_flights

        return with_flight_schema_columns(read_opensky_flights(target_date))
    if algorithm == "adsbx":
        from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day

        return with_flight_schema_columns(get_adsbx_flights_for_day(target_date))
    raise ValueError(
        f"Unknown flights algorithm: {algorithm!r}. "
        f"Expected one of {FLIGHT_ALGORITHMS!r}"
    )


def get_flights(
    start_date: date,
    end_date: date | None = None,
    no_airports_model: bool = False,
    adsb_src: str = "adsblol",
    algorithm: str = "algorithm",
):
    if end_date is None:
        return _get_flights_for_day(
            start_date,
            no_airports_model=no_airports_model,
            adsb_src=adsb_src,
            algorithm=algorithm,
        )
    if end_date <= start_date:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    return pl.concat([
        _get_flights_for_day(
            target_date,
            no_airports_model=no_airports_model,
            adsb_src=adsb_src,
            algorithm=algorithm,
        )
        for target_date in _date_range(start_date, end_date)
    ])

def add_lat_lon_to_airport_ident_columns(df: pl.DataFrame) -> pl.DataFrame:
    from airports.airport_lookup import AirportLookup

    airport_lookup = AirportLookup()

    def add_airport_coords(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        ident_col = f"{prefix}_airport_ident"
        lat_col = f"{prefix}_airport_lat"
        lon_col = f"{prefix}_airport_lon"

        def get_coords(airport_ident: str | None) -> dict[str, float | None]:
            coords = airport_lookup.get_airport_coordinates(airport_ident)
            if coords is None:
                return {lat_col: None, lon_col: None}

            lat, lon = coords
            return {lat_col: lat, lon_col: lon}

        return (
            df.with_columns(
                pl.col(ident_col)
                .map_elements(
                    get_coords,
                    return_dtype=pl.Struct({
                        lat_col: pl.Float64,
                        lon_col: pl.Float64,
                    }),
                )
                .alias("_airport_coords")
            )
            .unnest("_airport_coords")
        )

    df = add_airport_coords(df, "takeoff")
    return add_airport_coords(df, "landing")

def _is_american_icao_expr() -> pl.Expr:
    icao = pl.col("icao").str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def _pia_or_american_ladd_filter() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (_is_american_icao_expr() & pl.col("ladd").fill_null(False))
    )


def pia_or_american_ladd_icao_filter() -> pl.Expr:
    return _pia_or_american_ladd_filter().any().over("icao")


def create_pia_evaluation_df(flights: list[Flight], mappings: dict[Flight, str], gold_mappings: dict[Flight, str]) -> pl.DataFrame:
    labels = []
    predicted_icaos = []
    real_icaos = []
    for flight in flights:
        if flight in mappings:
            real_icao = mappings[flight]
            assert flight in gold_mappings, f"Flight {flight.flight_id} is in mappings but not in gold_mappings"
            gold_real_icao = gold_mappings[flight]
            if real_icao == gold_real_icao:
                labels.append(True)
            else:
                labels.append(False)
            predicted_icaos.append(real_icao)
            real_icaos.append(gold_real_icao)
        else:
            predicted_icaos.append(flight.icao)
            real_icaos.append(flight.icao)
            labels.append(None)
    df = flights_to_flights_df(flights).with_columns(
        pl.Series("predicted_icao", predicted_icaos),
        pl.Series("real_icao", real_icaos),
        pl.Series("correct", labels, dtype=pl.Boolean),
    )
    df = df.sort(["predicted_icao", "takeoff_time"])
    return df
