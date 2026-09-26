from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from airports.catalog import AirportCatalogView


FLIGHT_POLARS_SCHEMA = {
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "takeoff_airport_id": pl.Int64,
    "takeoff_airport_ident": pl.String,
    "landing_time": pl.Datetime("ms"),
    "landing_airport_id": pl.Int64,
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
    takeoff_airport_ident: str | None
    landing_airport_ident: str | None
    takeoff_airport_id: int | None = None
    landing_airport_id: int | None = None
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

    def __init__(self, flights: list[Flight]) -> None:
        """Index flights by ICAO, with PIA and LADD time indexes."""
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
        """Return the flight for ``flight_id``."""
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

    def remove_flights(self, flight_ids: set[str]) -> None:
        """Remove flights from every lookup maintained by the index."""
        if not flight_ids:
            return

        for flight_id in flight_ids:
            self._flights_by_id.pop(flight_id, None)

        for icao, flights in list(self.flights_by_icao.items()):
            remaining_flights = [
                flight for flight in flights if flight.flight_id not in flight_ids
            ]
            if remaining_flights:
                self.flights_by_icao[icao] = remaining_flights
            else:
                del self.flights_by_icao[icao]

        self._pia_flights = [
            flight for flight in self._pia_flights if flight.flight_id not in flight_ids
        ]
        self._pia_takeoff_times = [
            flight.takeoff_time for flight in self._pia_flights
        ]
        self._ladd_flights = [
            flight for flight in self._ladd_flights if flight.flight_id not in flight_ids
        ]
        self._ladd_takeoff_times = [
            flight.takeoff_time for flight in self._ladd_flights
        ]

    def pia_flights_after(
        self,
        time: datetime,
        airport_id: int | None,
        before_time: datetime | None = None,
    ) -> list[Flight]:
        """Return PIA flights departing ``airport_id`` at or after ``time``.

        When ``before_time`` is provided, only flights that land strictly
        before it are returned.
        """
        return self._flights_after_from_airport(
            self._pia_flights,
            self._pia_takeoff_times,
            time,
            airport_id,
            before_time,
        )

    def ladd_flights_after(
        self,
        time: datetime,
        airport_id: int | None,
        before_time: datetime | None = None,
    ) -> list[Flight]:
        """Return LADD flights departing ``airport_id`` at or after ``time``.

        When ``before_time`` is provided, only flights that land strictly
        before it are returned.
        """
        return self._flights_after_from_airport(
            self._ladd_flights,
            self._ladd_takeoff_times,
            time,
            airport_id,
            before_time,
        )

    def _flights_after_from_airport(
        self,
        flights: list[Flight],
        takeoff_times: list[datetime],
        time: datetime,
        airport_id: int | None,
        before_time: datetime | None,
    ) -> list[Flight]:
        index = bisect_left(takeoff_times, time)
        return [
            flight
            for flight in flights[index:]
            if (before_time is None or flight.landing_time < before_time)
            and (
                airport_id is not None
                and flight.takeoff_airport_id == airport_id
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

def add_lat_lon_to_airport_columns(
    df: pl.DataFrame,
    airport_catalog: "AirportCatalogView",
) -> pl.DataFrame:
    def add_airport_coords(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        id_col = f"{prefix}_airport_id"
        lat_col = f"{prefix}_airport_lat"
        lon_col = f"{prefix}_airport_lon"
        coordinates = pl.DataFrame(
            {
                id_col: [airport.id for airport in airport_catalog.airports],
                lat_col: [airport.lat for airport in airport_catalog.airports],
                lon_col: [airport.lon for airport in airport_catalog.airports],
            },
            schema={
                id_col: pl.Int64,
                lat_col: pl.Float64,
                lon_col: pl.Float64,
            },
        )
        return df.join(coordinates, on=id_col, how="left")

    df = add_airport_coords(df, "takeoff")
    return add_airport_coords(df, "landing")

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
