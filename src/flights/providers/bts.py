from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from timezonefinder import TimezoneFinder

from airports.catalog import AirportCatalogView
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from data_engineering.utils import OUTPUT_DIR

tf = TimezoneFinder()

BASE_COLUMNS = [
    "fl_date", "op_unique_carrier", "op_carrier_fl_num",
    "origin", "dest", "wheels_off", "wheels_on",
    "cancelled", "air_time", "diverted", "crs_dep_time", "crs_arr_time",
    "arr_delay", "tail_num",
]

DIVERSION_COUNT = 5
DIVERSION_COLUMNS = [
    "dep_delay",
    "actual_elapsed_time",
    "div_airport_landings",
    "div_reached_dest",
    "div_actual_elapsed_time",
    "div_arr_delay",
    *[
        f"div{index}_{field}"
        for index in range(1, DIVERSION_COUNT + 1)
        for field in ("airport", "wheels_on", "wheels_off", "tail_num")
    ],
]
COLUMNS = set([*BASE_COLUMNS, *DIVERSION_COLUMNS])

DIVERSION_COLUMN_DTYPES = {
    "dep_delay": pl.Float64,
    "actual_elapsed_time": pl.Float64,
    "div_airport_landings": pl.Int64,
    "div_reached_dest": pl.Float64,
    "div_actual_elapsed_time": pl.Float64,
    "div_arr_delay": pl.Float64,
    **{
        f"div{index}_{field}": (
            pl.String if field in {"airport", "tail_num"} else pl.Int64
        )
        for index in range(1, DIVERSION_COUNT + 1)
        for field in ("airport", "wheels_on", "wheels_off", "tail_num")
    },
}


# BTS operating-carrier codes to their FAA/ICAO three-letter designators.
# Keep this explicit: the previously referenced OpenFlights data is stale for
# current regional carriers such as Envoy (MQ), PSA (OH), and Republic (YX).
# Source: https://www.aspm.faa.gov/aspmhelp/index/ASQP__Carrier_Codes_and_Names.html
BTS_CARRIER_ICAO_CODES = {
    "9E": "EDV",
    "AA": "AAL",
    "AS": "ASA",
    "AX": "LOF",
    "B6": "JBU",
    "C5": "UCA",
    "CP": "CPZ",
    "DL": "DAL",
    "EM": "CFS",
    "EV": "ASQ",
    "F9": "FFT",
    "G4": "AAY",
    "G7": "GJS",
    "HA": "HAL",
    "KS": "NLA",
    "MQ": "ENY",
    "NK": "NKS",
    "OH": "JIA",
    "OO": "SKW",
    "PT": "PDT",
    "QX": "QXE",
    "UA": "UAL",
    "WN": "SWA",
    "YV": "ASH",
    "YX": "RPA",
    "ZW": "AWI",
}

BTS_OVERNIGHT_ROLLOVER_THRESHOLD_MINUTES = 12 * 60
MAX_DIVERSION_LEG_DURATION = timedelta(hours=20)
BTS_REQUIRED_TIMING_FIELDS = ("air_time", "wheels_on", "arr_delay")


def _bts_callsign_expr() -> pl.Expr:
    """Build the expected ATC callsign from BTS operating-flight fields."""
    carrier_designator = pl.col("op_unique_carrier").replace_strict(
        BTS_CARRIER_ICAO_CODES,
        default=None,
        return_dtype=pl.String,
    )
    flight_number = (
        pl.col("op_carrier_fl_num")
        .cast(pl.Int64, strict=False)
        .cast(pl.String)
    )
    return pl.concat_str([carrier_designator, flight_number]).alias("callsign")


def _hhmm_to_minutes(value: int) -> int:
    return (value // 100) * 60 + (value % 100)


def _has_value(value) -> bool:
    return value is not None and value == value


def _diversion_report_expr() -> pl.Expr:
    return (
        (pl.col("diverted").fill_null(0) == 1)
        | (pl.col("div_airport_landings").fill_null(0) > 0)
    )


def _coerce_hhmm(value: object) -> int | None:
    """Normalize numeric BTS clock values, including zero-padded strings."""
    if not _has_value(value):
        return None
    try:
        hhmm = int(float(value))
    except (TypeError, ValueError):
        return None
    hours, minutes = divmod(hhmm, 100)
    if hhmm < 0 or minutes >= 60 or hours > 24 or (hours == 24 and minutes):
        return None
    return hhmm


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


def _next_utc_datetime(
    reference: datetime,
    hhmm: int,
    timezone_name: str,
) -> datetime:
    """Return the first local BTS clock occurrence strictly after reference."""
    reference_utc = reference.replace(tzinfo=UTC)
    local_reference = reference_utc.astimezone(ZoneInfo(timezone_name))
    candidate = _local_datetime(
        local_reference.date(),
        hhmm,
        timezone_name,
    )
    while candidate.astimezone(UTC) <= reference_utc:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC).replace(tzinfo=None)


def _diverted_takeoff_datetime(row: dict, timezone_name: str) -> datetime:
    """Anchor a diverted flight's first takeoff to its actual departure day."""
    wheels_off = _coerce_hhmm(row.get("wheels_off"))
    if wheels_off is None:
        raise ValueError("Diverted BTS row is missing a valid wheels_off time")

    scheduled_departure = _local_datetime(
        row["fl_date"],
        row["crs_dep_time"],
        timezone_name,
    )
    if _has_value(row.get("dep_delay")):
        actual_gate_departure = scheduled_departure.astimezone(UTC) + (
            timedelta(minutes=float(row["dep_delay"]))
        )
        takeoff = _closest_local_datetime(
            actual_gate_departure,
            wheels_off,
            timezone_name,
        )
    else:
        takeoff = scheduled_departure.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) + timedelta(minutes=_hhmm_to_minutes(wheels_off))
        scheduled_minutes = _hhmm_to_minutes(row["crs_dep_time"])
        wheels_off_minutes = _hhmm_to_minutes(wheels_off)
        if (
            wheels_off_minutes < scheduled_minutes
            and scheduled_minutes - wheels_off_minutes
            > BTS_OVERNIGHT_ROLLOVER_THRESHOLD_MINUTES
        ):
            takeoff += timedelta(days=1)
    return takeoff.astimezone(UTC).replace(tzinfo=None)


def _diverted_final_landing_datetime(
    row: dict,
    origin_timezone: str,
    destination_timezone: str,
) -> datetime | None:
    """Anchor final wheels-on using BTS's reported gate-to-gate duration."""
    elapsed_time = row.get("div_actual_elapsed_time")
    if not _has_value(elapsed_time):
        elapsed_time = row.get("actual_elapsed_time")
    wheels_on = _coerce_hhmm(row.get("wheels_on"))
    if (
        not _has_value(elapsed_time)
        or not _has_value(row.get("dep_delay"))
        or wheels_on is None
    ):
        return None

    scheduled_departure = _local_datetime(
        row["fl_date"],
        row["crs_dep_time"],
        origin_timezone,
    )
    actual_gate_departure = scheduled_departure.astimezone(UTC) + timedelta(
        minutes=float(row["dep_delay"])
    )
    actual_gate_arrival = actual_gate_departure + timedelta(
        minutes=float(elapsed_time)
    )
    landing = _closest_local_datetime(
        actual_gate_arrival,
        wheels_on,
        destination_timezone,
    )
    return landing.astimezone(UTC).replace(tzinfo=None)


def _valid_diversion_leg(takeoff: datetime, landing: datetime) -> bool:
    duration = landing - takeoff
    return timedelta(0) < duration <= MAX_DIVERSION_LEG_DURATION


def _utc_clock_candidates(
    hhmm: int,
    timezone_name: str,
    *,
    after: datetime,
    before: datetime,
) -> list[datetime]:
    """Return occurrences of one local clock in an open UTC interval."""
    timezone = ZoneInfo(timezone_name)
    start_date = after.replace(tzinfo=UTC).astimezone(timezone).date()
    end_date = before.replace(tzinfo=UTC).astimezone(timezone).date()
    candidates: set[datetime] = set()
    local_date = start_date - timedelta(days=1)
    while local_date <= end_date + timedelta(days=1):
        candidate = _local_datetime(
            local_date,
            hhmm,
            timezone_name,
        ).astimezone(UTC).replace(tzinfo=None)
        if after < candidate < before:
            candidates.add(candidate)
        local_date += timedelta(days=1)
    return sorted(candidates)


def _diverted_flight_rows(
    row: dict,
    airport_timezones: dict[str, str],
) -> list[dict]:
    """Expand one diverted BTS report into its actual airborne legs."""
    origin = row.get("origin")
    registration = row.get("tail_num")
    origin_timezone = airport_timezones.get(origin)
    if (
        not _has_value(origin)
        or not _has_value(registration)
        or origin_timezone is None
        or not _has_value(row.get("crs_dep_time"))
    ):
        return []

    try:
        takeoff_time = _diverted_takeoff_datetime(row, origin_timezone)
    except (TypeError, ValueError):
        return []

    populated_diversions = [
        index
        for index in range(1, DIVERSION_COUNT + 1)
        if _has_value(row.get(f"div{index}_airport"))
    ]
    if not populated_diversions or populated_diversions != list(
        range(1, populated_diversions[-1] + 1)
    ):
        return []

    reported_landings = row.get("div_airport_landings")
    if _has_value(reported_landings):
        reported_landings = int(reported_landings)
        if 1 <= reported_landings <= DIVERSION_COUNT:
            if len(populated_diversions) != reported_landings:
                return []
        elif reported_landings != 9:
            return []
    landing_count = len(populated_diversions)
    fly_return = reported_landings == 9
    reached_destination = row.get("div_reached_dest") == 1 or (
        fly_return
        and row.get("cancelled") == 0
        and _coerce_hhmm(row.get("wheels_on")) is not None
    )

    destination = row.get("dest")
    destination_timezone = airport_timezones.get(destination)
    final_landing_time = None
    if reached_destination:
        if not _has_value(destination) or destination_timezone is None:
            return []
        final_landing_time = _diverted_final_landing_datetime(
            row,
            origin_timezone,
            destination_timezone,
        )
        if final_landing_time is not None and takeoff_time >= final_landing_time:
            return []

    diversions = []
    for index in range(1, landing_count + 1):
        diversion_airport = row.get(f"div{index}_airport")
        wheels_on = _coerce_hhmm(row.get(f"div{index}_wheels_on"))
        diversion_timezone = airport_timezones.get(diversion_airport)
        if (
            not _has_value(diversion_airport)
            or wheels_on is None
            or diversion_timezone is None
        ):
            return []
        wheels_off = _coerce_hhmm(row.get(f"div{index}_wheels_off"))
        next_registration = row.get(f"div{index}_tail_num")
        needs_another_takeoff = index < landing_count or reached_destination
        if needs_another_takeoff and (
            wheels_off is None or not _has_value(next_registration)
        ):
            return []
        diversions.append(
            {
                "airport": diversion_airport,
                "timezone": diversion_timezone,
                "wheels_on": wheels_on,
                "wheels_off": wheels_off,
                "next_registration": next_registration,
            }
        )

    if reached_destination and final_landing_time is None:
        final_wheels_on = _coerce_hhmm(row.get("wheels_on"))
        if final_wheels_on is None:
            return []

    def flight_row(
        leg_origin: str,
        leg_destination: str,
        leg_registration: str,
        leg_takeoff: datetime,
        leg_landing: datetime,
    ) -> dict:
        flight = {
            "registration": leg_registration,
            "origin": leg_origin,
            "dest": leg_destination,
            "takeoff_time": leg_takeoff,
            "landing_time": leg_landing,
        }
        if "callsign" in row:
            flight["callsign"] = row["callsign"]
        return flight

    if final_landing_time is not None:
        first_diversion = diversions[0]
        first_landing_deadline = min(
            takeoff_time
            + MAX_DIVERSION_LEG_DURATION
            + timedelta(microseconds=1),
            final_landing_time,
        )
        first_landing_candidates = _utc_clock_candidates(
            first_diversion["wheels_on"],
            first_diversion["timezone"],
            after=takeoff_time,
            before=first_landing_deadline,
        )
        main_wheels_off = _coerce_hhmm(row.get("wheels_off"))
        # Some return-to-origin reports overwrite the first takeoff with the
        # successful post-return departure. The earlier airborne leg has no
        # recoverable start, but the continuation remains fully specified.
        if (
            not first_landing_candidates
            and first_diversion["airport"] == origin
            and first_diversion["wheels_off"] == main_wheels_off
        ):
            origin = first_diversion["airport"]
            registration = first_diversion["next_registration"]
            diversions = diversions[1:]
            landing_count -= 1
            if not diversions:
                if not _valid_diversion_leg(
                    takeoff_time,
                    final_landing_time,
                ):
                    return []
                return [
                    flight_row(
                        origin,
                        destination,
                        registration,
                        takeoff_time,
                        final_landing_time,
                    )
                ]

    def solve_anchored(
        diversion_index: int,
        leg_origin: str,
        leg_registration: str,
        leg_takeoff: datetime,
    ) -> list[dict] | None:
        assert final_landing_time is not None
        diversion = diversions[diversion_index]
        landing_deadline = min(
            leg_takeoff + MAX_DIVERSION_LEG_DURATION + timedelta(microseconds=1),
            final_landing_time,
        )
        landings = _utc_clock_candidates(
            diversion["wheels_on"],
            diversion["timezone"],
            after=leg_takeoff,
            before=landing_deadline,
        )
        for landing_time in landings:
            leg = flight_row(
                leg_origin,
                diversion["airport"],
                leg_registration,
                leg_takeoff,
                landing_time,
            )
            takeoffs = _utc_clock_candidates(
                diversion["wheels_off"],
                diversion["timezone"],
                after=landing_time,
                before=final_landing_time,
            )
            for next_takeoff in takeoffs:
                if diversion_index + 1 < landing_count:
                    remaining = solve_anchored(
                        diversion_index + 1,
                        diversion["airport"],
                        diversion["next_registration"],
                        next_takeoff,
                    )
                    if remaining is not None:
                        return [leg, *remaining]
                elif _valid_diversion_leg(
                    next_takeoff,
                    final_landing_time,
                ):
                    return [
                        leg,
                        flight_row(
                            diversion["airport"],
                            destination,
                            diversion["next_registration"],
                            next_takeoff,
                            final_landing_time,
                        ),
                    ]
        return None

    if reached_destination and final_landing_time is not None:
        return solve_anchored(
            0,
            origin,
            registration,
            takeoff_time,
        ) or []

    flights: list[dict] = []
    current_origin = origin
    current_registration = registration
    for index, diversion in enumerate(diversions):
        landing_time = _next_utc_datetime(
            takeoff_time,
            diversion["wheels_on"],
            diversion["timezone"],
        )
        if not _valid_diversion_leg(takeoff_time, landing_time):
            return []
        flights.append(
            flight_row(
                current_origin,
                diversion["airport"],
                current_registration,
                takeoff_time,
                landing_time,
            )
        )
        needs_another_takeoff = index + 1 < landing_count or reached_destination
        if not needs_another_takeoff:
            return flights
        takeoff_time = _next_utc_datetime(
            landing_time,
            diversion["wheels_off"],
            diversion["timezone"],
        )
        current_origin = diversion["airport"]
        current_registration = diversion["next_registration"]

    if reached_destination:
        if final_landing_time is None:
            final_wheels_on = _coerce_hhmm(row.get("wheels_on"))
            final_landing_time = _next_utc_datetime(
                takeoff_time,
                final_wheels_on,
                destination_timezone,
            )
        if not _valid_diversion_leg(takeoff_time, final_landing_time):
            return []
        flights.append(
            flight_row(
                current_origin,
                destination,
                current_registration,
                takeoff_time,
                final_landing_time,
            )
        )
    return flights


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


def airport_timezone_mapping(airport_catalog: AirportCatalogView) -> pl.DataFrame:
    rows = []
    for airport in airport_catalog.airports:
        tz = tf.timezone_at(
            lng=airport.lon,
            lat=airport.lat,
        )
        if tz:
            rows.append({"airport_id": airport.id, "timezone": tz})
    return pl.DataFrame(
        rows,
        schema={"airport_id": pl.Int64, "timezone": pl.String},
    )


def iata_airport_mapping(
    airport_catalog: AirportCatalogView,
    codes: list[str],
) -> pl.DataFrame:
    """Resolve BTS IATA codes, omitting codes absent from the catalog."""
    resolved = []
    for code in sorted(set(codes)):
        try:
            airport = airport_catalog.resolve_code(
                code,
                code_priority=("iata_code",),
            )
        except ValueError:
            airport = None
        if airport is None:
            continue
        resolved.append((code, airport))

    return pl.DataFrame(
        {
            "iata": [code for code, _ in resolved],
            "airport_id": [airport.id for _, airport in resolved],
            "ident": [airport.ident for _, airport in resolved],
        },
        schema={"iata": pl.String, "airport_id": pl.Int64, "ident": pl.String},
    )


def get_registration_to_icao(at_date: date | None = None) -> pl.DataFrame:
    return (
        scan_latest_faa_registration_icao_for_join(
            real_icao_column="icao",
            at_date=at_date,
        )
        .collect()
    )


def _remove_overlapping_flights(
    df: pl.DataFrame,
    target_date: date,
) -> pl.DataFrame:
    overlaps = (
        df.sort(["icao", "takeoff_time"])
        .with_columns(
            pl.col("landing_time").shift(1).over("icao").alias("prev_landing_time")
        )
        .filter(pl.col("takeoff_time") < pl.col("prev_landing_time"))
    )
    print(len(overlaps), "overlapping flights found for", target_date)
    if not overlaps.is_empty():
        overlapping_icaos = (
            overlaps.get_column("icao").drop_nulls().unique().to_list()
        )
        df = df.filter(~pl.col("icao").is_in(overlapping_icaos))
    return df.sort(["icao", "takeoff_time"])


def _incomplete_bts_takeoff_dates(row: dict) -> list[date]:
    """Find UTC days to exclude when a regular flight has incomplete timing.

    Departure delay anchors the actual local day even for multi-day delays.
    Without it, cover both the reported local day and an overnight departure.
    """
    local_date = row["fl_date"]
    wheels_off = _coerce_hhmm(row.get("wheels_off"))
    scheduled_time = _coerce_hhmm(row.get("crs_dep_time"))
    has_departure_anchor = (
        scheduled_time is not None and _has_value(row.get("dep_delay"))
    )
    timezone_name = row.get("origin_timezone")
    if not timezone_name:
        if has_departure_anchor:
            if wheels_off is not None:
                local_date = _diverted_takeoff_datetime(row, "UTC").date()
            else:
                local_date = (
                    _local_datetime(local_date, scheduled_time, "UTC")
                    + timedelta(minutes=float(row["dep_delay"]))
                ).date()
        # Unknown origin timezone: cover the UTC extent of both local days.
        return [local_date + timedelta(days=offset) for offset in (-1, 0, 1, 2)]

    if wheels_off is not None and has_departure_anchor:
        return [_diverted_takeoff_datetime(row, timezone_name).date()]
    if wheels_off is not None:
        return sorted({
            _local_datetime(local_date + timedelta(days=offset), wheels_off, timezone_name)
            .astimezone(UTC).date()
            for offset in (0, 1)
        })

    if has_departure_anchor:
        start = _local_datetime(local_date, scheduled_time, timezone_name).astimezone(UTC)
        start += timedelta(minutes=float(row["dep_delay"]))
        end = start + timedelta(days=1)
    else:
        start = _local_datetime(local_date, 0, timezone_name).astimezone(UTC)
        end = _local_datetime(local_date + timedelta(days=2), 0, timezone_name).astimezone(UTC)
        end -= timedelta(microseconds=1)
    return [
        start.date() + timedelta(days=offset)
        for offset in range((end.date() - start.date()).days + 1)
    ]


def _incomplete_bts_aircraft_days(
    incomplete_rows: pl.DataFrame,
    registration_mapping: pl.DataFrame,
    airport_mapping: pl.DataFrame,
    timezone_mapping: pl.DataFrame,
) -> pl.DataFrame:
    # Identify aircraft before airport joins can discard incomplete reports.
    origins = airport_mapping.join(timezone_mapping, on="airport_id").select(
        pl.col("iata").alias("origin"),
        pl.col("timezone").alias("origin_timezone"),
    )
    mapped_rows = incomplete_rows.join(registration_mapping, on="registration").join(
        origins, on="origin", how="left"
    )
    return pl.DataFrame(
        [
            {"icao": row["icao"].lower(), "_excluded_date": affected_date}
            for row in mapped_rows.drop_nulls("icao").iter_rows(named=True)
            for affected_date in _incomplete_bts_takeoff_dates(row)
        ],
        schema={"icao": pl.String, "_excluded_date": pl.Date},
    ).unique()


def _bts_month_paths(year: int, month: int) -> list[Path]:
    """Return source files that can contribute to a UTC takeoff month."""
    month_start = date(year, month, 1)
    previous_month = month_start - timedelta(days=1)
    next_month = (
        date(year + 1, 1, 1)
        if month == 12
        else date(year, month + 1, 1)
    )
    return [
        OUTPUT_DIR
        / "data"
        / "raw"
        / "bts"
        / "v2"
        / f"year={source_month.year}"
        / f"month={source_month.month:02d}"
        / "T_ONTIME_REPORTING.csv"
        for source_month in (previous_month, month_start, next_month)
    ]


def _select_bts_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Select v2 inputs while tolerating pre-diversion test fixtures."""
    missing_diversion_columns = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in DIVERSION_COLUMN_DTYPES.items()
        if column not in df.columns
    ]
    if missing_diversion_columns:
        df = df.with_columns(missing_diversion_columns)
    return df.select([*BASE_COLUMNS, *DIVERSION_COLUMNS]).with_columns(
        pl.col(column).cast(dtype, strict=False)
        for column, dtype in DIVERSION_COLUMN_DTYPES.items()
    )


def _join_bts_identifiers(
    df: pl.DataFrame,
    registration_mapping: pl.DataFrame,
    airport_mapping: pl.DataFrame,
) -> pl.DataFrame:
    df = df.join(registration_mapping, on="registration", how="inner")
    df = df.join(
        airport_mapping,
        left_on="origin",
        right_on="iata",
        how="inner",
    ).rename(
        {
            "airport_id": "takeoff_airport_id",
            "ident": "takeoff_airport_ident",
        }
    )
    return df.join(
        airport_mapping,
        left_on="dest",
        right_on="iata",
        how="inner",
    ).rename(
        {
            "airport_id": "landing_airport_id",
            "ident": "landing_airport_ident",
        }
    )


def _to_bts_flight_schema(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(False).alias("pia"),
        pl.lit(False).alias("ladd"),
        pl.lit(False).alias("military"),
        pl.lit(False).alias("interesting"),
        pl.lit(None).cast(pl.String).alias("aircraft_type"),
        pl.lit(None).cast(pl.String).alias("owner"),
        pl.lit(None).cast(pl.String).alias("aircraft_description"),
        pl.lit("A3").cast(pl.String).alias("category"),
    )
    return with_flight_schema_columns(df).with_columns(
        pl.col("icao").str.to_lowercase()
    )


def _create_flights_from_bts_month(
    year: int,
    month: int,
    airport_catalog: AirportCatalogView,
    openairframes_date: date | None = None,
) -> pl.DataFrame:
    '''
    On 2025, 10 for example some flights from UTC day 2025-11-01 and 2025-09-30 are included
    '''
    dfs = []
    # A BTS source month can contain computed UTC takeoffs in either adjacent
    # month, so load both sides before assigning flights by takeoff_time.
    for path in _bts_month_paths(year, month):
        if path.exists():
            df_tmp = pl.read_csv(path)
            df_tmp = df_tmp.rename({c: c.lower() for c in df_tmp.columns})
            df_tmp = _select_bts_columns(df_tmp)
            dfs.append(df_tmp)
    df = pl.concat(dfs, how="vertical_relaxed").unique()
    df = df.with_columns(
        pl.col("fl_date")
        .str.to_datetime(format="%m/%d/%Y %I:%M:%S %p", strict=False)
        .dt.strftime("%Y-%m-%d")
        .str.to_date(format="%Y-%m-%d")
        .alias("fl_date"),
        _bts_callsign_expr(),
    )

    registration_mapping = get_registration_to_icao(
        at_date=openairframes_date or date(year, month, 1)
    )
    airport_code_columns = [
        "origin",
        "dest",
        *[f"div{index}_airport" for index in range(1, DIVERSION_COUNT + 1)],
    ]
    airport_codes = pl.concat(
        [df.get_column(column) for column in airport_code_columns]
    ).drop_nulls().unique().to_list()
    airport_mapping = iata_airport_mapping(airport_catalog, airport_codes)
    timezone_mapping = airport_timezone_mapping(airport_catalog)

    diversion_report = _diversion_report_expr()
    regular_df = df.filter(
        (pl.col("cancelled") == 0) & ~diversion_report
    ).rename({"tail_num": "registration"})
    excluded_days = _incomplete_bts_aircraft_days(
        regular_df.filter(pl.any_horizontal(
            pl.col(field).is_null() for field in BTS_REQUIRED_TIMING_FIELDS
        )),
        registration_mapping,
        airport_mapping,
        timezone_mapping,
    )
    regular_df = regular_df.drop_nulls(BTS_REQUIRED_TIMING_FIELDS)
    regular_df = _join_bts_identifiers(
        regular_df,
        registration_mapping,
        airport_mapping,
    )
    regular_df = regular_df.join(
        timezone_mapping,
        left_on="takeoff_airport_id",
        right_on="airport_id",
        how="inner",
    ).rename({"timezone": "origin_timezone"})
    regular_df = regular_df.join(
        timezone_mapping,
        left_on="landing_airport_id",
        right_on="airport_id",
        how="inner",
    ).rename({"timezone": "dest_timezone"})
    regular_df = (
        regular_df.with_columns(
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
        .rename(
            {
                "time_takeoff": "takeoff_time",
                "time_landing": "landing_time",
            }
        )
    )
    regular_df = _to_bts_flight_schema(regular_df)

    airport_timezones = dict(
        airport_mapping.join(
            timezone_mapping,
            on="airport_id",
            how="inner",
        )
        .select("iata", "timezone")
        .iter_rows()
    )
    diverted_rows = [
        flight
        for row in df.filter(diversion_report).iter_rows(named=True)
        for flight in _diverted_flight_rows(row, airport_timezones)
    ]
    diverted_df = pl.DataFrame(
        diverted_rows,
        schema={
            "registration": pl.String,
            "origin": pl.String,
            "dest": pl.String,
            "takeoff_time": pl.Datetime("us"),
            "landing_time": pl.Datetime("us"),
            "callsign": pl.String,
        },
    )
    diverted_df = _join_bts_identifiers(
        diverted_df,
        registration_mapping,
        airport_mapping,
    )
    diverted_df = _to_bts_flight_schema(diverted_df)
    return (
        pl.concat([regular_df, diverted_df])
        .with_columns(pl.col("takeoff_time").dt.date().alias("_excluded_date"))
        .join(excluded_days, on=["icao", "_excluded_date"], how="anti")
        .drop("_excluded_date")
    )


def create_flights_from_bts(
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Create the finalized BTS flights dataframe for one UTC takeoff day."""
    monthly_df = _create_flights_from_bts_month(
        target_date.year,
        target_date.month,
        airport_catalog,
        openairframes_date=target_date,
    )
    daily_df = monthly_df.filter(
        pl.col("takeoff_time").dt.date() == target_date
    )
    return _remove_overlapping_flights(daily_df, target_date)


@dataclass(frozen=True)
class BTSFlightSource(NonADSBFlightSource):
    name: str = "bts"
    version: int = 15

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_flights_from_bts(dt, airport_catalog)


if __name__ == "__main__":
    from airports.runtime import airport_catalog_for_date

    target_date = date(2026, 3, 1)
    df = create_flights_from_bts(
        target_date,
        airport_catalog_for_date(target_date),
    )
    print(df)
