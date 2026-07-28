"""OpenSky-specific state-vector cleanup.

OpenSky state vectors aggregate observations from many receivers and can
contain isolated position, altitude, and ground-state glitches that are not
present in readsb trace archives. Keep these source corrections outside the
flight segmentation algorithm so every downstream OpenSky consumer sees the
same normalized trace.
"""

from __future__ import annotations

import math

import polars as pl


MIN_VALID_BARO_ALTITUDE_FT = -2_000.0
MAX_VALID_BARO_ALTITUDE_FT = 60_000.0
MAX_SPIKE_NEIGHBOR_GAP_SECONDS = 30.0
MIN_UNOBSERVED_STOP_GAP_SECONDS = 45.0 * 60.0
UNOBSERVED_STOP_TREND_WINDOW = "45m"
MIN_UNOBSERVED_STOP_ALTITUDE_TREND_FT = 2_000.0
POSITION_ERROR_ALLOWANCE_KM = 2.0
SPEED_ERROR_ALLOWANCE_KT = 200.0
MISSING_SPEED_ALLOWANCE_KT = 600.0
ISOLATED_ALTITUDE_SPIKE_FT = 5_000.0
NEIGHBOR_ALTITUDE_AGREEMENT_FT = 2_000.0
KM_PER_DEGREE_LATITUDE = 111.195
KM_PER_NAUTICAL_MILE = 1.852

_PREFIX = "_opensky_preprocess_"
OPENSKY_PREPROCESSING_COLUMNS = (
    "time",
    "icao",
    "lat",
    "lon",
    "ground_speed_kt",
    "on_ground",
    "baro_altitude_ft",
    "callsign",
)


def _neighbor(column: str, offset: int) -> pl.Expr:
    return pl.col(column).shift(offset).over("icao")


def _seconds_between(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
    return (later - earlier).dt.total_milliseconds() / 1_000


def _distance_km(
    lat_0: pl.Expr,
    lon_0: pl.Expr,
    lat_1: pl.Expr,
    lon_1: pl.Expr,
) -> pl.Expr:
    """Fast local-distance approximation with antimeridian handling."""
    delta_lat_km = (lat_1 - lat_0) * KM_PER_DEGREE_LATITUDE
    delta_lon_deg = ((lon_1 - lon_0 + 180.0) % 360.0) - 180.0
    mean_lat_rad = ((lat_0 + lat_1) / 2.0) * (math.pi / 180.0)
    delta_lon_km = (
        delta_lon_deg
        * KM_PER_DEGREE_LATITUDE
        * mean_lat_rad.cos()
    )
    return (delta_lat_km.pow(2) + delta_lon_km.pow(2)).sqrt()


def _speed_allowance(speed_0: pl.Expr, speed_1: pl.Expr) -> pl.Expr:
    return (
        pl.when(speed_0.is_null() & speed_1.is_null())
        .then(pl.lit(MISSING_SPEED_ALLOWANCE_KT))
        .otherwise(
            pl.max_horizontal(
                speed_0.fill_null(0.0),
                speed_1.fill_null(0.0),
            )
        )
        + SPEED_ERROR_ALLOWANCE_KT
    )


def _allowed_distance_km(
    elapsed_seconds: pl.Expr,
    speed_0: pl.Expr,
    speed_1: pl.Expr,
) -> pl.Expr:
    return (
        POSITION_ERROR_ALLOWANCE_KM
        + _speed_allowance(speed_0, speed_1)
        * KM_PER_NAUTICAL_MILE
        * elapsed_seconds
        / 3_600.0
    )


def _remove_isolated_position_spikes(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        _neighbor("time", 1).alias(f"{_PREFIX}previous_time"),
        _neighbor("time", -1).alias(f"{_PREFIX}next_time"),
        _neighbor("lat", 1).alias(f"{_PREFIX}previous_lat"),
        _neighbor("lat", -1).alias(f"{_PREFIX}next_lat"),
        _neighbor("lon", 1).alias(f"{_PREFIX}previous_lon"),
        _neighbor("lon", -1).alias(f"{_PREFIX}next_lon"),
        _neighbor("ground_speed_kt", 1).alias(f"{_PREFIX}previous_speed"),
        _neighbor("ground_speed_kt", -1).alias(f"{_PREFIX}next_speed"),
    )

    previous_elapsed = _seconds_between(
        pl.col("time"),
        pl.col(f"{_PREFIX}previous_time"),
    )
    next_elapsed = _seconds_between(
        pl.col(f"{_PREFIX}next_time"),
        pl.col("time"),
    )
    skip_elapsed = _seconds_between(
        pl.col(f"{_PREFIX}next_time"),
        pl.col(f"{_PREFIX}previous_time"),
    )
    previous_distance = _distance_km(
        pl.col(f"{_PREFIX}previous_lat"),
        pl.col(f"{_PREFIX}previous_lon"),
        pl.col("lat"),
        pl.col("lon"),
    )
    next_distance = _distance_km(
        pl.col("lat"),
        pl.col("lon"),
        pl.col(f"{_PREFIX}next_lat"),
        pl.col(f"{_PREFIX}next_lon"),
    )
    skip_distance = _distance_km(
        pl.col(f"{_PREFIX}previous_lat"),
        pl.col(f"{_PREFIX}previous_lon"),
        pl.col(f"{_PREFIX}next_lat"),
        pl.col(f"{_PREFIX}next_lon"),
    )

    previous_transition_is_impossible = previous_distance > _allowed_distance_km(
        previous_elapsed,
        pl.col(f"{_PREFIX}previous_speed"),
        pl.col("ground_speed_kt"),
    )
    next_transition_is_impossible = next_distance > _allowed_distance_km(
        next_elapsed,
        pl.col("ground_speed_kt"),
        pl.col(f"{_PREFIX}next_speed"),
    )
    neighbors_form_plausible_transition = skip_distance <= _allowed_distance_km(
        skip_elapsed,
        pl.col(f"{_PREFIX}previous_speed"),
        pl.col(f"{_PREFIX}next_speed"),
    )
    has_nearby_neighbors = (
        previous_elapsed.is_between(
            0.0,
            MAX_SPIKE_NEIGHBOR_GAP_SECONDS,
            closed="right",
        )
        & next_elapsed.is_between(
            0.0,
            MAX_SPIKE_NEIGHBOR_GAP_SECONDS,
            closed="right",
        )
    )
    isolated_spike = (
        has_nearby_neighbors
        & previous_transition_is_impossible
        & next_transition_is_impossible
        & neighbors_form_plausible_transition
    )
    return df.filter(~isolated_spike.fill_null(False)).drop(
        pl.selectors.starts_with(_PREFIX)
    )


def _clean_altitude_and_ground_state(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        pl.when(
            pl.col("baro_altitude_ft").is_between(
                MIN_VALID_BARO_ALTITUDE_FT,
                MAX_VALID_BARO_ALTITUDE_FT,
                closed="both",
            )
        )
        .then(pl.col("baro_altitude_ft"))
        .otherwise(None)
        .alias("baro_altitude_ft")
    )
    df = df.with_columns(
        _neighbor("time", 1).alias(f"{_PREFIX}previous_time"),
        _neighbor("time", -1).alias(f"{_PREFIX}next_time"),
        _neighbor("baro_altitude_ft", 1).alias(f"{_PREFIX}previous_altitude"),
        _neighbor("baro_altitude_ft", -1).alias(f"{_PREFIX}next_altitude"),
        _neighbor("on_ground", 1).alias(f"{_PREFIX}previous_on_ground"),
        _neighbor("on_ground", -1).alias(f"{_PREFIX}next_on_ground"),
    )

    previous_elapsed = _seconds_between(
        pl.col("time"),
        pl.col(f"{_PREFIX}previous_time"),
    )
    next_elapsed = _seconds_between(
        pl.col(f"{_PREFIX}next_time"),
        pl.col("time"),
    )
    has_nearby_neighbors = (
        previous_elapsed.is_between(
            0.0,
            MAX_SPIKE_NEIGHBOR_GAP_SECONDS,
            closed="right",
        )
        & next_elapsed.is_between(
            0.0,
            MAX_SPIKE_NEIGHBOR_GAP_SECONDS,
            closed="right",
        )
    )

    previous_altitude = pl.col(f"{_PREFIX}previous_altitude")
    next_altitude = pl.col(f"{_PREFIX}next_altitude")
    isolated_altitude_spike = (
        has_nearby_neighbors
        & pl.col("baro_altitude_ft").is_not_null()
        & previous_altitude.is_not_null()
        & next_altitude.is_not_null()
        & (
            (pl.col("baro_altitude_ft") - previous_altitude).abs()
            >= ISOLATED_ALTITUDE_SPIKE_FT
        )
        & (
            (pl.col("baro_altitude_ft") - next_altitude).abs()
            >= ISOLATED_ALTITUDE_SPIKE_FT
        )
        & (
            (previous_altitude - next_altitude).abs()
            <= NEIGHBOR_ALTITUDE_AGREEMENT_FT
        )
    )

    previous_on_ground = pl.col(f"{_PREFIX}previous_on_ground")
    next_on_ground = pl.col(f"{_PREFIX}next_on_ground")
    isolated_ground_flip = (
        has_nearby_neighbors
        & pl.col("on_ground").is_not_null()
        & previous_on_ground.is_not_null()
        & next_on_ground.is_not_null()
        & (previous_on_ground == next_on_ground)
        & (pl.col("on_ground") != previous_on_ground)
    )

    return (
        df.with_columns(
            pl.when(isolated_altitude_spike)
            .then(None)
            .otherwise(pl.col("baro_altitude_ft"))
            .alias("baro_altitude_ft"),
            pl.when(isolated_ground_flip)
            .then(previous_on_ground)
            .otherwise(pl.col("on_ground"))
            .alias("on_ground"),
        )
        .drop(pl.selectors.starts_with(_PREFIX))
    )


def _expose_unobserved_stops_to_readsb(df: pl.LazyFrame) -> pl.LazyFrame:
    """Expose unobserved stops to readsb's native gap rule.

    OpenSky can miss an entire descent, ground dwell, and initial climb.
    readsb may then lack the state transitions needed to recognize a leg
    boundary, particularly when reception resumes at high altitude. A
    callsign change or descent/climb trend distinguishes these hidden stops
    from ordinary receiver outages. Mark only the first resumed point as an
    inferred ground/missing-altitude observation so readsb can recognize the
    gap without dropping or moving it.
    """
    recent_max_column = f"{_PREFIX}recent_max_altitude"
    future_max_column = f"{_PREFIX}future_max_altitude"
    reverse_time_column = f"{_PREFIX}reverse_time"

    df = df.with_columns(
        pl.col("baro_altitude_ft")
        .rolling_max_by(
            "time",
            window_size=UNOBSERVED_STOP_TREND_WINDOW,
            min_samples=1,
        )
        .over("icao")
        .alias(recent_max_column),
        pl.from_epoch(
            -pl.col("time").dt.epoch("ms"),
            time_unit="ms",
        ).alias(reverse_time_column),
    )
    df = (
        df.sort(["icao", reverse_time_column])
        .with_columns(
            pl.col("baro_altitude_ft")
            .rolling_max_by(
                reverse_time_column,
                window_size=UNOBSERVED_STOP_TREND_WINDOW,
                min_samples=1,
            )
            .over("icao")
            .alias(future_max_column)
        )
        .sort(["icao", "time"])
    )

    previous_time = _neighbor("time", 1)
    previous_on_ground = _neighbor("on_ground", 1)
    previous_altitude = _neighbor("baro_altitude_ft", 1)
    previous_recent_max = _neighbor(recent_max_column, 1)
    previous_callsign = (
        _neighbor("callsign", 1)
        .fill_null("")
        .str.replace_all(" ", "")
    )
    callsign = (
        pl.col("callsign")
        .fill_null("")
        .str.replace_all(" ", "")
    )
    gap_seconds = _seconds_between(pl.col("time"), previous_time)
    callsign_changed = (
        (previous_callsign != "")
        & (callsign != "")
        & (previous_callsign != callsign)
    )
    descended_before_gap = (
        previous_recent_max - previous_altitude
        >= MIN_UNOBSERVED_STOP_ALTITUDE_TREND_FT
    )
    climbed_after_gap = (
        pl.col(future_max_column) - pl.col("baro_altitude_ft")
        >= MIN_UNOBSERVED_STOP_ALTITUDE_TREND_FT
    )
    has_stop_evidence = (
        callsign_changed
        | (descended_before_gap & climbed_after_gap)
    )
    unobserved_stop = (
        (gap_seconds >= MIN_UNOBSERVED_STOP_GAP_SECONDS)
        & previous_on_ground.not_()
        & pl.col("on_ground").not_()
        & has_stop_evidence
    )
    return (
        df.with_columns(
            pl.when(unobserved_stop.fill_null(False))
            .then(None)
            .otherwise(pl.col("baro_altitude_ft"))
            .alias("baro_altitude_ft"),
            pl.when(unobserved_stop.fill_null(False))
            .then(True)
            .otherwise(pl.col("on_ground"))
            .alias("on_ground"),
        )
        .drop(pl.selectors.starts_with(_PREFIX))
    )


def preprocess_opensky_adsb(df: pl.LazyFrame) -> pl.LazyFrame:
    """Clean normalized OpenSky rows without applying flight-specific logic."""
    schema = set(df.collect_schema().names())
    required = set(OPENSKY_PREPROCESSING_COLUMNS)
    missing = sorted(required - schema)
    if missing:
        raise ValueError(
            "OpenSky preprocessing requires columns: "
            + ", ".join(missing)
        )

    df = (
        df.filter(
            pl.col("lat").is_between(-90.0, 90.0, closed="both")
            & pl.col("lon").is_between(-180.0, 180.0, closed="both")
        )
        .sort(["icao", "time"])
    )
    df = _remove_isolated_position_spikes(df)
    df = _clean_altitude_and_ground_state(df)
    return _expose_unobserved_stops_to_readsb(df)
