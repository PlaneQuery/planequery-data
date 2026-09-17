"""Bounded, candidate-relative endpoint trajectories, identical at train/inference."""
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from functools import cached_property
import numpy as np
import polars as pl
from airports.airport_lookup import Airport, AirportLookup
from data_engineering.adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import Flight, flights_df_to_flights

FEATURE_VERSION = 2
FEATURE_PERIOD = timedelta(minutes=10)
MAX_POINTS = 32
MAX_CANDIDATES = 32
CANDIDATE_RADIUS_KM = 200


class FeatureName(StrEnum):
    NORTH_KM = 'north_km'
    EAST_KM = 'east_km'
    RADIAL_SPEED_KM_S = 'radial_speed_km_s'
    TIME_DELTA_S = 'time_delta_s'
    DISTANCE = 'log_distance_km'
    ALTITUDE = 'altitude_above_airport'
    SPEED = 'ground_speed'
    VERTICAL_RATE = 'vertical_rate'
    HEADING_SIN = 'heading_sin'
    HEADING_COS = 'heading_cos'
    ELAPSED = 'elapsed'
    ON_GROUND = 'on_ground'
    ELEVATION = 'elevation'
    SMALL = 'small_airport'
    MEDIUM = 'medium_airport'
    LARGE = 'large_airport'
    SCHEDULED = 'scheduled_service'
    LANDING = 'landing'
    ALT_MISSING = 'altitude_missing'
    SPEED_MISSING = 'speed_missing'
    TRACK_MISSING = 'track_missing'
    RATE_MISSING = 'rate_missing'
    ELEVATION_MISSING = 'elevation_missing'
    COVERAGE_GAP = 'coverage_gap'


@dataclass(frozen=True)
class FeatureConfig:
    enabled: tuple[FeatureName, ...]

    def __post_init__(self):
        if not self.enabled or len(set(self.enabled)) != len(self.enabled):
            raise ValueError('Features must be nonempty and unique')

    @property
    def input_size(self):
        return len(self.enabled)

    @property
    def column_indices(self):
        return [tuple(FeatureName).index(name) for name in self.enabled]


ALL_FEATURE_CONFIG = FeatureConfig(tuple(FeatureName))


@dataclass
class AirportCandidateFeatures:
    flight_id: str
    predicted_airport: Airport
    true_airport: Airport | None
    features: np.ndarray
    endpoint: str = 'takeoff'

    def features_to_numpy(self, max_length, feature_config):
        values = np.asarray(self.features, dtype=np.float32)[:, feature_config.column_indices]
        x = np.zeros((max_length, feature_config.input_size), dtype=np.float32)
        mask = np.arange(max_length) < len(values)
        x[:len(values)] = values
        return x, mask, np.float32(self.predicted_airport == self.true_airport)


@dataclass
class FlightFeatures:
    flight_id: str
    true_airport: Airport | None
    candidates: list[AirportCandidateFeatures]
    endpoint: str = 'takeoff'


@dataclass
class PlaneFeatures:
    icao: str
    flight_features: list[FlightFeatures]


def _valid(value):
    return value is not None and np.isfinite(value)


def create_features_for_flight(f: Flight, messages: list[AdsbMessage],
                               airport_look_radius_km=CANDIDATE_RADIUS_KM,
                               endpoint='takeoff'):
    if endpoint not in ('takeoff', 'landing'):
        raise ValueError('endpoint must be takeoff or landing')
    lookup = AirportLookup()
    truth = lookup.ident_index.get(getattr(f, endpoint + '_airport_ident'))
    result = FlightFeatures(f.flight_id, truth, [], endpoint)
    # Never consume a neighboring segment, even when this flight is very short.
    start, end = f.takeoff_time, f.landing_time
    if start is None or end is None or end < start:
        return result
    landing = endpoint == 'landing'
    boundary = end if landing else start
    selected = [m for m in messages if start <= m.time <= end
                and abs((m.time - boundary).total_seconds()) <= FEATURE_PERIOD.total_seconds()
                and _valid(m.lat) and _valid(m.lon) and abs(m.lat) <= 90 and abs(m.lon) <= 180]
    # Endpoint first for both directions; preserve near-endpoint detail.
    selected.sort(key=lambda m: m.time, reverse=landing)
    selected = list({m.time: m for m in selected}.values())
    if not selected:
        return result
    if len(selected) > MAX_POINTS:
        indices = np.unique(np.rint(np.linspace(0, np.sqrt(len(selected)-1), MAX_POINTS)**2).astype(int))
        selected = [selected[i] for i in indices]
    first = selected[0]
    candidates = lookup.getAirportsWithinRadius(first.lat, first.lon, airport_look_radius_km)
    candidates.sort(key=lambda a: (a.distance_to(first.lat, first.lon), a.ident))
    # Reserve space for larger fields obscured by numerous nearby private strips.
    nearby = candidates[:24]
    larger = [a for a in candidates if a.type != 'small_airport' and a not in nearby][:8]
    candidates = (nearby + larger + [a for a in candidates[24:] if a not in larger])[:MAX_CANDIDATES]
    if not candidates:
        return result
    lat = np.radians([m.lat for m in selected])[None, :]
    lon = np.radians([m.lon for m in selected])[None, :]
    alat = np.radians([a.lat for a in candidates])[:, None]
    alon = np.radians([a.lon for a in candidates])[:, None]
    dlon = (lon - alon + np.pi) % (2*np.pi) - np.pi
    h = np.sin((lat-alat)/2)**2 + np.cos(lat)*np.cos(alat)*np.sin(dlon/2)**2
    distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    bearing = np.arctan2(np.sin(dlon)*np.cos(lat), np.cos(alat)*np.sin(lat)-np.sin(alat)*np.cos(lat)*np.cos(dlon))
    elapsed = np.array([abs((m.time-first.time).total_seconds()) for m in selected])
    dt = np.diff(elapsed, prepend=0)
    radial = np.diff(distance, axis=1, prepend=distance[:, :1]) / np.maximum(dt, 1)
    # Radial derivative refers to physical time, even for reversed landing tracks.
    radial *= -1 if landing else 1
    def field(name):
        return np.array([getattr(m, name) if _valid(getattr(m, name)) else 0 for m in selected])
    def missing(name):
        return np.array([not _valid(getattr(m, name)) for m in selected], dtype=float)
    elevation = np.array([a.elevation_ft or 0 for a in candidates])[:, None]
    track = np.radians(field('track_deg'))[None, :]
    altitude = field('baro_altitude_ft')[None, :] - elevation
    altitude[:, missing('baro_altitude_ft').astype(bool)] = 0
    shape = distance.shape
    columns = [distance*np.cos(bearing)/100, distance*np.sin(bearing)/100,
               radial/0.2, np.log1p(dt)/5, np.log1p(distance)/5,
               altitude/10000, field('ground_speed_kt')/300,
               field('baro_rate_fpm')/2000, np.sin(track-bearing), np.cos(track-bearing),
               elapsed/600, field('on_ground'), elevation/5000,
               np.array([a.type == 'small_airport' for a in candidates])[:, None],
               np.array([a.type == 'medium_airport' for a in candidates])[:, None],
               np.array([a.type == 'large_airport' for a in candidates])[:, None],
               np.array([a.scheduled_service == 'yes' for a in candidates])[:, None],
               float(landing), missing('baro_altitude_ft'), missing('ground_speed_kt'),
               missing('track_deg'), missing('baro_rate_fpm'),
               np.array([a.elevation_ft is None for a in candidates])[:, None],
               abs((first.time-boundary).total_seconds())/600]
    values = np.stack([np.broadcast_to(c, shape) for c in columns], axis=-1)
    values = np.nan_to_num(values, nan=0, posinf=10, neginf=-10).clip(-10, 10).astype(np.float32)
    result.candidates = [AirportCandidateFeatures(f.flight_id, a, truth, x, endpoint)
                         for a, x in zip(candidates, values)]
    return result


def create_features_for_plane(flights, messages):
    messages = sorted(messages, key=lambda m: m.time)
    # Binary search avoids scanning the full aircraft day for every flight.
    from bisect import bisect_left, bisect_right
    times = [m.time for m in messages]
    endpoints = []
    for flight in flights:
        segment = messages[bisect_left(times, flight.takeoff_time):bisect_right(times, flight.landing_time)]
        for endpoint in ('takeoff', 'landing'):
            endpoints.append(create_features_for_flight(flight, segment, endpoint=endpoint))
    return PlaneFeatures(flights[0].icao if flights else '', endpoints)


class Features:
    def __init__(self, df_flights, df_adsb):
        self.flights_by_id = {}
        self.plane_features = []
        adsb = df_adsb.partition_by('icao', as_dict=True)
        for key, frame in df_flights.partition_by('icao', as_dict=True).items():
            flights = flights_df_to_flights(frame)
            for flight in flights:
                if flight.flight_id in self.flights_by_id:
                    raise ValueError(f'Duplicate flight ID: {flight.flight_id}')
                self.flights_by_id[flight.flight_id] = flight
            messages = adsb_messages_from_parquet_df(adsb[key]) if key in adsb else []
            self.plane_features.append(create_features_for_plane(flights, messages))

    @property
    def endpoints(self):
        return [f for p in self.plane_features for f in p.flight_features]

    @cached_property
    def max_messages_length(self):
        return max((len(c.features) for f in self.endpoints for c in f.candidates), default=1)

    def numpy_features(self, feature_config=ALL_FEATURE_CONFIG):
        rows = [c for f in self.endpoints for c in f.candidates]
        if not rows:
            return (np.empty((0, 1, feature_config.input_size), np.float32),
                    np.empty((0, 1), bool), np.empty(0, np.float32), rows)
        x, mask, y = zip(*(c.features_to_numpy(self.max_messages_length, feature_config) for c in rows))
        return np.asarray(x), np.asarray(mask), np.asarray(y), rows


def create_features(df_flights, df_adsb):
    return Features(df_flights, df_adsb)


def create_features_for_day(target_dt: date):
    from flights_airport_nn.features.ground_truth_join_segmentation_algorithm import flights_features_df
    flights = flights_features_df(target_dt)
    columns = ['time', 'icao', 'lat', 'lon', 'baro_altitude_ft', 'ground_speed_kt',
               'track_deg', 'baro_rate_fpm', 'on_ground']
    adsb = read_adsb(target_dt, icaos=flights['icao'].unique().to_list(), columns=columns)
    return create_features(flights, adsb)
