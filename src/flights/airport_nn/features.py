"""Airport-relative trajectory features shared by training and inference.

No aircraft identity, absolute coordinates, dates or target labels enter the NN.
Only observations inside the supplied segment and ten-minute endpoint window
are used. Airports compete using the same observed trajectory.
"""

from dataclasses import dataclass
import numpy as np
import polars as pl

from airports.catalog import AirportCatalogView

VERSION = 3
POINTS, CANDIDATES, INPUT_SIZE = 32, 32, 24
WINDOW_MS = 600_000
COLUMNS = [
    "icao",
    "time",
    "lat",
    "lon",
    "baro_altitude_ft",
    "ground_speed_kt",
    "track_deg",
    "baro_rate_fpm",
    "on_ground",
]
# Feature positions used by the model and the training augmentations.
DISTANCE, ALTITUDE, ELAPSED, GAP = 4, 5, 10, 23


@dataclass
class Endpoint:
    row: int
    endpoint: str
    candidate_ids: list[int]
    candidate_idents: list[str]
    values: np.ndarray  # [candidate, time, feature]
    nearest_id: int | None = None
    nearest_ident: str = ""


def clean_messages(messages):
    return (
        messages.filter(
            pl.col("time").is_not_null()
            & pl.col("lat").is_finite()
            & pl.col("lon").is_finite()
            & (pl.col("lat").abs() <= 90)
            & (pl.col("lon").abs() <= 180)
        )
        .unique(subset=["icao", "time"])
        .sort(["icao", "time"])
    )


def _candidate_airports(matcher, lat, lon):
    if hasattr(matcher, "airports_within_radius"):
        airport_distances = matcher.airports_within_radius(lat, lon, 200)
        airport_distances = sorted(
            airport_distances,
            key=lambda item: (item[1], item[0].ident),
        )
        airports = [airport for airport, _ in airport_distances]
    else:
        # Preserve compatibility with small matchers used by downstream users/tests.
        matches = sorted(
            matcher.within_radius(lat, lon, 200),
            key=lambda match: (match.distance_km, match.ident),
        )
        airports = [match.airport for match in matches]

    # Reserve eight slots for larger airports behind dense private strips.
    nearby = airports[:24]
    larger = [a for a in airports[24:] if a.type != "small_airport"][:8]
    larger_ids = {a.id for a in larger}
    return (
        nearby
        + larger
        + [a for a in airports[24:] if a.id not in larger_ids]
    )[:CANDIDATES]


def trajectories(
    flights,
    messages,
    airport_catalog: AirportCatalogView,
    loss_seconds=0,
    include_nearest=True,
):
    """Yield two endpoints per flight, including those without observations."""
    if loss_seconds < 0:
        raise ValueError("Reception loss must be nonnegative")
    tracks = {}
    cleaned = clean_messages(messages.select(COLUMNS))
    for (icao,), frame in cleaned.group_by("icao", maintain_order=True):
        tracks[icao] = frame.select(
            pl.col("time").dt.epoch("ms"),
            *[pl.col(c).cast(pl.Float64) for c in COLUMNS[2:]],
        ).to_numpy()
    for row, (icao, takeoff_time, landing_time) in enumerate(
        flights.select("icao", "takeoff_time", "landing_time").iter_rows()
    ):
        start = np.datetime64(takeoff_time, "ms").astype(np.int64)
        end = np.datetime64(landing_time, "ms").astype(np.int64)
        track = tracks.get(icao)
        for landing in (False, True):
            name = "landing" if landing else "takeoff"
            boundary = end if landing else start
            lo = max(start + loss_seconds * 1000, boundary - WINDOW_MS)
            hi = min(end - loss_seconds * 1000, boundary + WINDOW_MS)
            if track is None or hi < lo:
                points = np.empty((0, 9))
            else:
                points = track[
                    np.searchsorted(track[:, 0], lo) : np.searchsorted(
                        track[:, 0], hi, side="right"
                    )
                ]
            if landing:
                points = points[::-1]
            if not len(points):
                yield Endpoint(
                    row=row,
                    endpoint=name,
                    candidate_ids=[],
                    candidate_idents=[],
                    values=np.empty((0, 0, INPUT_SIZE), np.float32),
                )
                continue
            if len(points) > POINTS:
                indices = np.unique(
                    np.rint(
                        np.linspace(0, np.sqrt(len(points) - 1), POINTS) ** 2
                    ).astype(int)
                )
                points = points[indices]
            lat, lon = points[0, 1:3]
            nearest_match = (
                airport_catalog.matcher.nearest(lat, lon)
                if include_nearest
                else None
            )
            airports = _candidate_airports(airport_catalog.matcher, lat, lon)
            values = relative_features(points, airports, boundary, landing)
            yield Endpoint(
                row=row,
                endpoint=name,
                candidate_ids=[a.id for a in airports],
                candidate_idents=[a.ident for a in airports],
                values=values,
                nearest_id=(nearest_match.id if nearest_match is not None else None),
                nearest_ident=(
                    nearest_match.ident if nearest_match is not None else ""
                ),
            )


def relative_features(points, airports, boundary, landing):
    if not airports:
        return np.empty((0, len(points), INPUT_SIZE), np.float32)
    lat, lon = np.radians(points[:, 1:3]).T
    alat = np.radians([a.lat for a in airports])[:, None]
    alon = np.radians([a.lon for a in airports])[:, None]
    dlon = (lon - alon + np.pi) % (2 * np.pi) - np.pi
    h = (
        np.sin((lat - alat) / 2) ** 2
        + np.cos(lat) * np.cos(alat) * np.sin(dlon / 2) ** 2
    )
    distance = 12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat),
        np.cos(alat) * np.sin(lat) - np.sin(alat) * np.cos(lat) * np.cos(dlon),
    )
    elapsed = np.abs(points[:, 0] - points[0, 0]) / 1000
    dt = np.diff(elapsed, prepend=0)
    radial = (
        np.diff(distance, axis=1, prepend=distance[:, :1])
        / np.maximum(dt, 1)
        * (-1 if landing else 1)
    )
    missing = ~np.isfinite(points)
    p = np.nan_to_num(points)
    elevation = np.array([a.elevation_ft or 0 for a in airports])[:, None]
    altitude = p[:, 3] - elevation
    altitude[:, missing[:, 3]] = 0
    heading = np.radians(p[:, 5]) - bearing
    columns = [
        distance * np.cos(bearing) / 100,
        distance * np.sin(bearing) / 100,
        radial / 0.2,
        np.log1p(dt) / 5,
        np.log1p(distance) / 5,
        altitude / 10000,
        p[:, 4] / 300,
        p[:, 6] / 2000,
        np.sin(heading),
        np.cos(heading),
        elapsed / 600,
        p[:, 7],
        elevation / 5000,
        *[
            np.array([a.type == t for a in airports])[:, None]
            for t in ("small_airport", "medium_airport", "large_airport")
        ],
        np.array([a.scheduled_service == "yes" for a in airports])[:, None],
        float(landing),
        missing[:, 3],
        missing[:, 4],
        missing[:, 5],
        missing[:, 6],
        np.array([a.elevation_ft is None for a in airports])[:, None],
        abs(points[0, 0] - boundary) / WINDOW_MS,
    ]
    return (
        np.nan_to_num(
            np.stack([np.broadcast_to(c, distance.shape) for c in columns], axis=-1)
        )
        .clip(-10, 10)
        .astype(np.float32)
    )


def pack(endpoints):
    values = np.zeros((len(endpoints), CANDIDATES, POINTS, INPUT_SIZE), np.float32)
    mask = np.zeros(values.shape[:3], bool)
    for i, endpoint in enumerate(endpoints):
        c, t = endpoint.values.shape[:2]
        values[i, :c, :t] = endpoint.values
        mask[i, :c, :t] = True
    return values, mask
