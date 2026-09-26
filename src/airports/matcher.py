"""Spatial matching over a flight-date airport catalog view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.neighbors import BallTree

from airports.models import (
    EARTH_RADIUS_KM,
    Airport,
    AirportMatch,
    AirportMatchProvenance,
)


class AirportMatcher:
    def __init__(
        self,
        airports: Sequence[Airport],
        provenance_by_id: Mapping[int, AirportMatchProvenance],
        source_commit_by_id: Mapping[int, str],
    ) -> None:
        self._airports = tuple(airports)
        self._provenance_by_id = provenance_by_id
        self._source_commit_by_id = source_commit_by_id
        coordinates = np.radians(
            [(airport.lat, airport.lon) for airport in self._airports]
        )
        self._tree = BallTree(coordinates, metric="haversine")

    def _match(self, index: int, distance_radians: float) -> AirportMatch:
        airport = self._airports[index]
        return AirportMatch(
            id=airport.id,
            ident=airport.ident,
            distance_km=float(distance_radians * EARTH_RADIUS_KM),
            provenance=self._provenance_by_id[airport.id],
            source_commit_sha=self._source_commit_by_id[airport.id],
            airport=airport,
        )

    def nearest(self, lat: float, lon: float) -> AirportMatch:
        query_point = np.radians([(lat, lon)])
        distances, indices = self._tree.query(query_point, k=1)
        return self._match(int(indices[0][0]), float(distances[0][0]))

    def within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float = 15.0,
    ) -> tuple[AirportMatch, ...]:
        query_point = np.radians([(lat, lon)])
        indices, distances = self._tree.query_radius(
            query_point,
            r=radius_km / EARTH_RADIUS_KM,
            return_distance=True,
            sort_results=True,
        )
        return tuple(
            self._match(int(index), float(distance))
            for index, distance in zip(indices[0], distances[0], strict=True)
        )

    def airports_within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float = 15.0,
    ) -> tuple[tuple[Airport, float], ...]:
        """Return airport/distance pairs without match provenance allocation."""
        query_point = np.radians([(lat, lon)])
        indices, distances = self._tree.query_radius(
            query_point,
            r=radius_km / EARTH_RADIUS_KM,
            return_distance=True,
            sort_results=True,
        )
        return tuple(
            (
                self._airports[int(index)],
                float(distance * EARTH_RADIUS_KM),
            )
            for index, distance in zip(indices[0], distances[0], strict=True)
        )
