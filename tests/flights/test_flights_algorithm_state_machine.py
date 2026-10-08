from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import polars as pl

from airports.models import (
    Airport,
    AirportMatch,
    AirportMatchProvenance,
    AirportType,
)
from flights.segmentation import algorithm
from flights.segmentation.algorithm import (
    AdsbPositionMessage,
    _segments_to_flights_df,
    identify_flight_segments,
)
from flights.segmentation.readsb_legs import mark_leg_indexes


ICAO = "abc123"
START = datetime(2026, 3, 1)


@dataclass(frozen=True)
class FakeAirportMatcher:
    airport: Airport

    def _match(self, lat: float, lon: float) -> AirportMatch:
        return AirportMatch(
            id=self.airport.id,
            ident=self.airport.ident,
            distance_km=self.airport.distance_to(lat, lon),
            provenance=AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE,
            source_commit_sha="test-commit",
            airport=self.airport,
        )

    def nearest(self, lat: float, lon: float) -> AirportMatch:
        return self._match(lat, lon)

    def within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> tuple[AirportMatch, ...]:
        return (self._match(lat, lon),)


@dataclass(frozen=True)
class FakeAirportCatalog:
    airport: Airport
    matcher: FakeAirportMatcher

    @property
    def id_index(self) -> dict[int, Airport]:
        return {self.airport.id: self.airport}

    def get_airport_by_ident(self, ident: str) -> Airport | None:
        return self.airport if ident == self.airport.ident else None

    def get_airport_by_id(self, airport_id: int) -> Airport | None:
        return self.airport if airport_id == self.airport.id else None


def fake_airport_catalog() -> FakeAirportCatalog:
    airport = Airport(
        id=101,
        ident="TEST",
        type=AirportType.SMALL_AIRPORT,
        name="Test Airport",
        lat=0.0,
        lon=0.0,
        elevation_ft=100,
    )
    return FakeAirportCatalog(airport, FakeAirportMatcher(airport))


def message(
    minutes: int,
    altitude: float | None,
    *,
    on_ground: bool | None = False,
    geom_altitude: int | None = None,
    lon: float | None = None,
    speed: float | None = None,
    callsign: str | None = None,
    registration: str | None = None,
) -> AdsbPositionMessage:
    return AdsbPositionMessage(
        time=START + timedelta(minutes=minutes),
        icao=ICAO,
        lat=0.0,
        lon=minutes * 0.1 if lon is None else lon,
        ground_speed_kt=(0.0 if on_ground else 200.0) if speed is None else speed,
        on_ground=on_ground,
        baro_altitude_ft=altitude,
        geom_altitude_ft=geom_altitude,
        callsign=callsign,
        registration=registration,
    )


def two_flight_trace() -> list[AdsbPositionMessage]:
    altitudes = [
        0,
        0,
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        30_000,
        30_000,
        25_000,
        18_000,
        10_000,
        5_000,
        1_000,
        0,
        0,
        0,
        0,
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        30_000,
    ]
    return [
        message(index, altitude, on_ground=altitude == 0)
        for index, altitude in enumerate(altitudes)
    ]


def test_processes_each_aircraft_group_before_requesting_the_next(monkeypatch):
    processed = []

    class Groups:
        def __iter__(self):
            yield ("aaa111",), "first"
            assert processed == ["first"]
            yield ("bbb222",), "second"

    class Frame:
        def filter(self, expression):
            return self

        def sort(self, columns):
            return self

        def group_by(self, column, maintain_order):
            return Groups()

    monkeypatch.setattr(
        algorithm,
        "_rows_to_position_messages",
        lambda frame: [frame],
    )
    monkeypatch.setattr(
        algorithm,
        "identify_flight_segments",
        lambda messages: processed.append(messages[0]) or [],
    )

    assert algorithm._segment_adsb_df(Frame()) == []
    assert processed == ["first", "second"]


def test_complete_aircraft_batches_match_one_shot_segmentation():
    airport_catalog = fake_airport_catalog()
    messages = two_flight_trace()
    messages += [replace(message, icao="def456") for message in two_flight_trace()]
    frame = pl.from_dicts([message.__dict__ for message in messages])

    expected = algorithm.create_flights_from_adsb(
        frame,
        START.date(),
        airport_catalog,
    )
    actual = pl.concat(
        [
            algorithm.create_flights_from_adsb(
                frame.filter(pl.col("icao") == icao),
                START.date(),
                airport_catalog,
            )
            for icao in (ICAO, "def456")
        ]
    )

    assert actual.equals(expected)


def test_algorithm_source_stitches_a_flight_across_utc_midnight(monkeypatch):
    from data_engineering.adsb.read_adsb import ADSBSource
    from flights import flight_source
    from flights.algorithm import AlgorithmNearestFlightSource

    target_date = START.date()
    flight_start = START + timedelta(hours=23, minutes=55)
    messages = [
        replace(
            message(index, altitude, on_ground=False),
            time=flight_start + timedelta(minutes=index),
        )
        for index, altitude in enumerate(
            [
                1_000,
                5_000,
                10_000,
                20_000,
                30_000,
                30_000,
                25_000,
                18_000,
                10_000,
                5_000,
                1_000,
                500,
            ]
        )
    ]
    frame = pl.from_dicts([item.__dict__ for item in messages])
    reads = []

    def read_adsb(
        start,
        end,
        *,
        source,
        columns=None,
        lazy=False,
        icaos=None,
    ):
        reads.append((start, end, source, lazy, icaos))
        return frame.lazy() if lazy else frame

    monkeypatch.setattr(flight_source, "read_adsb", read_adsb)

    flights = flight_source.get_flights_for_catalog(
        AlgorithmNearestFlightSource(),
        target_date,
        fake_airport_catalog(),
        ADSBSource.ADSBLOL,
        use_cache=False,
    )

    assert flights.height == 1
    assert flights.select("takeoff_time", "landing_time").row(0) == (
        flight_start,
        flight_start + timedelta(minutes=11),
    )
    assert len(reads) == 2
    assert all(
        start == START - timedelta(days=1)
        and end == START + timedelta(days=2)
        and source is ADSBSource.ADSBLOL
        for start, end, source, _, _ in reads
    )


def test_combined_segment_counts_match_individual_collectors():
    messages = [
        replace(
            item,
            pia=index % 2 == 0,
            callsign=("FIRST" if index < 3 else "SECOND"),
            registration=("N1" if index % 3 else "N2"),
            owner=("Owner A" if index < 4 else "Owner B"),
        )
        for index, item in enumerate(two_flight_trace()[:8])
    ]

    assert algorithm._segment_counts_for_messages(messages) == (
        algorithm._db_flags_counts_for_messages(messages),
        algorithm._flight_identity_counts_for_messages(messages),
        algorithm._aircraft_metadata_counts_for_messages(messages),
    )


def test_airport_assignment_does_not_decide_if_a_flight_exists(monkeypatch):
    messages = two_flight_trace()[:15]
    frame = pl.from_dicts([message.__dict__ for message in messages])
    same_airport_catalog = object()
    different_airports_catalog = object()

    def airport_guesses(segments, airport_catalog):
        assert len(segments) == 1
        takeoff = algorithm.AirportGuess(1, "A", 0.0)
        landing = (
            takeoff
            if airport_catalog is same_airport_catalog
            else algorithm.AirportGuess(2, "B", 0.0)
        )
        return [takeoff], [landing]

    monkeypatch.setattr(
        algorithm,
        "_airport_guesses_for_segments",
        airport_guesses,
    )

    same_airport = algorithm.create_flights_from_adsb(
        frame,
        START.date(),
        same_airport_catalog,
    )
    different_airports = algorithm.create_flights_from_adsb(
        frame,
        START.date(),
        different_airports_catalog,
    )

    airport_columns = [
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ]
    assert same_airport.height == different_airports.height == 1
    assert same_airport.drop(airport_columns).equals(
        different_airports.drop(airport_columns)
    )
    assert same_airport.select(
        "takeoff_airport_id",
        "landing_airport_id",
    ).row(0) == (1, 1)
    assert different_airports.select(
        "takeoff_airport_id",
        "landing_airport_id",
    ).row(0) == (1, 2)


def test_marks_new_leg_during_ground_interval_between_descent_and_climb():
    messages = two_flight_trace()

    assert mark_leg_indexes(messages) == [16]


def test_converts_readsb_leg_markers_to_existing_flight_segments():
    segments = identify_flight_segments(two_flight_trace())

    assert len(segments) == 2
    assert segments[0].takeoff_time == START + timedelta(minutes=2)
    assert segments[0].landing_time == START + timedelta(minutes=14)
    assert segments[1].takeoff_time == START + timedelta(minutes=18)
    assert segments[1].landing_time == START + timedelta(minutes=23)


def test_uses_ground_air_transitions_instead_of_day_and_leg_boundaries():
    messages = two_flight_trace()
    messages[0] = message(0, 0, on_ground=True)
    messages[1] = message(1, 0, on_ground=True)

    segments = identify_flight_segments(messages)

    assert segments[0].takeoff_message.on_ground is False
    assert segments[0].landing_message.on_ground is True
    assert segments[1].takeoff_message.on_ground is False


def test_falls_back_to_trace_boundaries_when_ground_state_is_missing():
    messages = [
        message(index, altitude, on_ground=None)
        for index, altitude in enumerate(
            [1_000, 5_000, 10_000, 20_000, 30_000, 30_000]
        )
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == START
    assert segments[0].landing_time == START + timedelta(minutes=5)


def test_ignores_isolated_airborne_status_blips_while_parked():
    messages = [
        message(0, None, on_ground=True, lon=0.0, speed=0.1),
        message(1, None, on_ground=False, lon=0.0, speed=0.1),
        message(2, None, on_ground=True, lon=0.0, speed=0.1),
        message(3, None, on_ground=True, lon=0.0, speed=0.1),
        message(4, 0, on_ground=False, lon=0.0, speed=87),
        message(5, 1_000, on_ground=False, lon=0.1, speed=150),
        message(6, 5_000, on_ground=False, lon=0.2, speed=200),
        message(10, None, on_ground=True, lon=0.3, speed=80),
        message(11, None, on_ground=False, lon=0.3, speed=0.1),
        message(12, None, on_ground=True, lon=0.3, speed=0.1),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == START + timedelta(minutes=4)
    assert segments[0].landing_time == START + timedelta(minutes=10)


def test_rejects_all_ground_slice_despite_duration_and_accumulated_movement():
    messages = [
        message(
            index * 2,
            None,
            on_ground=True,
            lon=index * 0.012,
            speed=15,
        )
        for index in range(6)
    ]

    assert identify_flight_segments(messages) == []


def test_readsb_does_not_mark_traces_shorter_than_twenty_points():
    messages = two_flight_trace()[:19]

    assert mark_leg_indexes(messages) == []
    assert len(identify_flight_segments(messages)) == 1


def test_uses_geometric_altitude_when_baro_altitude_is_missing():
    messages = [
        message(
            index,
            None,
            on_ground=source.baro_altitude_ft == 0,
            geom_altitude=int(source.baro_altitude_ft or 0),
        )
        for index, source in enumerate(two_flight_trace())
    ]

    assert mark_leg_indexes(messages) == [16]


def test_flight_identity_uses_the_mode_of_messages_in_its_sequence():
    messages = [
        message(
            index,
            altitude,
            on_ground=altitude == 0,
            callsign="TEST123" if index != 4 else "OTHER456",
            registration="N123AB" if index != 5 else "N999ZZ",
        )
        for index, altitude in enumerate(
            [0, 0, 1_000, 5_000, 10_000, 10_000, 5_000, 0, 0]
        )
    ]

    segments = identify_flight_segments(messages)
    flights = _segments_to_flights_df(segments, fake_airport_catalog())

    assert flights.select("callsign").item() == "TEST123"
    assert flights.select("registration").item() == "N123AB"
    assert flights.select(
        "takeoff_airport_id",
        "landing_airport_id",
    ).row(0) == (101, 101)


def test_default_worker_count_is_capped_at_eight(monkeypatch):
    monkeypatch.setattr(algorithm, "default_worker_count", lambda: 32)

    assert algorithm._number_of_workers_default() == 8
