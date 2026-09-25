from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime

import polars as pl
import pytest

from airports.catalog import AirportCatalogView, add_calendar_month
from airports.models import AirportMatchProvenance, AirportType


FLIGHT_DATE = date(2024, 1, 15)
KNOWLEDGE_CUTOFF = date(2024, 2, 15)


@pytest.mark.parametrize(
    ("flight_date", "expected"),
    [
        (date(2024, 1, 31), date(2024, 2, 29)),
        (date(2025, 1, 31), date(2025, 2, 28)),
        (date(2025, 12, 31), date(2026, 1, 31)),
    ],
)
def test_knowledge_window_is_one_calendar_month(flight_date, expected):
    assert add_calendar_month(flight_date) == expected


def airport_row(
    airport_id: int,
    ident: str,
    *,
    airport_type: str = "small_airport",
    lat: float = 0.0,
    lon: float = 0.0,
    iso_country: str = "US",
    source_commit_sha: str,
    iata_code: str | None = None,
    local_code: str | None = None,
) -> dict[str, object]:
    return {
        "airport_id": airport_id,
        "ident": ident,
        "type": airport_type,
        "name": f"Airport {ident}",
        "latitude_deg": lat,
        "longitude_deg": lon,
        "elevation_ft": None,
        "continent": "NA",
        "iso_country": iso_country,
        "iso_region": f"{iso_country}-X",
        "municipality": None,
        "scheduled_service": "no",
        "icao_code": ident if len(ident) == 4 else None,
        "iata_code": iata_code,
        "gps_code": ident if len(ident) == 4 else None,
        "local_code": local_code,
        "home_link": None,
        "wikipedia_link": None,
        "keywords": None,
        "source_commit_sha": source_commit_sha,
    }


class History:
    def __init__(
        self,
        states: dict[date, list[dict[str, object]]],
        aliases: list[dict[str, object]] | None = None,
    ) -> None:
        self.states = states
        self.aliases = aliases or []

    def as_of(self, observed_on: date) -> pl.DataFrame:
        return pl.DataFrame(self.states[observed_on])

    def source_commit_sha(self, observed_on: date) -> str:
        return f"snapshot-{observed_on.isoformat()}"

    def aliases_through(
        self,
        start_date: date,
        cutoff_date: date,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            self.aliases,
            schema={
                "airport_id": pl.Int64,
                "code": pl.String,
                "code_type": pl.String,
            },
        )


def make_view(
    observed: list[dict[str, object]],
    knowledge: list[dict[str, object]],
) -> AirportCatalogView:
    return AirportCatalogView.from_history(
        History({FLIGHT_DATE: observed, KNOWLEDGE_CUTOFF: knowledge}),
        FLIGHT_DATE,
        KNOWLEDGE_CUTOFF,
    )


def test_airport_active_on_flight_date_survives_closure_by_cutoff():
    observed = airport_row(
        1,
        "KOLD",
        lat=40.0,
        lon=-75.0,
        source_commit_sha="flight-commit",
    )
    closed = airport_row(
        1,
        "KOLD",
        airport_type="closed",
        lat=40.0,
        lon=-75.0,
        source_commit_sha="knowledge-commit",
    )

    view = make_view([observed], [closed])
    match = view.matcher.nearest(40.0, -75.0)

    assert match.id == 1
    assert match.airport.type is AirportType.SMALL_AIRPORT
    assert match.provenance is (
        AirportMatchProvenance.ACTIVE_ON_FLIGHT_DATE_CLOSED_BY_CUTOFF
    )
    assert match.source_commit_sha == "flight-commit"


def test_airport_learned_during_grace_period_is_a_spatial_candidate():
    existing = airport_row(
        1,
        "KAAA",
        lat=30.0,
        lon=-80.0,
        source_commit_sha="flight-commit",
    )
    discovered = airport_row(
        2,
        "KNEW",
        lat=45.0,
        lon=-70.0,
        source_commit_sha="knowledge-commit",
    )

    view = make_view([existing], [existing | {"source_commit_sha": "knowledge-commit"}, discovered])
    match = view.matcher.nearest(45.0, -70.0)

    assert match.id == 2
    assert match.provenance is AirportMatchProvenance.LEARNED_DURING_GRACE_PERIOD
    assert match.source_commit_sha == "knowledge-commit"


def test_changed_ident_is_an_alias_for_the_same_stable_airport_id():
    old = airport_row(
        10,
        "KOLD",
        local_code="OLD",
        source_commit_sha="flight-commit",
    )
    new = airport_row(
        10,
        "KNEW",
        local_code="NEW",
        source_commit_sha="knowledge-commit",
    )

    view = make_view([old], [new])

    assert view.get_airport_by_id(10) is view.get_airport_by_ident("KOLD")
    assert view.get_airport_by_ident("KNEW") is view.get_airport_by_id(10)
    assert view.resolve_code("OLD").id == 10
    assert view.resolve_code("NEW").id == 10
    assert replace(view.get_airport_by_id(10), ident="ANYTHING") == view.get_airport_by_id(10)


def test_transient_alias_during_grace_period_resolves_to_stable_id():
    old = airport_row(10, "KOLD", source_commit_sha="flight-commit")
    new = airport_row(10, "KNEW", source_commit_sha="knowledge-commit")
    history = History(
        {FLIGHT_DATE: [old], KNOWLEDGE_CUTOFF: [new]},
        aliases=[
            {"airport_id": 10, "code": "KTMP", "code_type": "ident"},
            {"airport_id": 10, "code": "TMP", "code_type": "iata_code"},
        ],
    )

    view = AirportCatalogView.from_history(
        history,
        FLIGHT_DATE,
        KNOWLEDGE_CUTOFF,
    )

    assert view.get_airport_by_ident("KTMP").id == 10
    assert view.resolve_code("TMP", code_priority=("iata_code",)).id == 10


def test_flight_day_code_owner_wins_over_grace_period_reassignment():
    original = airport_row(
        1,
        "KONE",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    reassigned = airport_row(
        2,
        "KTWO",
        local_code="DUP",
        source_commit_sha="knowledge-commit",
    )
    history = History(
        {
            FLIGHT_DATE: [original],
            KNOWLEDGE_CUTOFF: [
                original
                | {"local_code": "ONE", "source_commit_sha": "knowledge-commit"},
                reassigned,
            ],
        },
        aliases=[
            {
                "airport_id": 2,
                "code": "DUP",
                "code_type": "local_code",
                "observed_at": datetime(2024, 2, 1),
            }
        ],
    )

    view = AirportCatalogView.from_history(
        history,
        FLIGHT_DATE,
        KNOWLEDGE_CUTOFF,
    )

    assert view.resolve_code("DUP").id == 1


def test_closed_airports_resolve_directly_but_are_not_spatial_candidates():
    closed = airport_row(
        1,
        "KCLO",
        airport_type="closed",
        lat=0.0,
        lon=0.0,
        source_commit_sha="flight-commit",
    )
    open_airport = airport_row(
        2,
        "KOPEN",
        lat=1.0,
        lon=1.0,
        source_commit_sha="flight-commit",
    )
    knowledge = [
        closed | {"source_commit_sha": "knowledge-commit"},
        open_airport | {"source_commit_sha": "knowledge-commit"},
    ]

    view = make_view([closed, open_airport], knowledge)

    assert view.resolve_code("KCLO").type is AirportType.CLOSED
    assert view.matcher.nearest(0.0, 0.0).id == 2


def test_code_resolution_prefers_active_airport_over_closed_airport():
    closed = airport_row(
        1,
        "KCLO",
        airport_type="closed",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    active = airport_row(
        2,
        "KACT",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    observed = [closed, active]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]

    view = make_view(observed, knowledge)

    assert view.resolve_code("DUP").id == 2


def test_nearest_and_radius_return_immutable_matches_sorted_by_distance():
    first = airport_row(
        1,
        "KONE",
        lat=40.0,
        lon=-75.0,
        source_commit_sha="flight-commit",
    )
    second = airport_row(
        2,
        "KTWO",
        lat=40.1,
        lon=-75.0,
        source_commit_sha="flight-commit",
    )
    observed = [first, second]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]
    view = make_view(observed, knowledge)

    assert view.matcher.nearest(40.01, -75.0).id == 1
    matches = view.matcher.within_radius(40.01, -75.0, radius_km=20.0)
    airport_distances = view.matcher.airports_within_radius(
        40.01,
        -75.0,
        radius_km=20.0,
    )

    assert [match.id for match in matches] == [1, 2]
    assert matches[0].distance_km < matches[1].distance_km
    assert [airport.id for airport, _ in airport_distances] == [1, 2]
    assert [distance for _, distance in airport_distances] == pytest.approx(
        [match.distance_km for match in matches]
    )
    with pytest.raises(FrozenInstanceError):
        matches[0].distance_km = 100.0


def test_all_ourairports_types_are_retained_but_only_fixed_wing_are_matched():
    type_names = [airport_type.value for airport_type in AirportType]
    observed = [
        airport_row(
            airport_id,
            f"A{airport_id:03}",
            airport_type=airport_type,
            lat=float(airport_id),
            source_commit_sha="flight-commit",
        )
        for airport_id, airport_type in enumerate(type_names, start=1)
    ]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]

    view = make_view(observed, knowledge)

    assert {airport.type.value for airport in view.airports} == set(type_names)
    assert len(view.matcher.within_radius(2.0, 0.0, radius_km=1.0)) == 1


def test_code_collisions_use_faa_priority_then_location():
    canadian = airport_row(
        1,
        "CAAA",
        lat=45.0,
        lon=-75.0,
        iso_country="CA",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    american = airport_row(
        2,
        "KAAA",
        lat=35.0,
        lon=-80.0,
        iso_country="US",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    observed = [canadian, american]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]
    view = make_view(observed, knowledge)

    assert view.resolve_code("DUP").id == 2
    assert (
        view.resolve_code(
            "DUP",
            lat=45.0,
            lon=-75.0,
            code_priority=(),
            prioritize_faa_airspace=False,
        ).id
        == 1
    )


def test_explicit_code_type_priority_precedes_faa_airspace_priority():
    canadian_iata = airport_row(
        1,
        "CYZZ",
        iso_country="CA",
        iata_code="DUP",
        source_commit_sha="flight-commit",
    )
    american_local = airport_row(
        2,
        "KDUP",
        iso_country="US",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    observed = [canadian_iata, american_local]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]
    view = make_view(observed, knowledge)

    assert view.resolve_code("DUP", code_priority=("iata_code",)).id == 1


def test_explicit_code_type_does_not_fall_back_to_another_namespace():
    local = airport_row(
        1,
        "KLOC",
        local_code="ONLY",
        source_commit_sha="flight-commit",
    )
    knowledge = local | {"source_commit_sha": "knowledge-commit"}
    view = make_view([local], [knowledge])

    assert view.resolve_code("ONLY", code_priority=("iata_code",)) is None


def test_default_code_priority_includes_iata_codes():
    iata = airport_row(
        1,
        "KONE",
        iata_code="ONE",
        source_commit_sha="flight-commit",
    )
    knowledge = iata | {"source_commit_sha": "knowledge-commit"}
    view = make_view([iata], [knowledge])

    assert view.resolve_code("ONE").id == 1


def test_empty_code_type_priority_resolves_across_all_namespaces():
    local = airport_row(
        1,
        "KLOC",
        local_code="ONLY",
        source_commit_sha="flight-commit",
    )
    knowledge = local | {"source_commit_sha": "knowledge-commit"}
    view = make_view([local], [knowledge])

    assert view.resolve_code("ONLY", code_priority=()).id == 1


def test_explicit_code_type_can_resolve_closed_airport():
    closed_iata = airport_row(
        1,
        "KCLO",
        airport_type="closed",
        iata_code="DUP",
        source_commit_sha="flight-commit",
    )
    active_local = airport_row(
        2,
        "KACT",
        local_code="DUP",
        source_commit_sha="flight-commit",
    )
    observed = [closed_iata, active_local]
    knowledge = [row | {"source_commit_sha": "knowledge-commit"} for row in observed]
    view = make_view(observed, knowledge)

    assert view.resolve_code("DUP", code_priority=("iata_code",)).id == 1


def test_recent_flight_uses_available_catalog_date_as_incomplete_cutoff():
    flight_date = date(2026, 9, 10)
    available_date = date(2026, 9, 24)
    row = airport_row(1, "KAAA", source_commit_sha="flight-commit")
    latest = row | {"source_commit_sha": "latest-commit"}
    view = AirportCatalogView.from_history(
        History({flight_date: [row], available_date: [latest]}),
        flight_date,
        available_date,
    )

    assert view.provenance.knowledge_cutoff_date == available_date
    assert view.provenance.knowledge_commit_sha == "snapshot-2026-09-24"
    assert not view.provenance.knowledge_complete
