from dataclasses import dataclass
from datetime import date
from pathlib import Path

from airports import runtime


@dataclass(frozen=True)
class FakeHistory:
    paths: Path
    latest_available_date: date
    latest_available_commit: str


class FakeAirportHistory:
    current: FakeHistory

    @classmethod
    def open(cls) -> FakeHistory:
        return cls.current


class FakeCatalogView:
    calls: list[tuple[FakeHistory, date, date]] = []

    @classmethod
    def from_history(
        cls,
        history: FakeHistory,
        flight_date: date,
        catalog_available_date: date,
    ) -> object:
        cls.calls.append((history, flight_date, catalog_available_date))
        return object()


def test_catalog_cache_is_keyed_by_refreshed_history_generation(
    monkeypatch,
    tmp_path,
):
    flight_date = date(2026, 9, 20)
    history_path = tmp_path / "history"
    first_history = FakeHistory(
        history_path,
        date(2026, 9, 24),
        "airport-commit-1",
    )
    FakeAirportHistory.current = first_history
    FakeCatalogView.calls = []
    runtime._airport_catalog_for_history_generation.cache_clear()
    monkeypatch.setattr(runtime, "AirportHistory", FakeAirportHistory)
    monkeypatch.setattr(runtime, "AirportCatalogView", FakeCatalogView)

    first = runtime.airport_catalog_for_date(flight_date)
    assert runtime.airport_catalog_for_date(flight_date) is first
    assert len(FakeCatalogView.calls) == 1

    FakeAirportHistory.current = FakeHistory(
        history_path,
        date(2026, 9, 24),
        "airport-commit-2",
    )
    second = runtime.airport_catalog_for_date(flight_date)
    assert second is not first

    FakeAirportHistory.current = FakeHistory(
        history_path,
        date(2026, 9, 25),
        "airport-commit-2",
    )
    third = runtime.airport_catalog_for_date(flight_date)
    assert third is not second
    assert len(FakeCatalogView.calls) == 3

    runtime._airport_catalog_for_history_generation.cache_clear()
