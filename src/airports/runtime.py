from datetime import date
from functools import lru_cache

from airports.catalog import AirportCatalogView
from airports.history import AirportHistory


@lru_cache(maxsize=2)
def _airport_catalog_for_history_generation(
    flight_date: date,
    history: AirportHistory,
    catalog_available_date: date,
    _latest_available_commit: str,
) -> AirportCatalogView:
    return AirportCatalogView.from_history(
        history,
        flight_date=flight_date,
        catalog_available_date=catalog_available_date,
    )


def airport_catalog_for_date(flight_date: date) -> AirportCatalogView:
    """Return the current immutable airport catalog for one flight day."""
    history = AirportHistory.open()
    return _airport_catalog_for_history_generation(
        flight_date,
        history,
        history.latest_available_date,
        history.latest_available_commit,
    )


def ensure_airport_history() -> AirportHistory:
    """Build the local temporal catalog when absent, then return it."""
    return AirportHistory.open()
